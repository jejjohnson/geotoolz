"""``himawari.L2Reader`` on a synthetic NOAA L2 cloud-mask file."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
from rasterio.windows import Window

from geoproducts import himawari, stack
from geoproducts.himawari.l2 import full_disk_grid, product_code


NAME = "AHI-CMSK_v1r1_h09_s202610070300214_e202610070309408_c202610070314065.nc"
GRID = (5500, 5500)


def _write(path: Path, *, title: str = "AHI-CMSK", link: str = NAME) -> Path:
    """A 2 km full-disk file; only a 10 x 10 block holds data (chunked, so tiny)."""
    with h5py.File(path, "w") as f:
        f.attrs["title"] = np.bytes_(title.encode())
        f.attrs["Metadata_Link"] = np.bytes_(link.encode())
        f.attrs["satellite_name"] = np.bytes_(b"Himawari-9")
        f.attrs["time_coverage_start"] = np.bytes_(b"2026-10-07T03:00:21Z")
        f.attrs["time_coverage_end"] = np.bytes_(b"2026-10-07T03:09:40Z")

        def int8(name: str, block: np.ndarray, **attrs) -> None:
            var = f.create_dataset(
                name, shape=GRID, dtype="i1", chunks=(500, 500), fillvalue=-128
            )
            var[2750:2760, 2750:2760] = block
            var.attrs["_FillValue"] = np.int8(-128)
            var.attrs["units"] = np.bytes_(b"1")
            for key, value in attrs.items():
                var.attrs[key] = value

        mask = np.full((10, 10), 3, dtype=np.int8)
        mask[0, 0] = 0
        int8(
            "CloudMask",
            mask,
            flag_values=np.array([0, 1, 2, 3], dtype=np.int8),
            flag_meanings=np.bytes_(b"clear probably_clear probably_cloudy cloudy"),
        )
        int8("CloudMaskBinary", (mask >= 2).astype(np.int8))
        int8("CloudMaskBinaryAWIPS", (mask >= 2).astype(np.int8))
        prob = f.create_dataset(
            "CloudProbability",
            shape=GRID,
            dtype="f4",
            chunks=(500, 500),
            fillvalue=-999.0,
        )
        prob[2750:2760, 2750:2760] = 0.75
        prob.attrs["_FillValue"] = np.float32(-999.0)
        prob.attrs["units"] = np.bytes_(b"1")
        for geo in ("Latitude", "Longitude"):
            f.create_dataset(geo, shape=GRID, dtype="f4", chunks=(500, 500))
    return path


@pytest.fixture
def cmsk(tmp_path: Path) -> Path:
    return _write(tmp_path / NAME)


def test_cloud_mask_defaults_and_flags(cmsk: Path) -> None:
    reader = himawari.L2Reader(cmsk)
    assert reader.product == "CMSK"
    assert reader.satellite == "Himawari-9"
    assert reader.start_time == datetime(2026, 10, 7, 3, 0, 21, tzinfo=UTC)
    assert reader.end_time == datetime(2026, 10, 7, 3, 9, 40, tzinfo=UTC)
    assert reader.variables == ("CloudMaskBinary", "CloudMask")
    assert reader.shape == (2, 5500, 5500)
    assert reader.dtype == np.int8
    assert reader.fill_value_default == -128
    assert reader.available_variables == (
        "CloudMask",
        "CloudMaskBinary",
        "CloudProbability",
    )
    assert reader.flags("CloudMask")[3] == "cloudy"
    tile = reader.read_from_window(Window(2748, 2748, 4, 4))
    values = np.asarray(tile)
    assert values[0, 0, 0] == -128  # outside the data block
    assert values[1, 2, 2] == 0 and values[1, 3, 3] == 3
    assert tile.attrs["band_names"] == ("CloudMaskBinary", "CloudMask")
    assert "CMSK" in repr(reader)


def test_grid_is_the_l1b_full_disk(cmsk: Path) -> None:
    reader = himawari.L2Reader(cmsk)
    grid = full_disk_grid(2.0)
    assert reader.transform == grid.transform
    assert reader.crs == grid.crs
    assert reader.res == pytest.approx((2000.0, 2000.0), rel=1e-6)
    assert reader.satellite_lon_deg == 140.7
    assert full_disk_grid(0.5).width == 22_000
    with pytest.raises(KeyError):
        full_disk_grid(4.0)


def test_float_variable_decodes_with_nan(cmsk: Path) -> None:
    reader = himawari.L2Reader(cmsk, variables="CloudProbability")
    values = np.asarray(reader.read_from_window(Window(2749, 2749, 2, 2)))[0]
    assert reader.dtype == np.float32
    assert np.isnan(values[0, 0]) and values[1, 1] == pytest.approx(0.75)


def test_variable_errors(cmsk: Path) -> None:
    with pytest.raises(ValueError, match="no variable 'Nope'"):
        himawari.L2Reader(cmsk, variables=["Nope"])
    with pytest.raises(ValueError, match="at least one"):
        himawari.L2Reader(cmsk, variables=[])


def test_other_products_default_to_every_data_variable(tmp_path: Path) -> None:
    link = NAME.replace("CMSK", "CPHS")
    reader = himawari.L2Reader(_write(tmp_path / link, title="AHI-CPHS", link=link))
    assert reader.product == "CPHS"
    assert reader.variables == ("CloudMask", "CloudMaskBinary", "CloudProbability")
    assert reader.dtype == np.float32  # mixed bands decode to float


def test_product_code() -> None:
    assert product_code(NAME) == "CMSK"
    assert product_code("HS_H09_x.DAT") == ""


def test_stacks_onto_an_l1b_grid(cmsk: Path) -> None:
    mask = himawari.L2Reader(cmsk, variables="CloudMaskBinary")
    out = stack([mask, himawari.L2Reader(cmsk, variables="CloudProbability")])
    assert out.attrs["band_names"] == ("CloudMaskBinary", "CloudProbability")
    assert out.shape == (2, 5500, 5500)
