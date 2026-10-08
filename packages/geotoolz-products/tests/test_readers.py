"""Tests for product reader framework."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

import geoproducts
from geoproducts import ProductReader, toy_sensor
from geoproducts.toy_sensor import constants as toy_constants


class MissingReader(ProductReader):
    """Incomplete reader used to verify ABC enforcement."""


def test_public_surface() -> None:
    assert set(geoproducts.__all__) == {
        "ProductReader",
        "__version__",
        "goes",
        "himawari",
        "stack",
        "toy_sensor",
    }
    assert geoproducts.ProductReader is ProductReader
    assert toy_sensor.Reader is not None


def test_product_reader_abc_enforces_required_surface() -> None:
    with pytest.raises(TypeError):
        MissingReader()  # type: ignore[abstract]


def test_toy_sensor_reader_passes_geodata_conformance() -> None:
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    reader = toy_sensor.Reader(
        "synthetic-toy",
        data=data,
        transform=Affine.translation(100, 200) * Affine.scale(10, -10),
        crs="EPSG:32631",
        fill_value_default=-9999.0,
    )

    assert isinstance(reader, ProductReader)
    assert reader.track == "A"
    assert reader.shape == data.shape
    assert reader.bands == ("blue", "green", "red", "nir")

    tile = reader.read_from_window(Window(1, 2, 3, 2))
    assert isinstance(tile, GeoTensor)
    np.testing.assert_array_equal(tile.values, data[:, 2:4, 1:4])
    assert tile.attrs["band_names"] == reader.bands

    boundless = reader.read_from_center_coords(95, 205, width=3, height=3)
    assert boundless.shape == (4, 3, 3)
    assert np.all(boundless.values[:, 0, 0] == -9999.0)


def test_center_coords_matches_georeader() -> None:
    from georeader import read

    data = np.arange(4 * 10 * 10, dtype=np.float32).reshape(4, 10, 10)
    reader = toy_sensor.Reader(
        "synthetic-toy",
        data=data,
        transform=Affine.translation(0, 100) * Affine.scale(10, -10),
        crs="EPSG:32631",
        fill_value_default=-9999.0,
    )

    ours = reader.read_from_center_coords(35, 45, width=4, height=4)
    ref = read.read_from_center_coords(reader, (35, 45), (4, 4))
    assert ours.transform == ref.transform
    np.testing.assert_array_equal(ours.values, ref.values)

    bounds = (12.0, 31.0, 58.0, 77.0)
    ours_b = reader.read_from_bounds(bounds)
    ref_b = read.read_from_bounds(reader, bounds)
    assert ours_b.transform == ref_b.transform
    np.testing.assert_array_equal(ours_b.values, ref_b.values)


class _FourDReader(ProductReader):
    """Minimal ``(time, band, y, x)`` reader for dims checks."""

    def __init__(self, shape: tuple[int, ...] = (2, 3, 4, 5)) -> None:
        self._array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    def _read_window(self, window: Window) -> np.ndarray:
        r0, c0 = int(window.row_off), int(window.col_off)
        return self._array[
            ..., r0 : r0 + int(window.height), c0 : c0 + int(window.width)
        ]

    _crs = "EPSG:4326"  # type: ignore[assignment]
    _transform = Affine.identity()  # type: ignore[assignment]
    _dtype = np.dtype("float32")  # type: ignore[assignment]
    _bands = ("a", "b", "c")  # type: ignore[assignment]
    _fill_value = 0.0  # type: ignore[assignment]
    _track = "A"  # type: ignore[assignment]

    @property
    def _shape(self) -> tuple[int, ...]:
        return self._array.shape


def test_dims_4d() -> None:
    reader = _FourDReader()
    assert tuple(reader.dims) == ("time", "band", "y", "x")
    assert reader.width == 5
    assert reader.height == 4

    tile = reader.read_from_window(Window(1, 1, 2, 2))
    assert tile.shape == (2, 3, 2, 2)
    assert tuple(tile.dims) == tuple(reader.dims)

    assert tuple(_FourDReader((4, 5)).dims) == ("y", "x")
    assert tuple(_FourDReader((3, 4, 5)).dims) == ("band", "y", "x")
    with pytest.raises(ValueError, match="2d-4d"):
        _ = _FourDReader((1, 2, 3, 4, 5)).dims


def test_toy_sensor_fill_default_matches_dtype() -> None:
    ints = np.ones((4, 2, 2), dtype=np.uint16)
    reader = toy_sensor.Reader("int-toy", data=ints)
    assert reader.fill_value_default == 0
    out = reader.read_from_window(Window(-1, -1, 3, 3))
    assert out.values[0, 0, 0] == reader.fill_value_default

    floats = np.ones((4, 2, 2), dtype=np.float32)
    assert np.isnan(toy_sensor.Reader("f-toy", data=floats).fill_value_default)
    assert (
        toy_sensor.Reader(
            "int-toy", data=ints, fill_value_default=65535
        ).fill_value_default
        == 65535
    )


@pytest.mark.parametrize("fill", [np.nan, 1.5, -1, 70000])
def test_toy_sensor_rejects_unrepresentable_fill(fill: float) -> None:
    ints = np.ones((4, 2, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="not representable"):
        toy_sensor.Reader("int-toy", data=ints, fill_value_default=fill)


def test_toy_sensor_constants_are_lazy_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_load_csv(package: str, resource: str) -> tuple[dict[str, str], ...]:
        calls.append(f"{package}:{resource}")
        return ({"name": "red"},)

    monkeypatch.setattr(toy_constants, "_CACHE", {})
    monkeypatch.setattr(toy_constants, "load_csv", fake_load_csv)

    assert calls == []
    assert toy_constants.BANDS == ({"name": "red"},)
    assert toy_constants.BANDS == ({"name": "red"},)
    assert toy_constants.CONSTANTS == {"solar_irradiance": ({"name": "red"},)}
    assert toy_constants.CONSTANTS == {"solar_irradiance": ({"name": "red"},)}
    assert calls == [
        "geoproducts.toy_sensor:data/bands.csv",
        "geoproducts.toy_sensor:data/solar_irradiance.csv",
    ]


def test_shared_csv_loader_caches_package_data() -> None:
    from geoproducts._src.constants import load_csv

    load_csv.cache_clear()
    bands = load_csv("geoproducts.toy_sensor", "data/bands.csv")
    assert bands[2]["name"] == "red"
    assert bands is load_csv("geoproducts.toy_sensor", "data/bands.csv")


def test_toy_sensor_ndvi_preset_matches_generic_operator() -> None:
    gz = pytest.importorskip("geotoolz")
    op = toy_sensor.NDVI()
    expected = gz.indices.NDVI(red="red", nir="nir")

    assert op.get_config() == expected.get_config()

    gt = GeoTensor(
        np.stack(
            [
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
                np.full((2, 2), 3.0, dtype=np.float32),
            ]
        ),
        transform=Affine.identity(),
        crs="EPSG:4326",
        attrs={"band_names": ("blue", "green", "red", "nir")},
    )
    np.testing.assert_allclose(op(gt).values, 0.5)


def test_ndvi_preset_without_geotoolz_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    # ``None`` in sys.modules makes any import of the name raise.
    for name in [m for m in sys.modules if m.split(".")[0] == "geotoolz"]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "geotoolz", None)
    with pytest.raises(ImportError, match=r"geotoolz-products\[operators\]"):
        toy_sensor.NDVI()


@pytest.mark.slow
def test_toy_sensor_package_data_is_in_wheel(tmp_path: Path) -> None:
    import shutil
    import subprocess

    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("building the wheel needs uv on PATH")
    package_dir = Path(__file__).resolve().parents[1]
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path), str(package_dir)],
        check=True,
        capture_output=True,
    )
    (wheel,) = tmp_path.glob("geotoolz_products-*.whl")
    with ZipFile(wheel) as zf:
        names = set(zf.namelist())

    assert "geoproducts/toy_sensor/data/bands.csv" in names
    assert "geoproducts/toy_sensor/data/solar_irradiance.csv" in names


def test_readers_write_only_band_names() -> None:
    """Readers label bands under ``band_names`` only (the key geotoolz reads)."""
    gt = toy_sensor.Reader("toy.tif").load()
    assert set(gt.attrs or {}) >= {"band_names"}
    assert "bands" not in (gt.attrs or {})
