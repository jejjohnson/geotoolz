"""The shared HDF5 / geostationary reader core, used outside GOES.

A reader for another product only subclasses ``PackedGridReader`` and
supplies its grid: this file builds a minimal one over a plain lat/lon
NetCDF-4 grid (no ABI conventions) to pin that contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window


h5py = pytest.importorskip("h5py")

from geoproducts import ProductReader, stack
from geoproducts._src.base import Track
from geoproducts._src.geostationary import geos_crs, scan_angle_transform
from geoproducts._src.hdf import PackedGridReader, scalar


class LatLonReader(PackedGridReader):
    """A toy NetCDF reader: ``(lat, lon)`` variables on a 0.1° grid."""

    extra = "toy"
    feature = "LatLonReader"

    def __init__(self, path: Path, variables: list[str]) -> None:
        self._source = path
        with self._open() as f:
            self._shape2d = tuple(int(n) for n in f["T"].shape)
            self._load_variables(f, variables)

    @property
    def _grid_shape(self) -> tuple[int, int]:
        return self._shape2d  # type: ignore[return-value]

    @property
    def _crs(self) -> CRS:
        return CRS.from_epsg(4326)

    @property
    def _transform(self) -> Affine:
        return Affine(0.1, 0.0, 10.0, 0.0, -0.1, 50.0)

    @property
    def _track(self) -> Track:
        return "A"

    @property
    def _bands(self) -> tuple[str, ...]:
        return self.variables


@pytest.fixture
def latlon_file(tmp_path: Path) -> Path:
    path = tmp_path / "grid.nc"
    with h5py.File(path, "w") as f:
        t = f.create_dataset("T", data=np.arange(12, dtype=np.int16).reshape(3, 4))
        t.attrs["scale_factor"] = np.float32(0.5)
        t.attrs["add_offset"] = np.float32(270.0)
        t.attrs["_FillValue"] = np.int16(0)
        t.attrs["units"] = np.bytes_(b"K")
        f.create_dataset("cls", data=np.full((3, 4), 2, dtype=np.uint8))
        f.create_dataset("scale", data=np.float32(-1.0))
        f["scale"].attrs["_FillValue"] = np.float32(-1.0)
    return path


def test_a_non_abi_reader_gets_the_full_contract(latlon_file: Path) -> None:
    reader = LatLonReader(latlon_file, ["T"])
    assert isinstance(reader, ProductReader)
    out = reader.load()
    assert out.shape == (1, 3, 4)
    assert out.attrs == {"band_names": ("T",), "units": ("K",)}
    values = np.asarray(out)[0]
    assert np.isnan(values[0, 0])  # stored 0 is the fill
    assert values[0, 1] == pytest.approx(270.5)
    pad = np.asarray(reader.read_from_window(Window(3, 2, 2, 2)))
    assert np.isnan(pad[0, :, 1]).all() and np.isnan(pad[0, 1, :]).all()


def test_integer_bands_stay_integer_and_stack(latlon_file: Path) -> None:
    classes = LatLonReader(latlon_file, ["cls"])
    assert classes.dtype == np.uint8
    out = stack([LatLonReader(latlon_file, ["T"]), classes])
    assert out.attrs["band_names"] == ("T", "cls")
    assert out.dtype == np.float32


def test_missing_variable_lists_the_grid_variables(latlon_file: Path) -> None:
    with pytest.raises(ValueError, match=r"no variable 'Q'.*'T', 'cls'"):
        LatLonReader(latlon_file, ["Q"])


def test_scalar_uses_the_declared_fill(latlon_file: Path) -> None:
    with h5py.File(latlon_file) as f:
        assert np.isnan(scalar(f, "scale"))
        assert np.isnan(scalar(f, "absent"))


def test_geostationary_helpers() -> None:
    t = scan_angle_transform(np.array([0.0, 1e-4, 2e-4]), np.array([1e-4, 0.0]), 1e6)
    assert t.a == pytest.approx(100.0)
    assert t.e == pytest.approx(-100.0)
    assert t.c == pytest.approx(-50.0)
    assert t.f == pytest.approx(150.0)

    def crs(sweep: str) -> CRS:
        return geos_crs(
            lon_0=0.0,
            height_m=35786400.0,
            semi_major_m=6378137.0,
            semi_minor_m=6356752.3,
            sweep=sweep,
        )

    # ``sweep=y`` is GDAL's own geostationary convention; ``x`` (ABI) is
    # carried as an explicit PROJ extension.
    assert "+sweep=x" in crs("x").to_wkt()
    assert "+sweep" not in crs("y").to_wkt()
    assert crs("x") != crs("y")
