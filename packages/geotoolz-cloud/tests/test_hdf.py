"""`geocloud.hdf`: HDF5 / NetCDF reads from local paths and a mounted bucket."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from obstore.store import MemoryStore


h5py = pytest.importorskip("h5py")

from geocloud import files
from geocloud.hdf import (
    PackedVariable,
    open_hdf5,
    read_hdf,
    read_netcdf,
    select_indexes,
)
from geocloud.store import mount, unmount


ROOT = "s3://hdf-test"


@pytest.fixture
def bucket() -> Iterator[str]:
    mount(ROOT, MemoryStore())
    yield ROOT
    unmount(ROOT)


@pytest.fixture
def h5_file(tmp_path: Path) -> Path:
    path = tmp_path / "swath.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        data = f.create_dataset("radiance", data=rng.random((3, 20, 30)).astype("f4"))
        data.attrs["_FillValue"] = np.array([-9999.0], dtype="f4")
        f["lat"] = np.ones((20, 30), dtype="f4")
        f["lon"] = np.zeros((20, 30), dtype="f4")
        f.create_group("meta").attrs["sensor"] = b"toy"
    return path


def test_open_hdf5_reads_local_and_remote_files(h5_file: Path, bucket: str):
    files.upload(h5_file, f"{bucket}/swath.h5")
    with open_hdf5(h5_file) as local, open_hdf5(f"{bucket}/swath.h5") as remote:
        np.testing.assert_array_equal(
            remote["radiance"][1, :5, :5], local["radiance"][1, :5, :5]
        )
    with h5_file.open("rb") as fh, open_hdf5(fh) as f:  # an open file object
        assert f["radiance"].shape == (3, 20, 30)


def test_read_hdf_selects_bands_and_metadata(h5_file: Path, bucket: str):
    files.upload(h5_file, f"{bucket}/swath.h5")
    for source in (h5_file, f"{bucket}/swath.h5"):
        out = read_hdf(
            source,
            "radiance",
            indexes=[3, 1],
            geolocation=("lat", "lon"),
            metadata_groups=["meta"],
        )
        with h5py.File(h5_file) as f:
            expected = f["radiance"][[0, 2]][::-1]
        np.testing.assert_array_equal(np.asarray(out), expected)  # (2, 20, 30)
        assert out.fill_value_default == -9999.0
        assert out.attrs["metadata"] == {"meta": {"sensor": "toy"}}
        assert out.attrs["geolocation"]["latitude"].shape == (20, 30)


def test_read_hdf_rejects_other_files(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"not an hdf file")
    with pytest.raises(ValueError, match="neither an HDF5 nor an HDF4"):
        read_hdf(path, "x")


def test_read_netcdf_decodes_cf_and_georeferencing(tmp_path: Path, bucket: str):
    netcdf4 = pytest.importorskip("netCDF4")
    from pyproj import CRS

    path = tmp_path / "l2.nc"
    with netcdf4.Dataset(path, "w") as ds:
        product = ds.createGroup("PRODUCT")
        product.createDimension("y", 4)
        product.createDimension("x", 5)
        crs = product.createVariable("crs", "i4")
        crs.crs_wkt = CRS("EPSG:32610").to_wkt()
        crs.GeoTransform = "750000 10 0 4350000 0 -10"
        ch4 = product.createVariable("ch4", "i2", ("y", "x"), fill_value=-999)
        ch4.scale_factor, ch4.add_offset, ch4.grid_mapping = 0.1, 1800.0, "crs"
        mask = np.zeros((4, 5), dtype=bool)
        mask[0, 0] = True
        ch4[:] = np.ma.array(np.full((4, 5), 1900.0), mask=mask)
    files.upload(path, f"{bucket}/l2.nc")
    for source in (path, f"{bucket}/l2.nc"):
        out = read_netcdf(source, "ch4", group="PRODUCT")
        assert out.crs is not None and out.transform.a == 10
        assert np.isnan(out.fill_value_default)
        assert np.isnan(np.asarray(out)[0, 0])
        np.testing.assert_allclose(np.asarray(out)[1, 1], 1900.0)


def test_packed_variable_decodes_unsigned_and_fill(tmp_path: Path):
    path = tmp_path / "packed.nc"
    with h5py.File(path, "w") as f:
        var = f.create_dataset("t", data=np.array([[0, 100, -1]], dtype="i2"))
        var.attrs["_Unsigned"] = b"true"
        var.attrs["_FillValue"] = np.array([-1], dtype="i2")
        var.attrs["scale_factor"] = np.float32(0.5)
        var.attrs["add_offset"] = np.float32(200.0)
        var.attrs["units"] = b"K"
    with open_hdf5(path) as f:
        packed = PackedVariable.from_file(f, "t")
        decoded = packed.decode(f["t"][...])
    assert packed.storage == np.dtype("u2") and packed.fill == 65535
    np.testing.assert_array_equal(decoded[0, :2], [200.0, 250.0])
    assert np.isnan(decoded[0, 2]) and packed.units == "K"


def test_band_selection_rules():
    stack = np.arange(12).reshape(3, 2, 2)
    np.testing.assert_array_equal(select_indexes(stack, [2])[:, 0, 0], [4])
    with pytest.raises(ValueError, match="1-based"):
        select_indexes(stack, [0])
    with pytest.raises(ValueError, match="leading band axis"):
        select_indexes(np.zeros((2, 2)), [2])


def test_a_missing_backend_names_the_extra(h5_file: Path, monkeypatch):
    real = importlib.import_module

    def fake(name: str, package: str | None = None):
        if name == "h5py":
            raise ImportError("no h5py")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", fake)
    with pytest.raises(ImportError, match=r"geotoolz-cloud\[hdf5\]") as info:
        read_hdf(h5_file, "radiance")
    assert info.value.name == "h5py"


def _cf_netcdf(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    *,
    names: tuple[str, str] = ("y", "x"),
    geotransform: str | None = None,
    units: str = "m",
) -> Path:
    """A ``(y, x)`` float variable georeferenced by CF 1-D coordinates."""
    netcdf4 = pytest.importorskip("netCDF4")
    from pyproj import CRS

    y_name, x_name = names
    with netcdf4.Dataset(path, "w") as ds:
        ds.createDimension(y_name, y.size)
        ds.createDimension(x_name, x.size)
        projected = x_name == "x"
        for name, values in ((x_name, x), (y_name, y)):
            coord = ds.createVariable(name, "f8", (name,))
            coord[:] = values
            coord.units = units if projected else "degrees"
        crs = ds.createVariable("crs", "i4")
        crs.crs_wkt = CRS("EPSG:32630" if projected else "EPSG:4326").to_wkt()
        if geotransform is not None:
            crs.GeoTransform = geotransform
        var = ds.createVariable("t2m", "f4", (y_name, x_name))
        var[:] = np.arange(y.size * x.size, dtype="f4").reshape(y.size, x.size)
        var.grid_mapping = "crs"
    return path


@pytest.mark.parametrize(
    ("y", "expected_e", "expected_f"),
    [
        (4_200_000 - 10 * (np.arange(3) + 0.5), -10.0, 4_200_000.0),  # north-up
        (4_200_000 + 10 * (np.arange(3) + 0.5), 10.0, 4_200_000.0),  # south-up
    ],
)
def test_read_netcdf_transform_from_cf_coordinates(
    tmp_path: Path, y: np.ndarray, expected_e: float, expected_f: float
):
    from affine import Affine

    x = 500_000 + 10 * (np.arange(4) + 0.5)
    out = read_netcdf(_cf_netcdf(tmp_path / "cf.nc", x, y), "t2m")  # (3, 4) float32
    assert out.transform == Affine(10.0, 0.0, 500_000.0, 0.0, expected_e, expected_f)
    assert out.crs.is_projected


def test_read_netcdf_transform_from_lon_lat(tmp_path: Path):
    from affine import Affine

    lon = -3.0 + 0.25 * (np.arange(8) + 0.5)
    lat = 40.0 - 0.25 * (np.arange(6) + 0.5)
    path = _cf_netcdf(tmp_path / "ll.nc", lon, lat, names=("lat", "lon"))
    out = read_netcdf(path, "t2m")  # (6, 8) float32
    assert out.transform == Affine(0.25, 0.0, -3.0, 0.0, -0.25, 40.0)


def test_read_netcdf_geotransform_wins_over_coordinates(tmp_path: Path):
    x = 500_000 + 10 * (np.arange(4) + 0.5)
    y = 4_200_000 - 10 * (np.arange(3) + 0.5)
    path = _cf_netcdf(tmp_path / "gt.nc", x, y, geotransform="0 1 0 0 0 -1")
    out = read_netcdf(path, "t2m")
    assert tuple(out.transform)[:6] == (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def test_read_netcdf_irregular_coordinates_warn(tmp_path: Path):
    from affine import Affine

    x = np.array([0.0, 1.0, 3.0, 7.0])
    y = np.array([2.0, 1.0, 0.0])
    path = _cf_netcdf(tmp_path / "irregular.nc", x, y)
    with pytest.warns(UserWarning, match="not evenly spaced"):
        out = read_netcdf(path, "t2m")
    assert out.transform == Affine.identity()


def test_read_netcdf_without_coordinates_warns(tmp_path: Path):
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "bare.nc"
    with netcdf4.Dataset(path, "w") as ds:
        ds.createDimension("row", 2)
        ds.createDimension("col", 3)
        ds.createVariable("v", "f4", ("row", "col"))[:] = np.ones((2, 3))
    with pytest.warns(UserWarning, match="no 1-D coordinate variable for 'row', 'col'"):
        read_netcdf(path, "v")


def test_read_netcdf_kilometre_coordinates_are_metres(tmp_path: Path):
    from affine import Affine

    x = 500 + 0.01 * (np.arange(4) + 0.5)  # km
    y = 4200 - 0.01 * (np.arange(3) + 0.5)
    out = read_netcdf(_cf_netcdf(tmp_path / "km.nc", x, y, units="km"), "t2m")
    assert out.transform.almost_equals(
        Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_200_000.0)
    )


def test_read_netcdf_unknown_projected_units_warn(tmp_path: Path):
    x = 1.0 + np.arange(4)
    y = 3.0 - np.arange(3)
    path = _cf_netcdf(tmp_path / "ft.nc", x, y, units="furlong")
    with pytest.warns(UserWarning, match="units 'furlong' are not metres"):
        read_netcdf(path, "t2m")


def test_read_netcdf_coordinates_from_a_parent_group(tmp_path: Path):
    netcdf4 = pytest.importorskip("netCDF4")
    from affine import Affine

    path = tmp_path / "grouped.nc"
    with netcdf4.Dataset(path, "w") as ds:
        ds.createDimension("lat", 2)
        ds.createDimension("lon", 3)
        lat = ds.createVariable("lat", "f8", ("lat",))
        lat[:], lat.units = [10.5, 9.5], "degrees_north"
        lon = ds.createVariable("lon", "f8", ("lon",))
        lon[:], lon.units = [0.5, 1.5, 2.5], "degrees_east"
        product = ds.createGroup("PRODUCT")
        product.createVariable("v", "f4", ("lat", "lon"))[:] = np.ones((2, 3))
    out = read_netcdf(path, "v", group="PRODUCT")
    assert out.transform == Affine(1.0, 0.0, 0.0, 0.0, -1.0, 11.0)


def test_read_netcdf_declared_coordinates_beat_index_dimensions(tmp_path: Path):
    netcdf4 = pytest.importorskip("netCDF4")
    from affine import Affine

    path = tmp_path / "indexed.nc"
    with netcdf4.Dataset(path, "w") as ds:
        ds.createDimension("j", 2)
        ds.createDimension("i", 3)
        ds.createVariable("j", "i4", ("j",))[:] = [0, 1]  # bare indexes
        ds.createVariable("i", "i4", ("i",))[:] = [0, 1, 2]
        lat = ds.createVariable("lat", "f8", ("j",))
        lat[:], lat.units = [10.5, 9.5], "degrees_north"
        lon = ds.createVariable("lon", "f8", ("i",))
        lon[:], lon.units = [0.5, 1.5, 2.5], "degrees_east"
        v = ds.createVariable("v", "f4", ("j", "i"))
        v[:], v.coordinates = np.ones((2, 3)), "lat lon"
    out = read_netcdf(path, "v")
    assert out.transform == Affine(1.0, 0.0, 0.0, 0.0, -1.0, 11.0)


def test_read_netcdf_curvilinear_coordinates_warn(tmp_path: Path):
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "swath.nc"
    with netcdf4.Dataset(path, "w") as ds:
        ds.createDimension("row", 2)
        ds.createDimension("col", 3)
        ds.createVariable("lat", "f8", ("row", "col"))[:] = np.ones((2, 3))
        ds.createVariable("lon", "f8", ("row", "col"))[:] = np.ones((2, 3))
        v = ds.createVariable("v", "f4", ("row", "col"))
        v[:], v.coordinates = np.ones((2, 3)), "lat lon"
    with pytest.warns(UserWarning, match="curvilinear"):
        read_netcdf(path, "v")


def test_read_netcdf_regular_axes_win_over_2d_auxiliaries(tmp_path: Path):
    netcdf4 = pytest.importorskip("netCDF4")
    from affine import Affine

    path = tmp_path / "cf52.nc"
    with netcdf4.Dataset(path, "w") as ds:
        ds.createDimension("y", 2)
        ds.createDimension("x", 3)
        x = ds.createVariable("x", "f8", ("x",))
        x[:], x.units = [5.0, 15.0, 25.0], "m"
        y = ds.createVariable("y", "f8", ("y",))
        y[:], y.units = [15.0, 5.0], "m"
        ds.createVariable("lat", "f8", ("y", "x"))[:] = np.ones((2, 3))
        ds.createVariable("lon", "f8", ("y", "x"))[:] = np.ones((2, 3))
        v = ds.createVariable("v", "f4", ("y", "x"))
        v[:], v.coordinates = np.ones((2, 3)), "lat lon"
    out = read_netcdf(path, "v")
    assert out.transform == Affine(10.0, 0.0, 0.0, 0.0, -10.0, 20.0)


def test_read_netcdf_float32_coordinates_near_the_antimeridian(tmp_path: Path):
    netcdf4 = pytest.importorskip("netCDF4")

    path = tmp_path / "f32.nc"
    lon = (170.05 + 0.1 * np.arange(100)).astype("f4")  # 0.1° to 179.95°
    lat = (10.0 - 0.1 * (np.arange(5) + 0.5)).astype("f4")
    with netcdf4.Dataset(path, "w") as ds:
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        v_lon = ds.createVariable("lon", "f4", ("lon",))
        v_lon[:], v_lon.units = lon, "degrees_east"
        v_lat = ds.createVariable("lat", "f4", ("lat",))
        v_lat[:], v_lat.units = lat, "degrees_north"
        ds.createVariable("v", "f4", ("lat", "lon"))[:] = np.ones((5, 100))
    out = read_netcdf(path, "v")
    assert out.transform.a == pytest.approx(0.1, rel=1e-5)
    assert out.transform.c == pytest.approx(170.0, abs=1e-4)
