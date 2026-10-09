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
