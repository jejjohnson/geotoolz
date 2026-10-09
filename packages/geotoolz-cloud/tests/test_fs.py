"""`geocloud.fs`: fsspec on the pool, against a mounted MemoryStore and local files."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest


fsspec = pytest.importorskip("fsspec")

from obstore.store import MemoryStore

from geocloud import files
from geocloud.fs import GeoCloudFileSystem, filesystem
from geocloud.store import mount, unmount


ROOT = "s3://fs-test"


@pytest.fixture
def fs() -> Iterator[GeoCloudFileSystem]:
    mount(ROOT, MemoryStore())
    yield filesystem()
    unmount(ROOT)


def test_open_reads_ranges_through_the_pool(fs: GeoCloudFileSystem):
    files.write_bytes(f"{ROOT}/a.bin", bytes(range(100)))
    with fs.open(f"{ROOT}/a.bin") as fh:
        fh.seek(10)
        assert fh.read(5) == bytes(range(10, 15))
        assert fh.tell() == 15
    assert fs.cat_file(f"{ROOT}/a.bin", start=95) == bytes(range(95, 100))
    assert fs.cat_file(f"{ROOT}/a.bin", start=-3, end=-1) == bytes([97, 98])
    assert fs.size(f"{ROOT}/a.bin") == 100


def test_write_ls_info_and_rm(fs: GeoCloudFileSystem):
    with fs.open(f"{ROOT}/dir/x.txt", "wb") as fh:
        fh.write(b"hello")
    fs.pipe_file(f"{ROOT}/dir/sub/y.txt", b"!")
    assert fs.cat_file(f"{ROOT}/dir/x.txt") == b"hello"
    names = fs.ls(f"{ROOT}/dir", detail=False)
    assert names == [f"{ROOT}/dir/sub", f"{ROOT}/dir/x.txt"]
    assert fs.info(f"{ROOT}/dir")["type"] == "directory"
    assert fs.isfile(f"{ROOT}/dir/x.txt")
    assert sorted(fs.find(f"{ROOT}/dir")) == [
        f"{ROOT}/dir/sub/y.txt",
        f"{ROOT}/dir/x.txt",
    ]
    fs.rm_file(f"{ROOT}/dir/x.txt")
    assert not fs.exists(f"{ROOT}/dir/x.txt")
    with pytest.raises(FileNotFoundError):
        fs.info(f"{ROOT}/nope.bin")


def test_local_paths_work_the_same(tmp_path: Path):
    fs = filesystem()
    path = str(tmp_path / "deep" / "a.bin")
    fs.pipe_file(path, b"abcdef")
    with fs.open(path) as fh:
        fh.seek(1)
        assert fh.read(2) == b"bc"
    assert fs.ls(str(tmp_path / "deep"), detail=False) == [path]


def test_parquet_through_a_file_handle(fs: GeoCloudFileSystem):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"a": [1, 2, 3]})
    with fs.open(f"{ROOT}/t.parquet", "wb") as fh:
        pq.write_table(table, fh)
    with fs.open(f"{ROOT}/t.parquet") as fh:
        assert pq.read_table(fh).equals(table)


def test_zarr_round_trip_through_a_mapper(fs: GeoCloudFileSystem):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    ds = xr.Dataset({"v": (("y", "x"), np.arange(12.0).reshape(3, 4))})
    ds.to_zarr(fs.get_mapper(f"{ROOT}/cube.zarr"), mode="w")
    back = xr.open_zarr(fs.get_mapper(f"{ROOT}/cube.zarr"))
    np.testing.assert_array_equal(back["v"].values, ds["v"].values)


def test_storage_options_reach_the_pool(fs: GeoCloudFileSystem, monkeypatch):
    """Paths resolve through `get_obstore` (which merges registered credentials)."""
    seen: list[object] = []
    import geocloud._src.files as impl

    real = impl.get_obstore

    def spy(uri, storage_options=None):
        seen.append(storage_options)
        return real(uri, storage_options=storage_options)

    monkeypatch.setattr(impl, "get_obstore", spy)
    files.write_bytes(f"{ROOT}/c.bin", b"x")
    filesystem({"client_options": {"timeout": "5s"}}).cat_file(f"{ROOT}/c.bin")
    assert {"client_options": {"timeout": "5s"}} in seen
