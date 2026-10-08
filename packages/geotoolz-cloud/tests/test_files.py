"""Tests for `geocloud.files` — URI verbs on the pool.

No network: ``s3://`` / ``az://`` roots are served by `MemoryStore`s
through `geocloud.store.mount`, and local paths by ``tmp_path``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import obstore
import pytest
from obstore.store import MemoryStore

from geocloud import files
from geocloud._src import files as impl
from geocloud.store import clear_obstore_pool, get_obstore, mount, unmount


S3 = "s3://bucket"
AZ = "az://account/container"


@pytest.fixture(autouse=True)
def stores() -> Iterator[dict[str, MemoryStore]]:
    """Two independent in-memory roots: an S3 bucket and an Azure container."""
    mounted = {S3: MemoryStore(), AZ: MemoryStore()}
    for root, store in mounted.items():
        mount(root, store)
    yield mounted
    for root in mounted:
        unmount(root)
    clear_obstore_pool()


def put(store: MemoryStore, key: str, data: bytes = b"data") -> None:
    obstore.put(store, key, data)


# --- ls / info / exists ----------------------------------------------------


def test_ls_recursive_names_full_uris_sorted(stores):
    for key in ("dir/b.bin", "dir/a.bin", "dir/sub/c.bin", "dirx/d.bin"):
        put(stores[S3], key)
    got = files.ls(f"{S3}/dir")
    assert [item.uri for item in got] == [
        f"{S3}/dir/a.bin",
        f"{S3}/dir/b.bin",
        f"{S3}/dir/sub/c.bin",
    ]
    assert got[0].size == 4 and got[0].last_modified is not None
    assert not got[0].is_dir


def test_ls_trailing_slash_and_root(stores):
    put(stores[S3], "dir/a.bin")
    put(stores[S3], "top.bin")
    assert [i.uri for i in files.ls(f"{S3}/dir/")] == [f"{S3}/dir/a.bin"]
    assert len(files.ls(S3)) == 2


def test_ls_one_level_lists_directories(stores):
    put(stores[AZ], "dir/a.bin")
    put(stores[AZ], "dir/sub/b.bin")
    got = files.ls(f"{AZ}/dir", recursive=False)
    assert [(i.uri, i.is_dir) for i in got] == [
        (f"{AZ}/dir/a.bin", False),
        (f"{AZ}/dir/sub/", True),
    ]


def test_ls_local_directory(tmp_path):
    files.write_bytes(tmp_path / "a" / "x.bin", b"abc")
    got = files.ls(tmp_path)
    assert [Path(i.uri) for i in got] == [tmp_path / "a" / "x.bin"]
    assert files.ls(tmp_path / "missing") == []


def test_ls_rejects_unlistable_uris():
    with pytest.raises(ValueError, match="cannot be listed"):
        files.ls("hf://org/repo/file.bin")
    with pytest.raises(ValueError, match="cannot be listed"):
        files.ls("https://example.com/dir/")


def test_local_paths_with_dotdot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files.write_bytes("sub/../a.bin", b"abc")
    assert files.read_bytes("sub/../a.bin") == b"abc"
    assert not files.exists("../definitely-missing.bin")


def test_info_exists(stores, tmp_path):
    put(stores[S3], "a.bin", b"abc")
    assert files.info(f"{S3}/a.bin").size == 3
    assert files.exists(f"{S3}/a.bin")
    assert not files.exists(f"{S3}/missing.bin")
    with pytest.raises(FileNotFoundError):
        files.info(f"{S3}/missing.bin")
    assert not files.exists(tmp_path / "nope")


# --- read / write / open ---------------------------------------------------


def test_read_write_bytes_remote_and_local(tmp_path):
    files.write_bytes(f"{S3}/x/y.bin", b"hello")
    assert files.read_bytes(f"{S3}/x/y.bin") == b"hello"
    local = tmp_path / "deep" / "y.bin"
    files.write_bytes(local, bytearray(b"there"))
    assert local.read_bytes() == b"there"
    assert files.read_bytes(f"file://{local}") == b"there"


def test_open_reader_and_writer():
    with files.open(f"{S3}/f.bin", "wb") as fh:
        fh.write(b"0123456789")
    fh = files.open(f"{S3}/f.bin")
    fh.seek(4)
    assert bytes(fh.read(3)) == b"456"
    fh.close()
    with pytest.raises(ValueError, match="mode"):
        files.open(f"{S3}/f.bin", "r")  # type: ignore[arg-type]


# --- download / upload -------------------------------------------------------


def test_download_into_directory_and_file(stores, tmp_path):
    put(stores[S3], "scenes/b04.tif", b"pixels")
    into_dir = files.download(f"{S3}/scenes/b04.tif", f"{tmp_path}/out/")
    assert into_dir == tmp_path / "out" / "b04.tif"
    assert into_dir.read_bytes() == b"pixels"
    (tmp_path / "existing").mkdir()
    assert files.download(f"{S3}/scenes/b04.tif", tmp_path / "existing") == (
        tmp_path / "existing" / "b04.tif"
    )
    named = files.download(f"{S3}/scenes/b04.tif", tmp_path / "renamed.tif")
    assert named.read_bytes() == b"pixels"
    assert not list(tmp_path.rglob("*.part"))


def test_download_overwrite_false_keeps_file(stores, tmp_path):
    put(stores[S3], "a.bin", b"new")
    dest = tmp_path / "a.bin"
    dest.write_bytes(b"old")
    assert files.download(f"{S3}/a.bin", dest, overwrite=False) == dest
    assert dest.read_bytes() == b"old"
    files.download(f"{S3}/a.bin", dest)
    assert dest.read_bytes() == b"new"


def test_download_short_read_leaves_nothing(stores, tmp_path, monkeypatch):
    put(stores[S3], "a.bin", b"abcdef")

    class Truncated:
        def bytes(self) -> bytes:
            return b"abc"

    monkeypatch.setattr(impl.obstore, "get", lambda *a, **k: Truncated())
    with pytest.raises(OSError, match="short read"):
        files.download(f"{S3}/a.bin", tmp_path / "a.bin")
    assert list(tmp_path.iterdir()) == []


def test_large_objects_move_in_ranges(stores, tmp_path, monkeypatch):
    """Each GET covers one bounded range, so no request outlives its timeout."""
    monkeypatch.setattr(impl, "_CHUNK", 4)
    payload = b"0123456789"
    put(stores[S3], "a.bin", payload)
    seen: list[tuple[int, int]] = []
    real_get, real_get_async = impl.obstore.get, impl.obstore.get_async

    def get(store, key, *, options=None):
        seen.append((options or {}).get("range"))
        return real_get(store, key, options=options)

    async def get_async(store, key, *, options=None):
        seen.append((options or {}).get("range"))
        return await real_get_async(store, key, options=options)

    monkeypatch.setattr(impl.obstore, "get", get)
    monkeypatch.setattr(impl.obstore, "get_async", get_async)
    assert files.download(f"{S3}/a.bin", tmp_path / "a.bin").read_bytes() == payload
    assert seen == [(0, 4), (4, 8), (8, 10)]
    seen.clear()
    files.copy(f"{S3}/a.bin", f"{AZ}/a.bin")
    assert seen == [(0, 4), (4, 8), (8, 10)]
    assert files.read_bytes(f"{AZ}/a.bin") == payload


def test_object_replaced_mid_download_fails(stores, tmp_path, monkeypatch):
    monkeypatch.setattr(impl, "_CHUNK", 4)
    put(stores[S3], "a.bin", b"0123456789")
    real_get = impl.obstore.get

    def get(store, key, *, options=None):
        result = real_get(store, key, options=options)
        put(stores[S3], "a.bin", b"replaced!!")  # a writer races the download
        return result

    monkeypatch.setattr(impl.obstore, "get", get)
    with pytest.raises(Exception, match="(?i)precondition"):
        files.download(f"{S3}/a.bin", tmp_path / "a.bin")
    assert list(tmp_path.iterdir()) == []


def test_download_missing_object(tmp_path):
    with pytest.raises(FileNotFoundError):
        files.download(f"{S3}/missing.bin", tmp_path / "x")
    assert list(tmp_path.iterdir()) == []


def test_download_and_upload_check_their_local_side(tmp_path):
    with pytest.raises(ValueError, match="local path"):
        files.download(f"{S3}/a.bin", f"{AZ}/a.bin")
    with pytest.raises(ValueError, match="local file"):
        files.upload(f"{S3}/a.bin", f"{AZ}/a.bin")


def test_upload_keeps_name_under_trailing_slash(stores, tmp_path):
    src = tmp_path / "ndvi.tif"
    src.write_bytes(b"x" * 100)
    assert files.upload(src, f"{AZ}/results/") == f"{AZ}/results/ndvi.tif"
    assert bytes(obstore.get(stores[AZ], "results/ndvi.tif").bytes()) == b"x" * 100
    src.write_bytes(b"changed")
    files.upload(src, f"{AZ}/results/ndvi.tif", overwrite=False)
    assert obstore.head(stores[AZ], "results/ndvi.tif")["size"] == 100


# --- copy ------------------------------------------------------------------------


def test_copy_within_one_store_is_server_side(stores, monkeypatch):
    put(stores[S3], "a.bin", b"abc")
    calls: list[tuple[str, str]] = []
    real = impl.obstore.copy

    def spy(store, src, dst, **kw):
        calls.append((src, dst))
        return real(store, src, dst, **kw)

    monkeypatch.setattr(impl.obstore, "copy", spy)
    files.copy(f"{S3}/a.bin", f"{S3}/b/a.bin")
    assert calls == [("a.bin", "b/a.bin")]
    assert files.read_bytes(f"{S3}/b/a.bin") == b"abc"


def test_copy_across_stores_streams(stores):
    payload = bytes(range(256)) * 50_000  # 12.8 MB: several chunks
    put(stores[S3], "big.bin", payload)
    assert files.copy(f"{S3}/big.bin", f"{AZ}/copied/") == f"{AZ}/copied/big.bin"
    assert bytes(obstore.get(stores[AZ], "copied/big.bin").bytes()) == payload


def test_copy_across_stores_inside_a_running_loop(stores):
    put(stores[S3], "a.bin", b"abc")

    async def from_async_code() -> None:
        files.copy(f"{S3}/a.bin", f"{AZ}/a.bin")

    asyncio.run(from_async_code())
    assert files.read_bytes(f"{AZ}/a.bin") == b"abc"


def test_copy_empty_object(stores):
    put(stores[S3], "empty", b"")
    files.copy(f"{S3}/empty", f"{AZ}/empty")
    assert files.info(f"{AZ}/empty").size == 0


def test_copy_local_to_local_is_a_real_copy(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"abc")
    dst = Path(files.copy(src, tmp_path / "b.bin"))
    dst.write_bytes(b"changed")  # edits in place must not reach the source
    assert src.read_bytes() == b"abc"


def test_copy_rejects_prefixes(stores, tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"abc")
    with pytest.raises(ValueError, match="destination"):
        files.copy(src, S3)  # a bucket root without `/` is not an object
    put(stores[S3], "d/a.bin")
    with pytest.raises(ValueError, match="source"):
        files.copy(f"{S3}/d/", tmp_path / "x")


def test_copy_between_roots_mounted_on_one_store_streams():
    shared = MemoryStore()
    mount("s3://one", shared)
    mount("s3://two", shared)
    try:
        files.write_bytes("s3://one/a.bin", b"abc")
        files.copy("s3://one/a.bin", "s3://two/b.bin")
        assert files.read_bytes("s3://two/b.bin") == b"abc"
    finally:
        unmount("s3://one")
        unmount("s3://two")


def test_copy_overwrite_false(stores):
    put(stores[S3], "a.bin", b"new")
    put(stores[AZ], "a.bin", b"old")
    files.copy(f"{S3}/a.bin", f"{AZ}/a.bin", overwrite=False)
    assert files.read_bytes(f"{AZ}/a.bin") == b"old"


# --- sync ------------------------------------------------------------------------


def test_sync_mirrors_and_resumes(stores, tmp_path):
    for key in ("scenes/a.bin", "scenes/sub/b.bin", "other/c.bin"):
        put(stores[S3], key, key.encode())
    copied = files.sync(f"{S3}/scenes", tmp_path / "mirror")
    assert sorted(Path(p).relative_to(tmp_path) for p in copied) == [
        Path("mirror/a.bin"),
        Path("mirror/sub/b.bin"),
    ]
    assert (tmp_path / "mirror" / "sub" / "b.bin").read_bytes() == b"scenes/sub/b.bin"
    assert files.sync(f"{S3}/scenes", tmp_path / "mirror") == []
    put(stores[S3], "scenes/a.bin", b"grown")  # size changed → copied again
    assert len(files.sync(f"{S3}/scenes/", f"{tmp_path}/mirror/")) == 1
    assert len(files.sync(f"{S3}/scenes", tmp_path / "mirror", overwrite=True)) == 2


def test_sync_local_to_remote(stores, tmp_path):
    files.write_bytes(tmp_path / "a.bin", b"1")
    files.write_bytes(tmp_path / "d" / "b.bin", b"2")
    files.sync(tmp_path, f"{AZ}/up")
    assert [i.uri for i in files.ls(AZ)] == [f"{AZ}/up/a.bin", f"{AZ}/up/d/b.bin"]


def test_sync_validates_concurrency():
    with pytest.raises(ValueError, match="max_concurrency"):
        files.sync(S3, AZ, max_concurrency=0)


# --- rm --------------------------------------------------------------------------


def test_rm_one_and_recursive(stores, monkeypatch):
    for key in ("d/a", "d/b", "d/e/c", "keep"):
        put(stores[S3], key)
    assert files.rm(f"{S3}/keep") == 1
    with pytest.raises(FileNotFoundError):
        files.rm(f"{S3}/keep")
    monkeypatch.setattr(impl, "_DELETE_BATCH", 2)
    assert files.rm(f"{S3}/d", recursive=True) == 3
    assert files.ls(S3) == []


# --- sign ------------------------------------------------------------------------


def test_sign_s3_url_offline():
    unmount(S3)
    options = {
        "aws_access_key_id": "AKIDEXAMPLE",
        "aws_secret_access_key": "secret",
        "region": "us-east-1",
    }
    url = files.sign(
        "s3://signed-bucket/a/b.tif",
        expires=timedelta(minutes=5),
        storage_options=options,
    )
    assert url.startswith("https://")
    assert "a/b.tif" in url and "X-Amz-Signature=" in url
    assert get_obstore("s3://signed-bucket/x", storage_options=options) is not None


def test_sign_refuses_what_cannot_sign(tmp_path):
    with pytest.raises(ValueError, match="local path"):
        files.sign(str(tmp_path / "a.bin"))
    with pytest.raises(ValueError, match="cannot pre-sign"):
        files.sign(f"{S3}/a.bin")  # a MemoryStore has no signer
