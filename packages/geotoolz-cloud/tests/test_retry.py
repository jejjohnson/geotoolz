"""S3's spurious ``NoSuchBucket`` is retried by every geocloud store call."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import obstore
import pytest
from obstore.store import MemoryStore

from geocloud import files
from geocloud._src import retry
from geocloud.store import mount, unmount


ROOT = "s3://flaky-bucket"
SPURIOUS = "<Code>NoSuchBucket</Code> The specified bucket does not exist"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry, "BASE_S", 0.0)


@pytest.fixture
def bucket() -> Iterator[MemoryStore]:
    store = MemoryStore()
    obstore.put(store, "a/b.bin", b"payload")
    mount(ROOT, store)
    yield store
    unmount(ROOT)


def flaky(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    errors: list[BaseException],
) -> list[int]:
    """Make ``obstore.<name>`` raise ``errors`` one by one, then work."""
    real: Callable[..., Any] = getattr(obstore, name)
    calls = [0]

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] <= len(errors):
            raise errors[calls[0] - 1]
        return real(*args, **kwargs)

    monkeypatch.setattr(obstore, name, wrapper)
    return calls


def test_read_bytes_survives_one_spurious_answer(bucket, monkeypatch) -> None:
    calls = flaky(monkeypatch, "get", [OSError(SPURIOUS)])
    assert files.read_bytes(f"{ROOT}/a/b.bin") == b"payload"
    assert calls[0] == 2


def test_ls_survives_one_spurious_answer(bucket, monkeypatch) -> None:
    flaky(monkeypatch, "list", [OSError(SPURIOUS)])
    assert [item.uri for item in files.ls(ROOT)] == [f"{ROOT}/a/b.bin"]


def test_bodiless_head_404_is_retried_when_the_bucket_probe_fails(
    bucket, monkeypatch
) -> None:
    # A HEAD 404 for a missing bucket has no body; the LIST probe names it.
    heads = flaky(monkeypatch, "head", [FileNotFoundError("404")])
    probes = flaky(monkeypatch, "list_with_delimiter", [OSError(SPURIOUS)])
    assert files.info(f"{ROOT}/a/b.bin").size == 7
    assert (heads[0], probes[0]) == (2, 1)


def test_missing_key_fails_at_once(bucket, monkeypatch) -> None:
    heads = flaky(monkeypatch, "head", [])
    with pytest.raises(FileNotFoundError):
        files.info(f"{ROOT}/a/missing.bin")
    assert heads[0] == 1  # the probe listed fine: a missing key, no retry


def test_download_survives_a_spurious_head(bucket, monkeypatch, tmp_path) -> None:
    flaky(monkeypatch, "head", [FileNotFoundError("404")])
    flaky(monkeypatch, "list_with_delimiter", [OSError(SPURIOUS)])
    dest = files.download(f"{ROOT}/a/b.bin", tmp_path / "b.bin")
    assert Path(dest).read_bytes() == b"payload"


def test_persistent_no_such_bucket_raises_after_the_attempts(
    bucket, monkeypatch
) -> None:
    calls = flaky(monkeypatch, "get", [OSError(SPURIOUS)] * 10)
    with pytest.raises(OSError, match="NoSuchBucket"):
        files.read_bytes(f"{ROOT}/a/b.bin")
    assert calls[0] == retry.ATTEMPTS


def test_other_errors_are_not_retried(bucket, monkeypatch) -> None:
    calls = flaky(monkeypatch, "get", [PermissionError("AccessDenied")])
    with pytest.raises(PermissionError):
        files.read_bytes(f"{ROOT}/a/b.bin")
    assert calls[0] == 1


def test_cog_source_open_survives_one_spurious_answer(
    tmp_path: Path, monkeypatch
) -> None:
    async_geotiff = pytest.importorskip("async_geotiff")
    rasterio = pytest.importorskip("rasterio")
    from obstore.store import LocalStore
    from rasterio.transform import from_origin

    from geocloud.cog import CogSource

    path = tmp_path / "x.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=1,
        dtype="float32",
        crs="EPSG:32630",
        transform=from_origin(500_000, 4_200_000, 10, 10),
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        dst.write(np.ones((1, 32, 32), dtype=np.float32))

    real = async_geotiff.GeoTIFF.open
    calls = [0]

    async def once(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] == 1:
            raise OSError(SPURIOUS)
        return await real(*args, **kwargs)

    monkeypatch.setattr(async_geotiff.GeoTIFF, "open", once)
    store = LocalStore(prefix=str(tmp_path))
    source = CogSource.open(url=f"file://{path}", store=store, path="x.tif")
    assert source.domain.shape[-2:] == (32, 32)
    assert calls[0] == 2


def test_recursive_rm_retries_each_batch(bucket, monkeypatch) -> None:
    calls = flaky(monkeypatch, "delete", [OSError(SPURIOUS)])
    assert files.rm(ROOT, recursive=True) == 1
    assert calls[0] == 2
    assert files.ls(ROOT) == []


def test_reads_through_an_open_handle_are_retried() -> None:
    from geocloud._src.files import _RetryingReader

    class Handle:
        def __init__(self) -> None:
            self.pos, self.failed = 0, False

        def tell(self) -> int:
            return self.pos

        def seek(self, pos: int) -> int:
            self.pos = pos
            return pos

        def read(self, size: int) -> bytes:
            if not self.failed:
                self.failed = True
                self.pos += 1  # a failed request must not move the position
                raise OSError(SPURIOUS)
            data = b"payload"[self.pos : self.pos + size]
            self.pos += len(data)
            return data

    reader = _RetryingReader(Handle())
    assert reader.read(3) == b"pay"
    assert reader.read(4) == b"load"


def test_async_probe_runs_off_the_event_loop() -> None:
    import asyncio
    import threading

    seen: list[bool] = []
    calls = [0]

    def probe() -> None:
        seen.append(threading.current_thread() is threading.main_thread())
        raise OSError(SPURIOUS)

    async def call() -> str:
        calls[0] += 1
        if calls[0] == 1:
            raise FileNotFoundError("404")
        return "ok"

    assert asyncio.run(retry.awith_bucket_retry(call, probe=probe)) == "ok"
    assert seen == [False]


def test_object_version_probes_a_bodiless_head_404(monkeypatch) -> None:
    pytest.importorskip("async_geotiff")
    from geocloud.cog import CogSource

    class Store:
        heads = 0

        def head(self, key: str) -> dict[str, Any]:
            Store.heads += 1
            if Store.heads == 1:
                raise FileNotFoundError("404")
            return {"e_tag": '"abc"'}

    flaky(monkeypatch, "list_with_delimiter", [OSError(SPURIOUS)])
    source = object.__new__(CogSource)
    object.__setattr__(source, "store", Store())
    object.__setattr__(source, "path", "a/x.tif")
    object.__setattr__(source, "url", "s3://flaky-bucket/a/x.tif")
    object.__setattr__(source, "storage_options", None)
    assert source.object_version() == 'etag:"abc"'
    assert Store.heads == 2
