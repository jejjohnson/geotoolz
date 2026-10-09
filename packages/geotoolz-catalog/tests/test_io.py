"""Tests for internal URI resolution helpers."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from geocatalog._src.utils.paths import (
    _close_resolved_uri,
    _is_fsspec_uri,
    _resolve_uri,
    _uri_name,
)


def test_resolve_uri_local_path_passthrough(tmp_path: Path) -> None:
    path = tmp_path / "tile.tif"

    assert _resolve_uri(path) == path


def test_resolve_uri_windows_drive_letter_passthrough() -> None:
    # `urlparse("C:/data/tile.tif").scheme == "c"`, so a naive scheme
    # check would mis-route Windows local paths through fsspec. The
    # `_FSSPEC_SCHEMES` allowlist must NOT include single-letter
    # schemes — verify the path passes through untouched.
    windows_path = "C:/data/tile.tif"

    assert not _is_fsspec_uri(windows_path)
    assert _resolve_uri(windows_path) == windows_path


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/key.tif",
        "gs://bucket/key.tif",
        "gcs://bucket/key.tif",
        "az://container/key.tif",
        "azure://container/key.tif",
        "http://example.com/key.tif",
        "https://example.com/key.tif",
        "hf://datasets/org/repo/key.tif",
    ],
)
def test_is_fsspec_uri_recognises_supported_schemes(uri: str) -> None:
    assert _is_fsspec_uri(uri)


def test_resolve_uri_requires_the_cloud_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "geocloud.fs", None)

    with pytest.raises(ImportError, match=r"geotoolz-catalog\[cloud\]"):
        _resolve_uri("s3://bucket/key.tif")


@pytest.fixture
def bucket() -> Iterator[str]:
    """A MemoryStore mounted at an ``s3://`` root on the shared pool."""
    pytest.importorskip("geocloud.fs")
    from geocloud.store import mount, unmount
    from obstore.store import MemoryStore

    mount("s3://catalog-io", MemoryStore())
    yield "s3://catalog-io"
    unmount("s3://catalog-io")


def test_resolve_uri_reads_through_the_pool(bucket: str) -> None:
    from geocloud import files

    files.write_bytes(f"{bucket}/key.bin", b"0123456789")
    resolved = _resolve_uri(f"{bucket}/key.bin", storage_options={"timeout": "5s"})
    resolved.seek(3)
    assert resolved.read(4) == b"3456"
    _close_resolved_uri(resolved)
    assert resolved.closed


def test_uri_name_handles_cloud_paths() -> None:
    assert _uri_name("s3://bucket/prefix/S2_T29SND_20240115.tif") == (
        "S2_T29SND_20240115.tif"
    )


def test_resolve_uri_zarr_returns_mapper(bucket: str) -> None:
    """``.zarr`` URIs resolve to a mapper, not a single file handle."""
    from geocloud import files

    files.write_bytes(f"{bucket}/store.zarr/zarr.json", b"{}")
    resolved = _resolve_uri(f"{bucket}/store.zarr")

    assert resolved["zarr.json"] == b"{}"
    # `_close_resolved_uri` must be a no-op for mappers (no `.close`).
    _close_resolved_uri(resolved)
