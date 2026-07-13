"""Tests for internal URI resolution helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from geocatalog._src.io import (
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


def test_resolve_uri_requires_fsspec_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fsspec", None)

    with pytest.raises(ImportError, match=r"geocatalog\[fsspec\]"):
        _resolve_uri("s3://bucket/key.tif")


def test_resolve_uri_forwards_storage_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class DummyFile:
        closed = False

        def close(self) -> None:
            self.closed = True

    handle = DummyFile()

    class DummyOpenFile:
        def open(self) -> DummyFile:
            return handle

    def fake_open(path: str, mode: str, **kwargs: object) -> DummyOpenFile:
        calls.append((path, mode, kwargs))
        return DummyOpenFile()

    monkeypatch.setitem(sys.modules, "fsspec", SimpleNamespace(open=fake_open))

    resolved = _resolve_uri("s3://bucket/key.tif", storage_options={"anon": True})

    assert resolved is handle
    assert calls == [("s3://bucket/key.tif", "rb", {"anon": True})]
    _close_resolved_uri(resolved)
    assert handle.closed


def test_uri_name_handles_cloud_paths() -> None:
    assert _uri_name("s3://bucket/prefix/S2_T29SND_20240115.tif") == (
        "S2_T29SND_20240115.tif"
    )


def test_resolve_uri_zarr_returns_mapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zarr URIs route through ``fsspec.get_mapper`` rather than ``.open()``.

    A single binary handle can't represent a directory-/mapping-based store,
    so the resolver must dispatch on the ``.zarr`` suffix.
    """
    mapper_calls: list[tuple[str, dict[str, object]]] = []
    open_calls: list[tuple[str, str, dict[str, object]]] = []
    sentinel_mapper = object()

    def fake_get_mapper(path: str, **kwargs: object) -> object:
        mapper_calls.append((path, kwargs))
        return sentinel_mapper

    def fake_open(path: str, mode: str, **kwargs: object) -> object:
        open_calls.append((path, mode, kwargs))
        raise AssertionError("fsspec.open must not be called for .zarr URIs")

    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        SimpleNamespace(open=fake_open, get_mapper=fake_get_mapper),
    )

    resolved = _resolve_uri("s3://bucket/store.zarr", storage_options={"anon": True})

    assert resolved is sentinel_mapper
    assert mapper_calls == [("s3://bucket/store.zarr", {"anon": True})]
    assert open_calls == []
    # `_close_resolved_uri` must be a no-op for mappers (no `.close`).
    _close_resolved_uri(resolved)
