"""Tests for ``SensorReader``'s optional obstore byte path.

Three things to pin:

1. Existing readers (``toy_sensor.Reader``) keep working without any
   obstore install — backwards-compat.
2. ``set_obstore_client`` attaches a pool client to a reader, which
   then routes remote-URI ``_read_bytes`` calls through it.
3. Local paths and ``file://`` URIs always take the local fallback,
   even when a client is attached.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from affine import Affine
from rasterio.windows import Window

from geotoolz.readers._base import SensorReader, Track


# --- minimal concrete reader for the test surface -----------------------


class _MinimalReader(SensorReader):
    """Bare-minimum concrete reader — keeps the abstract surface satisfied.

    No file I/O; just enough state to exercise the obstore hook from a
    test that doesn't touch real disk.
    """

    def __init__(
        self, path: str | Path = "scene", *, shape: tuple[int, int, int] = (1, 4, 4)
    ) -> None:
        self.path = str(path)
        self._array = np.zeros(shape, dtype=np.float32)

    def _read_window(self, window: Window) -> np.ndarray:
        return self._array[
            :,
            int(window.row_off) : int(window.row_off + window.height),
            int(window.col_off) : int(window.col_off + window.width),
        ]

    @property
    def _crs(self) -> Any:
        return "EPSG:4326"

    @property
    def _transform(self) -> Affine:
        return Affine.identity()

    @property
    def _shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def _dtype(self) -> Any:
        return self._array.dtype

    @property
    def _bands(self) -> tuple[str, ...]:
        return ("a",)

    @property
    def _fill_value(self) -> Any:
        return 0

    @property
    def _track(self) -> Track:
        return "A"


# --- backwards-compat: existing readers don't need the hook --------------


def test_existing_reader_does_not_need_obstore_init():
    """A reader that doesn't call ``super().__init__()`` still works."""
    r = _MinimalReader()
    # Loading should succeed without the obstore_client attribute set.
    assert r.obstore_client is None


def test_obstore_client_defaults_to_none():
    r = _MinimalReader()
    assert r.obstore_client is None


def test_set_obstore_client_round_trips():
    r = _MinimalReader()
    sentinel = object()
    r.set_obstore_client(sentinel)  # type: ignore[arg-type]
    assert r.obstore_client is sentinel
    r.set_obstore_client(None)
    assert r.obstore_client is None


# --- local fallback (no client needed) ----------------------------------


def test_read_bytes_local_path(tmp_path: Path):
    """Local path reads go through ``_read_bytes_local`` regardless of client."""
    payload = b"this is the test payload, in raw bytes"
    src = tmp_path / "blob.bin"
    src.write_bytes(payload)

    r = _MinimalReader()
    got = r._read_bytes(str(src), 5, 10)
    assert got == payload[5:15]


def test_read_bytes_file_uri(tmp_path: Path):
    """``file://`` URIs take the local path too — no remote scheme."""
    payload = b"abcdefghijklmnop"
    src = tmp_path / "blob.bin"
    src.write_bytes(payload)

    r = _MinimalReader()
    got = r._read_bytes(f"file://{src}", 0, 8)
    assert got == payload[:8]


def test_read_bytes_windows_drive_letter_path_treated_as_local():
    """Regression: ``C:/foo.bin`` must not be classified as a remote scheme.

    ``urlsplit("C:/foo.bin").scheme == "c"`` — without the ``"://"``
    guard this would route through the obstore path and raise the
    "remote scheme but no client" ``RuntimeError`` on Windows, even
    though the user passed a plain local path. Verifies the guard
    using a path that doesn't actually exist on disk (we only need
    the scheme-classification logic to take the local branch — the
    ``open()`` will of course raise ``FileNotFoundError``, which is
    the *correct* failure mode for a non-existent local path).
    """
    from geotoolz.readers._base import _has_remote_scheme

    # The classifier must NOT mark this as remote (no "://" in URI).
    assert not _has_remote_scheme("C:/scene.bin")
    assert not _has_remote_scheme(r"C:\scene.bin")
    # End-to-end: the SensorReader path must surface FileNotFoundError
    # (correct local-path failure), not RuntimeError (wrong-path bug).
    r = _MinimalReader()
    with pytest.raises(FileNotFoundError):
        r._read_bytes("C:/nonexistent_drive_letter_path.bin", 0, 4)


def test_read_bytes_local_with_client_attached(tmp_path: Path):
    """An attached client doesn't divert local reads to obstore.

    Mirrors what ``RasterField`` does for non-remote inputs — local
    paths skip the pool even when a client is wired up.
    """
    payload = b"local_path_payload"
    src = tmp_path / "blob.bin"
    src.write_bytes(payload)

    r = _MinimalReader()
    # Attach a sentinel non-callable so any accidental obstore route
    # would blow up loudly.
    r.set_obstore_client(object())  # type: ignore[arg-type]
    assert r._read_bytes(str(src), 0, 5) == payload[:5]


# --- remote scheme without an attached client raises --------------------


def test_remote_uri_without_client_raises():
    r = _MinimalReader()
    with pytest.raises(RuntimeError, match="no obstore client is attached"):
        r._read_bytes("s3://bucket/key.bin", 0, 16)


# --- remote scheme with a LocalStore-backed client uses obstore ---------


def test_remote_scheme_routes_through_client(tmp_path: Path):
    """End-to-end: an attached ``LocalStore`` resolves an ``s3://`` URI."""
    obstore_store = pytest.importorskip("obstore.store")
    payload = b"the quick brown fox jumps over the lazy dog"
    src = tmp_path / "blob.bin"
    src.write_bytes(payload)

    # LocalStore happily resolves any URI's path component relative to
    # its prefix — we feed a fake `s3://...` URI but the key is
    # `blob.bin`, which exists in the tmpdir prefix.
    client = obstore_store.LocalStore(prefix=str(tmp_path))
    r = _MinimalReader()
    r.set_obstore_client(client)
    got = r._read_bytes("s3://bucket/blob.bin", 16, 10)
    assert got == payload[16:26]
