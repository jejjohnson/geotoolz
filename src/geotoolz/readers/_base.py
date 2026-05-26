"""Shared reader primitives for sensor-specific modules."""

from __future__ import annotations

import asyncio
import importlib.util
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from affine import Affine
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor
from rasterio.windows import (
    Window,
    from_bounds as window_from_bounds,
    transform as window_transform,
)


if TYPE_CHECKING:
    from obstore.store import ObjectStore


Track = Literal["A", "B"]


class SensorReader(GeoData, ABC):
    """ABC for georeader-compatible sensor readers.

    Subclasses provide sensor-specific metadata and implement
    :meth:`_read_window`; the base class supplies the small GeoData surface
    needed by geotoolz operators.

    Examples:
        Implement a file-backed sensor reader::

            class Reader(SensorReader):
                def __init__(self, path): ...
                def _read_window(self, window): ...

        Load a full scene into a ``GeoTensor``::

            scene = Reader("scene.dat").load()

        Read a tile without changing the reader::

            tile = Reader("scene.dat").read_from_window(Window(0, 0, 256, 256))
    """

    @abstractmethod
    def _read_window(self, window: Window) -> np.ndarray:
        """Read a sensor-native pixel window into a numpy array.

        Implementations receive clipped windows when ``boundless=False`` and
        may receive out-of-bounds windows when ``boundless=True``.
        """

    @property
    @abstractmethod
    def _crs(self) -> Any:
        """Reader CRS."""

    @property
    @abstractmethod
    def _transform(self) -> Affine:
        """Reader affine transform."""

    @property
    @abstractmethod
    def _shape(self) -> tuple[int, ...]:
        """Reader array shape as ``(..., height, width)``."""

    @property
    @abstractmethod
    def _dtype(self) -> Any:
        """Reader array dtype."""

    @property
    @abstractmethod
    def _bands(self) -> Sequence[str]:
        """Band names in array order."""

    @property
    @abstractmethod
    def _fill_value(self) -> Any:
        """Default fill value for boundless reads."""

    @property
    @abstractmethod
    def _track(self) -> Track:
        """Track A for clean affine grids, Track B for irregular geolocation."""

    @property
    def crs(self) -> Any:
        """Reader CRS."""
        return self._crs

    @property
    def transform(self) -> Affine:
        """Affine transform for Track A readers."""
        return self._transform

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape as ``(..., height, width)``."""
        return self._shape

    @property
    def dtype(self) -> Any:
        """Numpy dtype read by this reader."""
        return self._dtype

    @property
    def dims(self) -> list[str]:
        """Dimension names compatible with georeader ``GeoData``.

        The framework expects 2D single-band or 3D band-first image arrays.
        """
        return ["band", "y", "x"] if len(self.shape) == 3 else ["y", "x"]

    @property
    def bands(self) -> tuple[str, ...]:
        """Band names in array order."""
        return tuple(self._bands)

    @property
    def fill_value_default(self) -> Any:
        """Default fill value for out-of-bounds pixels."""
        return self._fill_value

    @property
    def track(self) -> Track:
        """Sensor reader track classification."""
        return self._track

    def load(self, boundless: bool = True) -> GeoTensor:
        """Load the reader's full extent into a ``GeoTensor``."""
        window = Window(col_off=0, row_off=0, width=self.width, height=self.height)
        return self.read_from_window(window, boundless=boundless)

    def read_from_window(self, window: Window, boundless: bool = True) -> GeoTensor:
        """Read a pixel window as a ``GeoTensor``."""
        if not boundless:
            window = self._clip_window(window)
        values = self._read_window(window)
        # Keep both keys for geotoolz.indices named-band compatibility.
        attrs = {"band_names": self.bands, "bands": self.bands}
        return GeoTensor(
            values,
            transform=window_transform(window, self.transform),
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=attrs,
        )

    def read_from_bounds(
        self,
        bounds: tuple[float, float, float, float],
        boundless: bool = True,
    ) -> GeoTensor:
        """Read map-coordinate bounds as a ``GeoTensor``."""
        window = window_from_bounds(*bounds, transform=self.transform)
        return self.read_from_window(window.round_offsets().round_lengths(), boundless)

    def read_from_center_coords(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        boundless: bool = True,
    ) -> GeoTensor:
        """Read a window centered on map coordinates."""
        col_px_float, row_px_float = ~self.transform * (x, y)
        row = int(np.floor(row_px_float))
        col = int(np.floor(col_px_float))
        window = Window(
            col_off=col - width // 2,
            row_off=row - height // 2,
            width=width,
            height=height,
        )
        return self.read_from_window(window, boundless)

    def _clip_window(self, window: Window) -> Window:
        base = Window(col_off=0, row_off=0, width=self.width, height=self.height)
        return window.intersection(base)

    # ------------------------------------------------------------------
    # Optional obstore byte path (opt-in, no abstract-method change).
    # ------------------------------------------------------------------

    def set_obstore_client(self, client: ObjectStore | None) -> None:
        """Attach (or clear) a pooled ``obstore`` client.

        Subclasses that read from cloud storage call this to opt in
        to HTTP/2 connection-pool reuse — every cloud read via
        :meth:`_read_bytes` then funnels through the same pooled
        client instead of building a fresh one per file.

        ``client=None`` clears the attachment and reverts to the local
        ``open(path, "rb").read()`` fallback. Callers can either pass
        a pre-built ``ObjectStore`` (e.g. for tests using
        :class:`obstore.store.LocalStore`) or fetch one from
        :func:`geotoolz._obstore.get_obstore`.
        """
        self._obstore_client = client

    @property
    def obstore_client(self) -> ObjectStore | None:
        """Currently attached obstore client (``None`` if not set)."""
        return getattr(self, "_obstore_client", None)

    def _read_bytes(self, uri: str, start: int, length: int) -> bytes:
        """Read ``length`` bytes starting at ``start`` from ``uri``.

        Two-path dispatch:

        1. **Attached client** — when :meth:`set_obstore_client` has
           wired up a client and the URI scheme matches what the pool
           speaks (``s3://``, ``gs://``, ``https://``, …), the read
           goes through ``obstore`` and HTTP/2 connection reuse
           applies.
        2. **Local fallback** — for plain filesystem paths (no
           scheme, or ``file://``) the method does a one-shot
           ``open(path, "rb").seek(start) + read(length)`` so
           existing on-disk sensor readers (the ``toy_sensor``
           reference, future MODIS HDF readers, …) keep working
           without an obstore install.

        Subclasses opt in by calling this from their own
        ``_read_window`` (or wherever they previously did manual
        ``open(...)`` byte reads). The ABC stays unchanged; nothing
        about the existing reader contract is altered.
        """
        client = self.obstore_client
        if client is not None and _has_remote_scheme(uri):
            return _run_coroutine_safely(_get_range_async(client, uri, start, length))
        return _read_bytes_local(uri, start, length)


def require_optional_dependency(package: str, *, extra: str) -> None:
    """Raise an actionable error when a sensor optional dependency is missing.

    Args:
        package: Import package name to check.
        extra: Sensor optional-extra name to include in the install command.
    """
    if importlib.util.find_spec(package) is not None:
        return
    raise ImportError(
        f"Missing optional dependency {package!r} required for "
        f"geotoolz.readers.{extra}. Install it with "
        f"`pip install 'geotoolz[{extra}]'`."
    )


def as_path(path: str | Path) -> Path:
    """Normalize a reader path argument.

    Args:
        path: String or ``Path`` reader input.

    Returns:
        The input converted to a ``Path`` object.
    """
    return Path(path)


# ----------------------------------------------------------------------
# Byte-range helpers for the SensorReader._read_bytes opt-in path.
# ----------------------------------------------------------------------

# URI schemes the obstore pool can talk to. ``file://`` is intentionally
# omitted — local files take the fast on-disk path even when a client
# is attached.
_REMOTE_SCHEMES = frozenset(
    {"s3", "s3a", "gs", "gcs", "az", "azure", "abfs", "http", "https"}
)


def _has_remote_scheme(uri: str) -> bool:
    """Return True when ``uri`` is one of the pool's known cloud schemes.

    Requires ``"://"`` in the URI before considering the scheme — this
    avoids mis-classifying Windows drive-letter paths like ``C:/foo.bin``,
    which ``urlsplit`` parses as scheme ``"c"`` and would otherwise route
    through obstore (and fail with a confusing "remote scheme but no
    client" error). Plain ``Path`` / ``str`` filesystem paths always
    take the local path.
    """
    if "://" not in uri:
        return False
    from urllib.parse import urlsplit

    return urlsplit(uri).scheme.lower() in _REMOTE_SCHEMES


async def _get_range_async(
    client: ObjectStore, uri: str, start: int, length: int
) -> bytes:
    """Fetch ``length`` bytes from ``uri`` via an attached obstore client."""
    from urllib.parse import urlsplit

    path = urlsplit(uri).path.lstrip("/")
    blob = await client.get_range_async(path, start=start, length=length)
    return bytes(blob)


def _run_coroutine_safely(coro: Any) -> Any:
    """Drive ``coro`` to completion regardless of running-loop state.

    Same pattern as ``geocatalog._src.raster._run_coroutine_safely`` and
    ``geopatcher._src.fields.obstore_cog._run_coroutine_safely``:
    ``asyncio.run`` raises ``RuntimeError`` when nested under a running
    loop (Jupyter, FastAPI handler, ``pytest-asyncio``). Detect that
    case and run on a worker thread with its own loop so the calling
    thread stays sync.
    """
    import threading

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: dict[str, Any] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            result_box["value"] = loop.run_until_complete(coro)
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result_box:
        raise result_box["error"]
    return result_box["value"]


def _read_bytes_local(uri: str, start: int, length: int) -> bytes:
    """Fall-back byte read for local paths and ``file://`` URIs.

    Anything without a ``://`` separator is treated as a plain
    filesystem path — including Windows drive-letter paths like
    ``C:/scene.bin`` that ``urlsplit`` would mis-parse as scheme
    ``"c"``. ``file://`` URIs are stripped to their path component
    via the standard library's ``url2pathname`` so cross-platform
    quoting / drive-letter / UNC conventions Just Work.
    """
    fs_path: str | Path
    if "://" not in uri:
        # Plain filesystem path (POSIX or Windows). Pass through
        # unchanged — open() handles both natively.
        fs_path = uri
    else:
        from urllib.parse import urlsplit
        from urllib.request import url2pathname

        parsed = urlsplit(uri)
        if parsed.scheme == "file":
            fs_path = url2pathname(parsed.path) or uri
        else:
            # Caller passed a remote URI without an attached client —
            # surface it with a clear message rather than silently fall
            # through to local open() which would fail cryptically.
            raise RuntimeError(
                f"SensorReader._read_bytes: URI {uri!r} has a remote scheme "
                "but no obstore client is attached. Call "
                "`set_obstore_client(...)` first, or pass a local path "
                "instead."
            )
    with open(fs_path, "rb") as f:
        f.seek(start)
        return f.read(length)
