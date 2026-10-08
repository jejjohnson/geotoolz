"""Shared reader primitives for sensor-specific modules."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from affine import Affine
from georeader import read
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor
from rasterio.windows import Window, transform as window_transform


if TYPE_CHECKING:
    from obstore.store import ObjectStore


Track = Literal["A", "B"]


class ProductReader(GeoData, ABC):
    """ABC for georeader-compatible product readers.

    Subclasses provide sensor-specific metadata and implement
    :meth:`_read_window`; the base class supplies the small GeoData surface
    needed by geotoolz operators.

    Examples:
        Implement a file-backed product reader::

            class Reader(ProductReader):
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
    def dims(self) -> tuple[str, ...]:
        """Dimension names, mirroring ``georeader.GeoTensor.dims``.

        ``("time", "band", "y", "x")[-ndim:]`` — i.e. ``("y", "x")`` for 2-D,
        ``("band", "y", "x")`` for 3-D and ``("time", "band", "y", "x")``
        for 4-D readers.

        Raises:
            ValueError: If the reader shape is not 2-D, 3-D or 4-D.
        """
        ndim = len(self.shape)
        if ndim not in (2, 3, 4):
            raise ValueError(
                f"ProductReader expects a 2d-4d array shape; got {self.shape}."
            )
        return ("time", "band", "y", "x")[-ndim:]

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
        """Reader track classification."""
        return self._track

    def load(self, boundless: bool = True) -> GeoTensor:
        """Load the reader's full extent into a ``GeoTensor``."""
        window = Window(col_off=0, row_off=0, width=self.width, height=self.height)
        return self.read_from_window(window, boundless=boundless)

    def _band_attrs(self) -> dict[str, Any]:
        """Extra ``attrs`` every read carries (per-band ``units``, ``wavelengths``, …).

        ``band_names`` is always set; readers override this to add more.
        """
        return {}

    def read_from_window(self, window: Window, boundless: bool = True) -> GeoTensor:
        """Read a pixel window as a ``GeoTensor``.

        ``attrs`` carry ``band_names`` plus whatever :meth:`_band_attrs`
        adds.
        """
        if not boundless:
            window = self._clip_window(window)
        values = self._read_window(window)
        # The canonical key every band resolver reads first (see
        # geotoolz._src.bands.DEFAULT_BAND_KEYS).
        attrs = {"band_names": self.bands, **self._band_attrs()}
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
        crs_bounds: Any = None,
    ) -> GeoTensor:
        """Read map-coordinate bounds as a ``GeoTensor``.

        Thin wrapper over :func:`georeader.read.read_from_bounds` so the
        window maths (outward rounding, CRS transform of ``bounds``) match
        georeader exactly.
        """
        return read.read_from_bounds(
            self, bounds, crs_bounds=crs_bounds, boundless=boundless
        )

    def read_from_center_coords(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        boundless: bool = True,
        crs_center_coords: Any = None,
    ) -> GeoTensor:
        """Read a ``height x width`` window centered on map coordinates.

        Thin wrapper over :func:`georeader.read.read_from_center_coords`
        so the window placement matches georeader exactly.
        """
        return read.read_from_center_coords(
            self,
            (x, y),
            (height, width),
            crs_center_coords=crs_center_coords,
            boundless=boundless,
        )

    def _read_boundless(
        self, window: Window, read: Callable[[slice, slice], np.ndarray]
    ) -> np.ndarray:
        """Read ``window`` through ``read``, padding outside the grid with the fill.

        The shared body of a ``_read_window``: allocates the output filled
        with ``fill_value_default``, asks ``read(rows, cols)`` only for the
        part of the window inside the grid (returning ``(..., h, w)``), and
        pastes it in place. Windows partly or wholly outside the grid
        (boundless reads) are handled here once for every reader.

        Args:
            window: Pixel window, possibly extending past the grid.
            read: Reads the in-grid ``rows`` / ``cols`` slices of every band.

        Returns:
            ``(..., window.height, window.width)`` array of ``self.dtype``.
        """
        row0, col0 = int(window.row_off), int(window.col_off)
        n_rows, n_cols = int(window.height), int(window.width)
        out = np.full(
            (*self.shape[:-2], n_rows, n_cols),
            self.fill_value_default,
            dtype=self.dtype,
        )
        r_start, r_stop = max(row0, 0), min(row0 + n_rows, self.height)
        c_start, c_stop = max(col0, 0), min(col0 + n_cols, self.width)
        if r_start >= r_stop or c_start >= c_stop:
            return out
        out[..., r_start - row0 : r_stop - row0, c_start - col0 : c_stop - col0] = read(
            slice(r_start, r_stop), slice(c_start, c_stop)
        )
        return out

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
        :class:`obstore.store.LocalStore`) or fetch the process-wide
        pooled one from ``geopatcher.objstore.get_obstore`` (the
        ``[obstore]`` extra). A pre-built client must be laid out like
        the pooled one — rooted at the bucket / Azure container, with
        no prefix — because reads request
        ``geopatcher.objstore.object_key(uri)``.
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
           existing on-disk product readers (the ``toy_sensor``
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


def resolve_fill_value(
    fill_value: float | None, dtype: np.dtype, *, owner: str = "reader"
) -> float | int | bool:
    """A fill value exactly representable in ``dtype``.

    ``None`` defaults to ``NaN`` for inexact dtypes and ``0`` otherwise.
    An explicit value that would change when cast to ``dtype`` (``NaN``,
    fractional or out-of-range values for integer data) raises instead of
    silently padding with a different value than ``fill_value_default``
    reports.

    Args:
        fill_value: Requested fill, or ``None`` for the default.
        dtype: The reader's data dtype.
        owner: Name used in the error message.

    Raises:
        ValueError: ``fill_value`` is not representable in ``dtype``.

    Examples:
        >>> resolve_fill_value(None, np.dtype("uint8"))
        0
        >>> resolve_fill_value(255, np.dtype("uint8"))
        255
    """
    if fill_value is None:
        return np.nan if np.issubdtype(dtype, np.inexact) else dtype.type(0).item()
    if np.issubdtype(dtype, np.inexact):
        return fill_value
    try:
        with np.errstate(invalid="ignore", over="ignore"):
            cast = dtype.type(fill_value)
        ok = bool(np.isfinite(fill_value)) and cast == fill_value
    except (OverflowError, TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(
            f"{owner}: fill_value_default={fill_value!r} is not "
            f"representable in data dtype {dtype}; pass a value that "
            "survives the cast (e.g. 0) or use floating-point data."
        )
    return cast.item()


# ----------------------------------------------------------------------
# Byte-range helpers for the ProductReader._read_bytes opt-in path.
# ----------------------------------------------------------------------

# URI schemes the obstore pool (``geopatcher.objstore.SUPPORTED_SCHEMES``)
# can talk to, spelled out so the check needs no optional import.
# ``file://`` is intentionally omitted — local files take the fast on-disk
# path even when a client is attached.
_REMOTE_SCHEMES = frozenset(
    {"s3", "s3a", "gs", "gcs", "az", "azure", "abfs", "abfss", "http", "https", "hf"}
)

_OBSTORE_INSTALL_HINT = (
    "ProductReader cloud reads need the [obstore] extra (the shared pool lives "
    "in geopatcher); install via `pip install 'geotoolz-products[obstore]'`."
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
    try:
        from geopatcher.objstore import object_key
    except ImportError as exc:
        raise ImportError(_OBSTORE_INSTALL_HINT) from exc

    # Same key derivation as the shared pool: for Azure the container is
    # bound into the store, and an http(s) query lives in the store's URL.
    blob = await client.get_range_async(object_key(uri), start=start, length=length)
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
                f"ProductReader._read_bytes: URI {uri!r} has a remote scheme "
                "but no obstore client is attached. Call "
                "`set_obstore_client(...)` first, or pass a local path "
                "instead."
            )
    with open(fs_path, "rb") as f:
        f.seek(start)
        return f.read(length)
