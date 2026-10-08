"""`CogSource` — Cloud-Optimized GeoTIFF reads via obstore + async-geotiff.

An opened, tiled COG on object storage. The substrate is async-geotiff's
`GeoTIFF` (over async-tiff) running on an obstore-pooled HTTP/2 client:
upstream parses the header, the georeferencing, the CRS, overviews and
internal masks, and fetches / decodes tiles. This module adds what
upstream leaves to the caller — the **batched** read path.
``read_windows(windows)`` collects every unique COG tile that overlaps
any of the requested windows, so a tile shared by many windows is
fetched and decoded once, then fetches them in concurrent groups of
``_TILES_PER_FETCH`` row-major-adjacent tiles (upstream coalesces each
group's contiguous byte ranges). Dedup cuts requests and bytes when
windows overlap; the concurrent groups keep scattered windows from
serialising behind one request on high-latency stores.

Surface
-------

``CogSource.open(url, ...)`` (``await CogSource.aopen(...)``) opens a
remote COG and exposes:

- ``domain`` — a `CogDomain` with ``crs``, ``transform``, ``shape``,
  ``bounds``, ``res`` and ``nodata``.
- ``read_window(window)`` / ``read_windows(windows)`` — one window, or a
  batch sharing tile fetches; each a ``GeoTensor`` carrying the window's
  transform, the CRS and the nodata fill value (``aread_*`` twins for
  callers inside an event loop).
- ``identity()`` / ``object_version()`` — what the source reads, and the
  object's ETag (else size + last-modified), for caches that must notice
  an object overwritten in place.

The source pickles by URL: unpickling re-opens the COG through
:meth:`CogSource.open` with the same ``ifd_index``, ``storage_options``,
``timeout`` (and ``store`` / ``path`` when one was supplied), so it can
be shipped to process-pool workers.

Georeferencing follows GDAL / rasterio. async-geotiff supplies the
transform (``ModelTransformationTag`` for rotated / sheared grids,
*PixelIsPoint* half-pixel shift, overviews scaled from IFD 0) and the
CRS (EPSG codes and user-defined CRSs spelled out in geokeys). Two
upstream gaps are covered here:

- a raster tiepoint ``(I, J) != (0, 0)`` is honoured
  (``x0 = X - I·sx``, ``y0 = Y + J·sy``); upstream assumes ``(0, 0)``;
- a CRS upstream cannot build (it raises for some user-defined
  datums) becomes ``domain.crs = None`` with a ``RuntimeWarning``
  instead of failing the open.

``GDAL_NODATA`` becomes ``domain.nodata`` / ``fill_value_default``;
out-of-image parts of a window, and pixels the COG's internal mask
marks invalid, are filled with it (``0`` when the COG declares no
nodata) — the same values as a masked GDAL read.

Windows are snapped outward to whole pixels (offsets floored, far
edges ceiled — georeader's ``round_outer_window`` convention) and the
returned transform describes the snapped window exactly.

Codec fidelity: lossless codecs (deflate, LZW, zstd, none, ...) decode
bit-for-bit identical to GDAL. JPEG tiles are decoded by async-tiff's
own JPEG decoder rather than GDAL's libjpeg, so values can differ from
``rasterio.read`` by a few DN — up to ±3 for ``PHOTOMETRIC=YCbCr`` (the
YCbCr→RGB conversion differs) and ±1 for plain-RGB JPEG in our checks.
Compare JPEG reads with a tolerance.

``async-geotiff`` (the ``[cog]`` extra) is required at *open* time —
importing this module is fine without it.
"""

from __future__ import annotations

import asyncio
import json
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import numpy as np

from geocloud._src.aio import _run_coroutine_safely
from geocloud._src.extras import missing_extra


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor
    from rasterio.windows import Window


async def _with_timeout(coro: Any, *, timeout: float | None, message: str) -> Any:
    """Await ``coro``, bounded by ``timeout`` seconds.

    Args:
        coro: The coroutine to drive.
        timeout: Seconds before giving up; ``None`` disables the bound.
        message: What the coroutine was doing — embedded in the error.

    Raises:
        TimeoutError: The coroutine did not finish within ``timeout``
            seconds. Named after ``message`` so a stalled read
            identifies its URL / tile batch instead of hanging the
            calling (or worker) thread forever.
    """
    if timeout is None:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout)
    except TimeoutError:
        raise TimeoutError(
            f"CogSource: {message} timed out after {timeout} s."
        ) from None


def _require_async_geotiff() -> Any:
    try:
        import async_geotiff
    except ImportError as exc:
        raise missing_extra("CogSource", "cog") from exc
    return async_geotiff


def _uri_path(uri: str) -> str:
    """Return the key inside the pooled store for ``uri``.

    Delegates to `geocloud._src.store.object_key`, which handles the
    Azure case (container segment lives in the store, not the key).
    """
    from geocloud._src.store import object_key

    return object_key(uri)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

# Tiles per concurrent ``fetch_tiles`` call in `read_windows`: small enough
# that scattered windows fan out over parallel requests, large enough that
# a cluster of adjacent tiles still coalesces into few range reads.
_TILES_PER_FETCH = 8
# Decimal places used to absorb float noise before snapping window edges
# to whole pixels (same as georeader's ``PIXEL_PRECISION``).
_PIXEL_PRECISION = 3


@dataclass(frozen=True)
class CogDomain:
    """Raster-domain view over a remote COG IFD.

    The I/O-free metadata twin of a `CogSource` (``crs``, ``transform``,
    ``shape``, ``bounds``, ``res``) — the attributes geopatcher's samplers
    and geometries expect from a raster domain — without holding the IFD.

    ``nodata`` is the parsed ``GDAL_NODATA`` tag (cast to the pixel
    dtype's kind), or ``None`` when the COG declares none.
    """

    crs: Any
    transform: Any
    shape: tuple[int, ...]
    bounds: tuple[float, float, float, float]
    res: tuple[float, float]
    nodata: float | int | None = None

    @property
    def fill_value_default(self) -> float | int:
        """Value used for out-of-image pixels: ``nodata``, else ``0``."""
        return 0 if self.nodata is None else self.nodata


def _dtype(geotiff: Any, *, url: str) -> np.dtype:
    """The pixel dtype async-geotiff derives from IFD 0's tags.

    Overviews share it (upstream exposes ``dtype`` on the `GeoTIFF` only).

    Raises:
        ValueError: The SampleFormat / BitsPerSample combination has no
            numpy dtype (upstream reports ``None``). Failing loud beats
            silently reinterpreting pixel bytes under a guessed dtype.
    """
    dtype = geotiff.dtype
    if dtype is None:
        raise ValueError(
            f"CogSource: cannot derive a dtype for {url!r}: "
            f"BitsPerSample={getattr(geotiff.ifd, 'bits_per_sample', None)!r} / "
            f"SampleFormat={getattr(geotiff.ifd, 'sample_format', None)!r} "
            "is not a supported combination."
        )
    return np.dtype(dtype)


def _tiepoint_offset(geotiff: Any) -> tuple[float, float]:
    """Raster-space tiepoint ``(I, J)`` of IFD 0, or ``(0, 0)``.

    async-geotiff anchors the transform at the tiepoint's model
    coordinates as if they belonged to pixel ``(0, 0)``; GDAL honours a
    non-zero ``(I, J)``. Only relevant when the transform comes from a
    tiepoint + pixel scale (a ``ModelTransformationTag`` carries no
    tiepoint).
    """
    ifd = geotiff.ifd
    tiepoint = getattr(ifd, "model_tiepoint", None)
    if tiepoint is None or getattr(ifd, "model_pixel_scale", None) is None:
        return (0.0, 0.0)
    i, j = (float(v) for v in list(tiepoint)[:2])
    return (i, j)


def _crs_or_none(level: Any, *, url: str) -> Any:
    """The COG's CRS from async-geotiff, or ``None`` with a warning.

    Upstream builds EPSG and user-defined CRSs from the geokeys, but
    raises for some user-defined datums it cannot express; that
    must not fail the open — the pixels and transform are still valid.
    """
    try:
        return level.crs
    except Exception as exc:
        # pyproj errors embed the whole PROJJSON; keep the tail (the reason).
        reason = str(exc)
        if len(reason) > 160:
            reason = "…" + reason[-160:]
        warnings.warn(
            f"CogSource: could not build the CRS of {url!r} ({reason}); "
            "domain.crs will be None. Read it with rasterio to get "
            "its full CRS.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def _build_domain(level: Any, geotiff: Any, *, url: str) -> CogDomain:
    """Read transform, CRS and nodata for ``level`` from async-geotiff.

    Args:
        level: The `async_geotiff.GeoTIFF` (full resolution) or
            `async_geotiff.Overview` being read.
        geotiff: The parent `async_geotiff.GeoTIFF` (IFD 0 carries the
            GeoTIFF tags).
        url: The COG's URL, for error / warning messages.
    """
    from rasterio.transform import Affine

    width = int(level.width)
    height = int(level.height)
    transform = level.transform
    i, j = _tiepoint_offset(geotiff)
    if i or j:
        # Shift the origin back from pixel (I, J) to pixel (0, 0), in this
        # level's pixel units (overviews are IFD 0 scaled by the size ratio).
        transform = transform * Affine.translation(
            -i * width / int(geotiff.width), -j * height / int(geotiff.height)
        )

    dtype = _dtype(geotiff, url=url)
    raw_nodata = getattr(level.ifd, "gdal_nodata", None)
    if raw_nodata is None:
        raw_nodata = getattr(geotiff.ifd, "gdal_nodata", None)
    nodata = _parse_nodata(raw_nodata, dtype, url=url)

    corners = [transform * (c, r) for c in (0, width) for r in (0, height)]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    a, b, _c, d, e, _f = transform[:6]
    if b == 0 and d == 0:
        res = (abs(a), abs(e))
    else:
        res = (math.hypot(a, d), math.hypot(b, e))

    return CogDomain(
        crs=_crs_or_none(level, url=url),
        transform=transform,
        shape=(int(level.count), height, width),
        bounds=(min(xs), min(ys), max(xs), max(ys)),
        res=res,
        nodata=nodata,
    )


def _parse_nodata(raw: Any, dtype: np.dtype, *, url: str) -> float | int | None:
    """Parse the ``GDAL_NODATA`` ASCII tag into a value of ``dtype``'s kind.

    GDAL stores nodata as text (``"-9999"``, ``"nan"``, ``"1e+20"``).
    Returns ``None`` when the tag is absent; warns and returns ``None``
    when the value can't be represented in ``dtype`` (e.g. ``nan`` or
    ``-1`` on an unsigned-integer raster) rather than filling with a
    silently wrapped value.
    """
    if raw is None:
        return None
    text = str(raw).strip().strip("\x00").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        warnings.warn(
            f"CogSource: unparseable GDAL_NODATA {text!r} in {url!r}; ignoring it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        if not (np.isfinite(value) and value.is_integer()) or not (
            info.min <= value <= info.max
        ):
            warnings.warn(
                f"CogSource: GDAL_NODATA {text!r} in {url!r} is not "
                f"representable as {dtype}; ignoring it.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        return int(value)
    return value


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


def _level_for_ifd(geotiff: Any, ifd_index: int, ifd: Any, *, url: str) -> Any:
    """The async-geotiff level (`GeoTIFF` or `Overview`) for raw IFD ``ifd_index``.

    ``ifd_index`` keeps counting raw TIFF IFDs (masks included), so
    pickled sources and cache identities stay stable; upstream keys
    overviews by their pixel size.

    Raises:
        ValueError: ``ifd`` is not IFD 0 and matches no overview.
    """
    if ifd_index == 0:
        return geotiff
    size = (int(ifd.image_width), int(ifd.image_height))
    for overview in geotiff.overviews:
        if (int(overview.width), int(overview.height)) == size:
            return overview
    raise ValueError(
        f"CogSource: IFD {ifd_index} of {url!r} is not an overview of IFD 0."
    )


def _store_identity(store: Any) -> str | None:
    """Printable configuration of an obstore store, else ``None``.

    A store whose ``repr`` is only its type and address (``MemoryStore``)
    has no configuration that names its contents.
    """
    text = repr(store)
    if " object at 0x" in text:
        return None
    return text


def _options_digest(options: dict[str, Any] | None) -> str | None:
    """sha256 of ``storage_options`` (sorted keys, ``repr`` values)."""
    if not options:
        return None
    import hashlib

    text = json.dumps(options, sort_keys=True, default=repr)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reopen(cls: type[CogSource], url: str, options: dict[str, Any]) -> CogSource:
    """Unpickle hook: re-open the COG by URL (see `CogSource.__reduce__`)."""
    return cls.open(url, **options)


@dataclass(eq=False)
class CogSource:
    """An opened, tiled COG with batched range-fetch reads.

    Open via :meth:`open` (or ``await`` :meth:`aopen`); the constructor
    takes the parsed handles. ``read_window`` / ``read_windows`` return
    `georeader.GeoTensor` chips with the window's transform, the CRS and
    the COG's nodata as ``fill_value_default``; out-of-image pixels, and
    pixels the COG's internal mask marks invalid, are filled with it. The
    source pickles by URL (see ``__reduce__``).

    Note:
        Lossless codecs match ``rasterio.read`` exactly. JPEG tiles are
        decoded by async-tiff, not GDAL's libjpeg: expect differences of
        a few DN (up to ±3 for ``PHOTOMETRIC=YCbCr``, ±1 for RGB JPEG).

    Args:
        url: Cloud URI the COG was opened from.
        level: The ``async_geotiff.GeoTIFF`` (``ifd_index`` 0) or
            ``async_geotiff.Overview`` being read.
        ifd: The selected ``async_tiff.ImageFileDirectory``
            (``level.ifd``).
        domain: I/O-free metadata twin (see `CogDomain`).
        timeout: Per-network-operation deadline in seconds for tile
            fetch + decode batches (`read_window` / `read_windows`).
            ``None`` disables the bound. On expiry a :class:`TimeoutError`
            naming the URL and tile batch is raised instead of hanging the
            calling (or worker) thread forever on a stalled read.
        ifd_index: Which IFD ``ifd`` is (``0`` = full resolution).
            Recorded so pickling re-opens the same overview.
        storage_options: Options the pooled store was built with;
            recorded for pickling.
        store: The explicit obstore store passed to :meth:`open`, if any
            (``None`` = the process-global pool). Recorded for pickling;
            obstore stores pickle by configuration.
        path: Object key inside ``store`` (only with an explicit store).
        dtype: Pixel dtype (``None`` = derived from ``level``'s parent
            GeoTIFF by :meth:`open`).

    Examples:
        Two overlapping windows share their tile fetches::

            src = CogSource.open("s3://bucket/scene.tif")
            src.domain.shape                     # (bands, H, W)
            a, b = src.read_windows([Window(0, 0, 256, 256),
                                     Window(128, 0, 256, 256)])
    """

    url: str
    level: Any  # async_geotiff.GeoTIFF | async_geotiff.Overview
    ifd: Any  # async_tiff.ImageFileDirectory
    domain: CogDomain
    timeout: float | None = 120.0
    ifd_index: int = 0
    storage_options: dict[str, Any] | None = None
    store: Any = None
    path: str | None = None
    dtype: Any = None

    @classmethod
    def open(
        cls,
        url: str,
        *,
        storage_options: dict[str, Any] | None = None,
        ifd_index: int = 0,
        store: Any = None,
        path: str | None = None,
        timeout: float | None = 120.0,
    ) -> Self:
        """Open a remote COG, parse its IFD, return a ready source.

        Args:
            url: Cloud URI (``s3://``, ``gs://``, ``https://``, …).
                Used both as the pool key and (after stripping scheme/
                bucket) as the object-store key for the file. Ignored
                when ``store`` is supplied — see below.
            storage_options: Forwarded to obstore on the first call
                for the URL's pool key (bucket + region).
            ifd_index: Which IFD to open — ``0`` for the full-resolution
                image, ``1+`` for overviews. Overviews inherit the CRS
                and origin of IFD 0 with the pixel size scaled by the
                size ratio.
            store: Optional pre-built obstore ``ObjectStore`` instance.
                When supplied, bypasses the pool — useful for tests
                with ``LocalStore`` / ``MemoryStore``, or for a custom
                auth / endpoint config that doesn't fit the pool's
                environment-driven keying.
            path: Object key inside ``store``. Required when ``store``
                is supplied; ignored otherwise (derived from ``url``).
            timeout: Deadline in seconds for opening/parsing the COG
                header, and (stored on the source) for each subsequent
                tile fetch + decode batch. ``None`` disables the bound.

        Raises:
            ImportError: ``[cog]`` extra missing.
            ValueError: COG is striped (not tiled), ``ifd_index`` names
                a mask IFD or no IFD, the file carries no GeoTIFF keys,
                or the tags needed to derive an affine transform or a
                dtype are missing.
            TimeoutError: Opening the COG took longer than ``timeout``
                seconds.
        """
        return _run_coroutine_safely(
            cls.aopen(
                url,
                storage_options=storage_options,
                ifd_index=ifd_index,
                store=store,
                path=path,
                timeout=timeout,
            )
        )

    @classmethod
    async def aopen(
        cls,
        url: str,
        *,
        storage_options: dict[str, Any] | None = None,
        ifd_index: int = 0,
        store: Any = None,
        path: str | None = None,
        timeout: float | None = 120.0,
    ) -> Self:
        """Async twin of :meth:`open` — same arguments, same source.

        For callers already inside an event loop (tile servers, async
        inference): awaits the header fetch instead of driving it on a
        helper thread.
        """
        async_geotiff = _require_async_geotiff()

        explicit_store = store
        if store is None:
            from geocloud._src.store import get_obstore

            store = get_obstore(url, storage_options=storage_options)
            object_path = _uri_path(url)
        else:
            if path is None:
                raise ValueError(
                    "CogSource.open: when `store` is supplied, `path` (the key "
                    "inside the store) must also be supplied."
                )
            object_path = path

        # Upstream only warns on striped TIFFs; the tiled check below raises.
        geotiff = await _with_timeout(
            async_geotiff.GeoTIFF.open(object_path, store=store),
            timeout=timeout,
            message=f"opening COG {url!r}",
        )
        ifd = geotiff.tiff.ifd(ifd_index)
        if ifd.tile_width is None or ifd.tile_height is None:
            raise ValueError(
                "CogSource: COG must be tiled (TileWidth + TileLength); "
                "striped TIFFs aren't supported. Read those with rasterio."
            )
        subfile_type = getattr(ifd, "new_subfile_type", None)
        if subfile_type is not None and int(subfile_type) & 4:
            raise ValueError(
                f"CogSource: IFD {ifd_index} of {url!r} is a "
                "transparency mask, not image data."
            )
        level = _level_for_ifd(geotiff, ifd_index, ifd, url=url)
        domain = _build_domain(level, geotiff, url=url)
        return cls(
            url=url,
            level=level,
            ifd=level.ifd,
            domain=domain,
            timeout=timeout,
            ifd_index=ifd_index,
            storage_options=storage_options,
            store=explicit_store,
            path=path if explicit_store is not None else None,
            dtype=_dtype(geotiff, url=url),
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Pickle by URL: the parsed TIFF handle is a Rust object.

        Unpickling calls :meth:`open` on the same class with the recorded
        ``ifd_index`` / ``storage_options`` / ``timeout`` (and the explicit
        ``store`` + ``path`` when one was supplied), so a worker process
        re-reads only the COG header.
        """
        options: dict[str, Any] = {
            "ifd_index": self.ifd_index,
            "storage_options": self.storage_options,
            "timeout": self.timeout,
        }
        if self.store is not None:
            options["store"] = self.store
            options["path"] = self.path
        return (_reopen, (type(self), self.url, options))

    def identity(self) -> dict[str, Any]:
        """What this source reads, as plain JSON-able data (no credentials).

        With the process pool the object is the ``url`` (plus a digest of
        ``storage_options``: the same url can name different objects under
        different endpoints); with an explicit ``store`` the ``url`` is
        only a label, so the identity is the store's printable
        configuration (e.g. ``LocalStore("/data")``) plus ``path``.
        ``store`` is ``None`` for an explicit store without one
        (``MemoryStore``, whose contents live only in that instance).

        Returns:
            ``{"url", "options", "store", "path", "ifd_index"}``.
        """
        explicit = self.store is not None
        return {
            "url": None if explicit else self.url,
            "options": None if explicit else _options_digest(self.storage_options),
            "store": _store_identity(self.store) if explicit else None,
            "path": self.path if explicit else None,
            "ifd_index": int(self.ifd_index),
        }

    def object_version(self) -> str | None:
        """ETag (else ``size:last_modified``) of the object, or ``None``.

        One ``HEAD`` request. When the store refuses it, or answers with
        neither an ETag nor a last-modified time, a `RuntimeWarning` says
        changes to the object cannot be detected and ``None`` is returned.
        """
        if self.store is not None:
            store, key = self.store, self.path
        else:
            from geocloud._src.store import get_obstore

            store = get_obstore(self.url, storage_options=self.storage_options)
            key = _uri_path(self.url)
        try:
            meta = store.head(key)
        except Exception as exc:
            warnings.warn(
                f"CogSource: HEAD of {self.url!r} failed ({exc}); changes to "
                f"the object will not be detected.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        e_tag = meta.get("e_tag")
        if e_tag:
            return f"etag:{e_tag}"
        last_modified = meta.get("last_modified")
        if last_modified is None:
            # Size alone cannot tell an overwrite of the same length apart.
            warnings.warn(
                f"CogSource: HEAD of {self.url!r} returned neither an ETag nor "
                f"a Last-Modified time; changes to the object will not be "
                f"detected.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        return f"size:{meta.get('size')}:{last_modified}"

    @property
    def fill_value_default(self) -> float | int:
        """Fill for out-of-image pixels and padding: nodata, else ``0``."""
        return self.domain.fill_value_default

    def read_window(self, window: Window) -> GeoTensor:
        """Read one window via the COG's tile grid.

        A thin wrapper around ``read_windows([window])``, so the
        single-window path goes through the same tile-coalescing code as
        the batched path.
        """
        return self.read_windows([window])[0]

    async def aread_window(self, window: Window) -> GeoTensor:
        """Async twin of :meth:`read_window`."""
        return (await self.aread_windows([window]))[0]

    def read_windows(self, windows: list[Window]) -> list[GeoTensor]:
        """Bulk-read every window, fetching each overlapping tile once.

        Sync facade over :meth:`aread_windows` (safe inside a running
        event loop — the coroutine then runs on a helper thread); see
        there for the read semantics.

        Raises:
            TimeoutError: The tile fetch + decode did not finish within
                ``self.timeout`` seconds.
        """
        return _run_coroutine_safely(self.aread_windows(windows))

    async def aread_windows(self, windows: list[Window]) -> list[GeoTensor]:
        """Async bulk read: every window, each overlapping tile fetched once.

        The headline path: collect every unique tile coordinate
        across all windows, fetch + decode them in concurrent groups
        (see `_fetch_and_decode_tiles`), then assemble per-window
        arrays by cropping each decoded tile to its window's
        intersection.

        Args:
            windows: Sequence of ``rasterio.windows.Window`` to read.
                Fractional windows are snapped outward to whole pixels;
                windows may extend past (or lie entirely outside) the
                image — those pixels get ``fill_value_default``.

        Returns:
            One ``GeoTensor`` per input window, in input order, each
            shaped ``(bands, height, width)`` with the (snapped)
            window's transform, the domain CRS and
            ``fill_value_default`` set to the COG's nodata.

        Raises:
            TimeoutError: The batched tile fetch + decode did not
                finish within ``self.timeout`` seconds.
        """
        from georeader.geotensor import GeoTensor
        from rasterio.transform import Affine

        if len(windows) == 0:
            return []

        tile_w = int(self.ifd.tile_width)
        tile_h = int(self.ifd.tile_height)
        image_w = int(self.ifd.image_width)
        image_h = int(self.ifd.image_height)

        # Collect the unique tile coordinates spanned by all windows.
        pixel_windows = [_snap_window(w) for w in windows]
        tile_coords: dict[tuple[int, int], None] = {}
        per_window_tile_ranges: list[tuple[int, int, int, int]] = []
        for w in pixel_windows:
            ranges = _tile_range_for_window(
                w,
                tile_w=tile_w,
                tile_h=tile_h,
                image_w=image_w,
                image_h=image_h,
            )
            per_window_tile_ranges.append(ranges)
            tx_min, ty_min, tx_max, ty_max = ranges
            for ty in range(ty_min, ty_max + 1):
                for tx in range(tx_min, tx_max + 1):
                    tile_coords[(tx, ty)] = None

        coord_list = list(tile_coords.keys())
        decoded = await _with_timeout(
            _fetch_and_decode_tiles(self.level, coord_list),
            timeout=self.timeout,
            message=(
                f"fetching/decoding a batch of {len(coord_list)} tiles "
                f"from {self.url!r}"
            ),
        )
        # Map decoded tiles by coord for the assembly loop.
        tile_data: dict[tuple[int, int], np.ndarray] = dict(
            zip(coord_list, decoded, strict=True)
        )

        # Derive (bands, dtype) from the metadata so the empty-tile-range
        # path (window entirely outside the image) returns an array of
        # the right shape/dtype even when no tile was fetched.
        bands = int(self.domain.shape[0])
        dtype = np.dtype(self.dtype)
        fill = self.fill_value_default

        results: list[GeoTensor] = []
        for window, (tx_min, ty_min, tx_max, ty_max) in zip(
            pixel_windows, per_window_tile_ranges, strict=True
        ):
            values = _assemble_window(
                window,
                tile_data=tile_data,
                tx_min=tx_min,
                ty_min=ty_min,
                tx_max=tx_max,
                ty_max=ty_max,
                tile_w=tile_w,
                tile_h=tile_h,
                image_w=image_w,
                image_h=image_h,
                bands=bands,
                dtype=dtype,
                fill=fill,
            )
            results.append(
                GeoTensor(
                    values=values,
                    transform=self.domain.transform
                    * Affine.translation(window.col_off, window.row_off),
                    crs=self.domain.crs,
                    fill_value_default=fill,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Tile arithmetic
# ---------------------------------------------------------------------------


class _PixelWindow(NamedTuple):
    """Integer pixel window (duck-types ``rasterio.windows.Window``)."""

    col_off: int
    row_off: int
    width: int
    height: int


def _snap_window(window: Any) -> _PixelWindow:
    """Snap ``window`` outward to whole pixels.

    Offsets are floored and far edges ceiled (after rounding to
    ``_PIXEL_PRECISION`` decimals to absorb float noise) — the same
    convention as ``georeader.window_utils.round_outer_window``.
    ``int()`` truncation toward zero would shift negative fractional
    offsets right and shrink windows.

    Raises:
        ValueError: The window has a negative width or height.
    """
    col0 = math.floor(round(float(window.col_off), _PIXEL_PRECISION))
    row0 = math.floor(round(float(window.row_off), _PIXEL_PRECISION))
    col1 = math.ceil(
        round(float(window.col_off) + float(window.width), _PIXEL_PRECISION)
    )
    row1 = math.ceil(
        round(float(window.row_off) + float(window.height), _PIXEL_PRECISION)
    )
    if col1 < col0 or row1 < row0:
        raise ValueError(
            f"CogSource: window {window!r} has a negative width or height."
        )
    return _PixelWindow(col0, row0, col1 - col0, row1 - row0)


def _tile_range_for_window(
    window: Window | _PixelWindow,
    *,
    tile_w: int,
    tile_h: int,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    """Return ``(tx_min, ty_min, tx_max, ty_max)`` for a window.

    Clamps to the image's tile-coverage grid; out-of-image regions of
    the window are filled with the nodata value by the assembly step.
    """
    w = _snap_window(window)
    col_off = max(0, w.col_off)
    row_off = max(0, w.row_off)
    col_end = min(image_w, w.col_off + w.width)
    row_end = min(image_h, w.row_off + w.height)
    if col_end <= col_off or row_end <= row_off:
        # Window is entirely outside the image — empty tile range.
        return (0, 0, -1, -1)
    tx_min = col_off // tile_w
    ty_min = row_off // tile_h
    tx_max = (col_end - 1) // tile_w
    ty_max = (row_end - 1) // tile_h
    return tx_min, ty_min, tx_max, ty_max


async def _fetch_and_decode_tiles(
    level: Any, coords: list[tuple[int, int]]
) -> list[np.ndarray]:
    """Fetch + decode ``coords`` in concurrent groups of adjacent tiles.

    The (already de-duplicated) tiles are sorted row-major — the order
    COGs store them in — and split into groups of ``_TILES_PER_FETCH``.
    Each group is one upstream ``fetch_tiles`` call, which coalesces
    the group's contiguous byte ranges and decodes on async-tiff's
    thread pool; the groups run concurrently. Tiles come back
    band-first ``(bands, H, W)`` whatever the planar configuration.

    Returns:
        One array per entry of ``coords``, in ``coords`` order. Where the
        COG has an internal mask the array is a `numpy.ma.MaskedArray`
        whose masked pixels the assembly step fills.
    """
    if not coords:
        return []
    ordered = sorted(coords, key=lambda xy: (xy[1], xy[0]))
    groups = [
        ordered[i : i + _TILES_PER_FETCH]
        for i in range(0, len(ordered), _TILES_PER_FETCH)
    ]
    batches = await asyncio.gather(*(level.fetch_tiles(group) for group in groups))
    by_coord: dict[tuple[int, int], np.ndarray] = {}
    for tile in (t for batch in batches for t in batch):
        data = np.asarray(tile.array.data)
        valid = tile.array.mask
        if valid is not None:
            # async-geotiff masks are True where valid.
            data = np.ma.MaskedArray(data, mask=np.broadcast_to(~valid, data.shape))
        by_coord[(tile.x, tile.y)] = data
    return [by_coord[xy] for xy in coords]


def _assemble_window(
    window: _PixelWindow,
    *,
    tile_data: dict[tuple[int, int], np.ndarray],
    tx_min: int,
    ty_min: int,
    tx_max: int,
    ty_max: int,
    tile_w: int,
    tile_h: int,
    image_w: int,
    image_h: int,
    bands: int,
    dtype: np.dtype,
    fill: float | int = 0,
) -> np.ndarray:
    """Crop the relevant tiles into a single window-shaped array.

    ``bands`` and ``dtype`` come from the domain and :func:`_dtype`,
    so the empty-tile-range fallback (window
    entirely outside the image) returns an array of the right shape
    even when no tile was decoded — preserving the documented
    ``(bands, h, w)`` contract regardless of batch composition.

    Copies are clamped to the image extent: edge tiles are stored at
    full tile size, and their padding past the image edge must not leak
    into the window — those pixels keep ``fill`` (the nodata value), as
    do pixels masked invalid in a `numpy.ma.MaskedArray` tile.
    """
    col_off, row_off, w, h = window
    out = np.full((bands, h, w), fill, dtype=dtype)

    if tx_max < tx_min or ty_max < ty_min:
        # Empty tile range — window is outside the image.
        return out

    col_stop = min(col_off + w, image_w)
    row_stop = min(row_off + h, image_h)
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            tile = tile_data[(tx, ty)]
            # Tile occupies pixel range [tx*tile_w, (tx+1)*tile_w) x
            # [ty*tile_h, (ty+1)*tile_h). Intersect with the window and
            # the image.
            tile_col_start = tx * tile_w
            tile_row_start = ty * tile_h
            inter_col_start = max(tile_col_start, col_off, 0)
            inter_row_start = max(tile_row_start, row_off, 0)
            inter_col_end = min(tile_col_start + tile_w, col_stop)
            inter_row_end = min(tile_row_start + tile_h, row_stop)
            if inter_col_end <= inter_col_start or inter_row_end <= inter_row_start:
                continue
            # Source slice within tile-local coords.
            src_c0 = inter_col_start - tile_col_start
            src_r0 = inter_row_start - tile_row_start
            src_c1 = inter_col_end - tile_col_start
            src_r1 = inter_row_end - tile_row_start
            # Destination slice within window-local coords.
            dst_c0 = inter_col_start - col_off
            dst_r0 = inter_row_start - row_off
            dst_c1 = inter_col_end - col_off
            dst_r1 = inter_row_end - row_off
            block = tile[..., src_r0:src_r1, src_c0:src_c1]
            if isinstance(block, np.ma.MaskedArray):
                block = block.filled(fill)
            out[..., dst_r0:dst_r1, dst_c0:dst_c1] = block
    return out
