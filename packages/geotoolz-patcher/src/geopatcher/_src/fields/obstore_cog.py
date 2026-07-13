"""`ObstoreCogField` — Cloud-Optimized GeoTIFF reads via obstore + async-tiff.

A `Field` adapter for tiled COGs hosted on object storage. The
substrate is async-tiff's `TIFF` parser running over an obstore-pooled
HTTP/2 client. The whole point of this class — and the reason the
plain `RasterField` isn't enough — is the **batched** read path:
``select_many(windows)`` collects every unique COG tile that overlaps
any of the requested windows and fetches them in *one* batched range
request, instead of one HTTP round trip per window. For a
``parallel_map`` over hundreds of patches on a single COG this is the
>=5x wall-clock win the integration plan promised.

Adapted from ``openEO-RuSTAC/crates/orbit-geo/src/async_download.rs:1073-1091``
— the upstream Rust pattern that motivated this PR.

Single-patch reads still work (``select(window)``), so the class is a
drop-in for `RasterField` when the runner doesn't know to batch. The
duck-type sniff in :func:`geopatcher.runners.parallel_map` picks the
batched path automatically when both the field and the patcher
support it.

Surface
-------

``ObstoreCogField.from_url(url, ...)`` opens a remote COG and exposes:

- ``domain`` — an ``ObstoreCogDomain`` with ``crs``, ``transform``,
  ``shape``, ``bounds``, ``res`` (mirrors the GeoData surface
  ``RasterField`` relies on).
- ``select(window)`` — single-window read; collects, fetches, and
  decodes the relevant tiles.
- ``select_many(windows)`` — bulk read over a list of windows; the
  batched fast path.
- ``with_data(array)`` — reconstruct a ``GeoTensor`` from operator
  output (delegates to georeader).

Both extras (``obstore`` and ``async-tiff``) are required at *call*
time — importing this module is fine without them, but
:meth:`ObstoreCogField.from_url` invokes the internal
``_require_async_tiff`` guard which raises :class:`ImportError` with
the install hint if either is missing. The lazy check keeps
``from geopatcher.fields import ObstoreCogField`` cheap on a slim
install (the lazy export in ``_src.fields.__init__`` doesn't import
this module unless the name is actually accessed).
"""

from __future__ import annotations

import asyncio
import threading
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from rasterio.windows import Window


def _run_coroutine_safely(coro: Any) -> Any:
    """Drive ``coro`` to completion regardless of running-loop state.

    Same pattern as ``geocatalog._src.raster._run_coroutine_safely``:
    ``asyncio.run`` raises ``RuntimeError`` when nested under a running
    loop (Jupyter, FastAPI handler, pytest-asyncio). Detect that case
    and run on a worker thread with its own loop so the calling thread
    stays sync.
    """
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
            f"ObstoreCogField: {message} timed out after {timeout} s."
        ) from None


_INSTALL_HINT = (
    "ObstoreCogField requires the [obstore-cog] extra; install via "
    "`pip install 'geopatcher[obstore-cog]'`."
)


def _require_async_tiff() -> Any:
    try:
        import async_tiff  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return async_tiff


def _uri_path(uri: str) -> str:
    """Return the key inside the pooled store for ``uri``.

    Delegates to `geopatcher._src.objstore.object_key`, which handles
    the Azure case (container segment lives in the store, not the key).
    """
    from geopatcher._src.objstore import object_key

    return object_key(uri)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObstoreCogDomain:
    """RasterDomain-shaped view over a remote COG IFD.

    Exposes the attributes the geopatcher samplers and geometries
    expect from a raster domain (``crs``, ``transform``, ``shape``,
    ``bounds``, ``res``) without holding the IFD object — the domain
    is the I/O-free metadata twin, by Protocol contract.
    """

    crs: Any
    transform: Any
    shape: tuple[int, ...]
    bounds: tuple[float, float, float, float]
    res: tuple[float, float]


def _dtype_from_ifd(ifd: Any, *, url: str) -> np.dtype:
    """Derive a numpy dtype from the IFD's sample-format + bit-depth tags.

    Args:
        ifd: The async-tiff ImageFileDirectory.
        url: The COG's URL — used to name the file in errors.

    Raises:
        ValueError: The BitsPerSample / SampleFormat tags are missing,
            unparseable, or describe an unsupported combination.
            Failing loud beats silently reinterpreting pixel bytes
            under a guessed dtype.
    """
    try:
        bps_raw = ifd.bits_per_sample
        sf_raw = ifd.sample_format
        # Both come back as lists (one entry per sample); we use the
        # first sample's spec because COGs uniformly type all samples.
        bps = int(bps_raw[0]) if hasattr(bps_raw, "__getitem__") else int(bps_raw)
        sf = sf_raw[0] if hasattr(sf_raw, "__getitem__") else sf_raw
        # ``async_tiff.enums.SampleFormat`` exposes a ``.value`` int.
        sf_int = int(getattr(sf, "value", sf))
    except (TypeError, ValueError, AttributeError, IndexError) as exc:
        raise ValueError(
            f"ObstoreCogField: cannot derive a dtype for {url!r}: "
            f"BitsPerSample={getattr(ifd, 'bits_per_sample', None)!r} / "
            f"SampleFormat={getattr(ifd, 'sample_format', None)!r} "
            f"could not be interpreted ({exc})."
        ) from exc

    # SampleFormat: 1 = unsigned int, 2 = signed int, 3 = float.
    prefix = {1: "uint", 2: "int", 3: "float"}.get(sf_int)
    if prefix is None:
        raise ValueError(
            f"ObstoreCogField: unsupported SampleFormat {sf_int!r} in {url!r} "
            "(expected 1=unsigned int, 2=signed int, 3=float)."
        )
    try:
        return np.dtype(f"{prefix}{bps}")
    except TypeError as exc:
        raise ValueError(
            f"ObstoreCogField: unsupported BitsPerSample {bps!r} for "
            f"SampleFormat {sf_int!r} in {url!r} (no numpy dtype "
            f"'{prefix}{bps}')."
        ) from exc


def _build_domain(ifd: Any) -> ObstoreCogDomain:
    """Read transform + CRS from the IFD's GeoTIFF tags."""
    from rasterio.transform import Affine

    width = int(ifd.image_width)
    height = int(ifd.image_height)
    samples = int(ifd.samples_per_pixel)

    geo_keys = ifd.geo_key_directory
    crs = _crs_from_geokeys(geo_keys)

    # ModelTiepointTag + ModelPixelScaleTag → affine transform. Most
    # COGs encode the upper-left corner and pixel size this way; the
    # full ModelTransformationTag is rare in COG outputs.
    tiepoint = list(ifd.model_tiepoint) if ifd.model_tiepoint is not None else None
    pixel_scale = (
        list(ifd.model_pixel_scale) if ifd.model_pixel_scale is not None else None
    )
    if tiepoint is None or pixel_scale is None:
        raise ValueError(
            "ObstoreCogField: COG IFD lacks ModelTiepointTag + ModelPixelScaleTag; "
            "cannot derive affine transform. Use rasterio for non-COG TIFFs."
        )
    _i, _j, _k, x_origin, y_origin, _z = tiepoint[:6]
    sx, sy, _sz = pixel_scale[:3]
    transform = Affine(sx, 0.0, x_origin, 0.0, -sy, y_origin)

    minx = x_origin
    maxy = y_origin
    maxx = x_origin + sx * width
    miny = y_origin - sy * height

    return ObstoreCogDomain(
        crs=crs,
        transform=transform,
        shape=(samples, height, width),
        bounds=(minx, miny, maxx, maxy),
        res=(sx, sy),
    )


def _crs_from_geokeys(geo_keys: Any) -> Any:
    """Best-effort CRS extraction from an async-tiff GeoKeyDirectory.

    Handles the common cases: an EPSG ProjectedCSTypeGeoKey
    (``projected_type``) or GeographicTypeGeoKey (``geographic_type``).
    Falls back to ``None`` for exotic GeoTIFFs — with a
    ``RuntimeWarning`` when an EPSG code was present but unusable — so
    the user can re-wrap with ``RasterField`` if needed.
    """
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    if geo_keys is None:
        return None
    epsg = getattr(geo_keys, "projected_type", None) or getattr(
        geo_keys, "geographic_type", None
    )
    if epsg is None:
        return None
    try:
        return CRS.from_epsg(int(epsg))
    except (TypeError, ValueError, CRSError) as exc:
        warnings.warn(
            f"ObstoreCogField: could not build a CRS from GeoTIFF key "
            f"EPSG:{epsg!r} ({exc}); domain.crs will be None.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class ObstoreCogField:
    """Tiled-COG `Field` with batched range-fetch reads.

    Open via :meth:`from_url`; constructor takes the parsed handles.

    Args:
        url: Cloud URI the COG was opened from.
        tiff: Parsed ``async_tiff.TIFF`` handle.
        ifd: The selected ``async_tiff.ImageFileDirectory``.
        domain: I/O-free metadata twin (see `ObstoreCogDomain`).
        timeout: Per-network-operation deadline in seconds for tile
            fetch + decode batches (`select` / `select_many`). ``None``
            disables the bound. On expiry a :class:`TimeoutError` naming
            the URL and tile batch is raised instead of hanging the
            calling (or worker) thread forever on a stalled read.
    """

    url: str
    tiff: Any  # async_tiff.TIFF
    ifd: Any  # async_tiff.ImageFileDirectory
    domain: ObstoreCogDomain
    timeout: float | None = 120.0

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        storage_options: dict[str, Any] | None = None,
        ifd_index: int = 0,
        store: Any = None,
        path: str | None = None,
        timeout: float | None = 120.0,
    ) -> ObstoreCogField:
        """Open a remote COG, parse its IFD, return a ready field.

        Args:
            url: Cloud URI (``s3://``, ``gs://``, ``https://``, …).
                Used both as the pool key and (after stripping scheme/
                bucket) as the object-store key for the file. Ignored
                when ``store`` is supplied — see below.
            storage_options: Forwarded to obstore on the first call
                for the URL's pool key (bucket + region).
            ifd_index: Which IFD to open — ``0`` for the full-resolution
                image, ``1+`` for overviews. v1 doesn't auto-select by
                resolution; if you need overview pyramid selection,
                open multiple fields and dispatch in user code.
            store: Optional pre-built obstore ``ObjectStore`` instance.
                When supplied, bypasses the pool — useful for tests
                with ``LocalStore`` / ``MemoryStore``, or for advanced
                users who want a custom auth / endpoint config that
                doesn't fit the pool's environment-driven keying.
            path: Object key inside ``store``. Required when ``store``
                is supplied; ignored otherwise (derived from ``url``).
            timeout: Deadline in seconds for opening/parsing the COG
                header, and (stored on the field) for each subsequent
                tile fetch + decode batch. ``None`` disables the bound.

        Raises:
            ImportError: ``[obstore-cog]`` extra missing.
            ValueError: COG is striped (not tiled) or lacks the
                GeoTIFF tags needed to derive an affine transform.
            TimeoutError: Opening the COG took longer than ``timeout``
                seconds.
        """
        async_tiff = _require_async_tiff()

        if store is None:
            from geopatcher._src.objstore import get_obstore

            store = get_obstore(url, storage_options=storage_options)
            object_path = _uri_path(url)
        else:
            if path is None:
                raise ValueError(
                    "ObstoreCogField.from_url: when `store` is supplied, "
                    "`path` (the key inside the store) must also be supplied."
                )
            object_path = path

        async def _open() -> Any:
            return await _with_timeout(
                async_tiff.TIFF.open(object_path, store=store),
                timeout=timeout,
                message=f"opening COG {url!r}",
            )

        tiff = _run_coroutine_safely(_open())
        ifd = tiff.ifd(ifd_index)
        if ifd.tile_width is None or ifd.tile_height is None:
            raise ValueError(
                "ObstoreCogField: COG must be tiled (TileWidth + TileLength); "
                "striped TIFFs aren't supported. Use RasterField for those."
            )
        domain = _build_domain(ifd)
        return cls(url=url, tiff=tiff, ifd=ifd, domain=domain, timeout=timeout)

    def select(self, window: Window) -> np.ndarray:
        """Read one window via the COG's tile grid.

        Implemented as a thin wrapper around ``select_many([window])`` —
        keeps the single-window path going through the same
        tile-coalescing code as the batched path, so there's no
        divergence in semantics.
        """
        return self.select_many([window])[0]

    def select_many(self, windows: list[Window]) -> list[np.ndarray]:
        """Bulk-read every window via one batched tile fetch.

        The headline path: collect every unique tile coordinate
        across all windows, dispatch a single ``ifd.fetch_tiles``
        call, then assemble per-window arrays by cropping each
        decoded tile to its window's intersection.

        Args:
            windows: Sequence of ``rasterio.windows.Window`` to read.

        Returns:
            One ndarray per input window, in input order, each shaped
            ``(bands, height, width)`` matching the window.

        Raises:
            TimeoutError: The batched tile fetch + decode did not
                finish within ``self.timeout`` seconds.
        """
        if len(windows) == 0:
            return []

        tile_w = int(self.ifd.tile_width)
        tile_h = int(self.ifd.tile_height)
        image_w = int(self.ifd.image_width)
        image_h = int(self.ifd.image_height)

        # Collect the unique tile coordinates spanned by all windows.
        tile_coords: dict[tuple[int, int], None] = {}
        per_window_tile_ranges: list[tuple[int, int, int, int]] = []
        for w in windows:
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
        # Reference the IFD attribute via a local so a monkeypatched
        # ``ifd.fetch_tiles`` (test hook) is picked up correctly.
        ifd = self.ifd
        decoded = _run_coroutine_safely(
            _with_timeout(
                _fetch_and_decode_tiles(ifd, coord_list),
                timeout=self.timeout,
                message=(
                    f"fetching/decoding a batch of {len(coord_list)} tiles "
                    f"from {self.url!r}"
                ),
            )
        )
        # Map decoded tiles by coord for the assembly loop.
        tile_data: dict[tuple[int, int], np.ndarray] = dict(
            zip(coord_list, decoded, strict=True)
        )

        # Derive (bands, dtype) from the IFD so the empty-tile-range
        # path (window entirely outside the image) returns an array of
        # the right shape/dtype even when no tile was fetched.
        bands = int(self.ifd.samples_per_pixel)
        dtype = _dtype_from_ifd(self.ifd, url=self.url)

        results: list[np.ndarray] = []
        for window, (tx_min, ty_min, tx_max, ty_max) in zip(
            windows, per_window_tile_ranges, strict=True
        ):
            results.append(
                _assemble_window(
                    window,
                    tile_data=tile_data,
                    tx_min=tx_min,
                    ty_min=ty_min,
                    tx_max=tx_max,
                    ty_max=ty_max,
                    tile_w=tile_w,
                    tile_h=tile_h,
                    bands=bands,
                    dtype=dtype,
                )
            )
        return results

    def with_data(self, array: np.ndarray) -> Any:
        """Wrap an operator output as a `georeader.GeoTensor`."""
        from georeader.geotensor import GeoTensor

        return GeoTensor(
            values=array,
            transform=self.domain.transform,
            crs=self.domain.crs,
        )


# ---------------------------------------------------------------------------
# Tile arithmetic
# ---------------------------------------------------------------------------


def _tile_range_for_window(
    window: Window,
    *,
    tile_w: int,
    tile_h: int,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    """Return ``(tx_min, ty_min, tx_max, ty_max)`` for a window.

    Clamps to the image's tile-coverage grid; out-of-image regions of
    the window are filled with zeros by the assembly step.
    """
    col_off = max(0, int(window.col_off))
    row_off = max(0, int(window.row_off))
    col_end = min(image_w, int(window.col_off) + int(window.width))
    row_end = min(image_h, int(window.row_off) + int(window.height))
    if col_end <= col_off or row_end <= row_off:
        # Window is entirely outside the image — empty tile range.
        return (0, 0, -1, -1)
    tx_min = col_off // tile_w
    ty_min = row_off // tile_h
    tx_max = (col_end - 1) // tile_w
    ty_max = (row_end - 1) // tile_h
    return tx_min, ty_min, tx_max, ty_max


async def _fetch_and_decode_tiles(
    ifd: Any, coords: list[tuple[int, int]]
) -> list[np.ndarray]:
    """One batched fetch + per-tile async decode.

    ``ifd.fetch_tiles(xy)`` pipelines all tile range requests over the
    pooled HTTP/2 connection — this is where the wall-clock win lives.
    Decode is per-tile because async-tiff's decoder API takes one tile
    at a time; we ``asyncio.gather`` the decodes so they overlap.

    Each tile lands as ``(H, W, samples)`` from async-tiff; we
    transpose to band-first ``(samples, H, W)`` so the assembly code
    can slice the last two axes uniformly with rasterio's convention.
    """
    if not coords:
        return []
    tiles = await ifd.fetch_tiles(coords)
    decoded = await asyncio.gather(*(t.decode() for t in tiles))
    out: list[np.ndarray] = []
    for d in decoded:
        arr = np.asarray(d)
        if arr.ndim == 3:
            # (H, W, samples) → (samples, H, W).
            arr = np.transpose(arr, (2, 0, 1))
        out.append(arr)
    return out


def _assemble_window(
    window: Window,
    *,
    tile_data: dict[tuple[int, int], np.ndarray],
    tx_min: int,
    ty_min: int,
    tx_max: int,
    ty_max: int,
    tile_w: int,
    tile_h: int,
    bands: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Crop the relevant tiles into a single window-shaped array.

    ``bands`` and ``dtype`` come from the IFD via :func:`_dtype_from_ifd`
    + ``samples_per_pixel``, so the empty-tile-range fallback (window
    entirely outside the image) returns an array of the right shape
    even when no tile was decoded — preserving the documented
    ``(bands, h, w)`` contract regardless of batch composition.
    """
    col_off = int(window.col_off)
    row_off = int(window.row_off)
    w = int(window.width)
    h = int(window.height)

    if tx_max < tx_min or ty_max < ty_min:
        # Empty tile range — window is outside the image. Use the
        # IFD-derived (bands, dtype) so the result is consistent with
        # any in-bounds window in the same select_many call.
        return np.zeros((bands, h, w), dtype=dtype)

    out = np.zeros((bands, h, w), dtype=dtype)

    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            tile = tile_data[(tx, ty)]
            # Tile occupies pixel range [tx*tile_w, (tx+1)*tile_w) x
            # [ty*tile_h, (ty+1)*tile_h). Intersect with the window.
            tile_col_start = tx * tile_w
            tile_row_start = ty * tile_h
            inter_col_start = max(tile_col_start, col_off)
            inter_row_start = max(tile_row_start, row_off)
            inter_col_end = min(tile_col_start + tile_w, col_off + w)
            inter_row_end = min(tile_row_start + tile_h, row_off + h)
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
            out[..., dst_r0:dst_r1, dst_c0:dst_c1] = tile[
                ..., src_r0:src_r1, src_c0:src_c1
            ]
    return out
