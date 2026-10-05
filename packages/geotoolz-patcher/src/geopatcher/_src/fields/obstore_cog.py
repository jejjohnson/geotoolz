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
  ``shape``, ``bounds``, ``res`` and ``nodata`` (mirrors the GeoData
  surface ``RasterField`` relies on).
- ``select(window)`` — single-window read; collects, fetches, and
  decodes the relevant tiles into a ``GeoTensor`` carrying the
  window's transform, the CRS and the nodata fill value.
- ``select_many(windows)`` — bulk read over a list of windows; the
  batched fast path (one ``GeoTensor`` per window).
- ``with_data(array)`` — reconstruct a ``GeoTensor`` from operator
  output (delegates to georeader), keeping the nodata fill value.

The field pickles by URL: unpickling re-opens the COG through
:meth:`ObstoreCogField.from_url` with the same ``ifd_index``,
``storage_options``, ``timeout`` (and ``store`` / ``path`` when one
was supplied), so it can be shipped to process-pool workers.

Georeferencing follows GDAL / rasterio:

- the raster tiepoint ``(I, J)`` is honoured (``x0 = X - I·sx``,
  ``y0 = Y + J·sy``), as is a ``ModelTransformationTag`` (rotated or
  sheared grids);
- ``GTRasterTypeGeoKey = 2`` (*PixelIsPoint*) shifts the origin by half
  a pixel up-left, matching GDAL's default (``GTIFF_POINT_GEO_IGNORE``
  unset);
- overview IFDs (``ifd_index >= 1``) take the CRS and origin from
  IFD 0 with the pixel size scaled by the full-res / overview size
  ratio per axis;
- ``GDAL_NODATA`` becomes ``domain.nodata`` / ``fill_value_default``,
  and out-of-image parts of a window are filled with it (``0`` when
  the COG declares no nodata).

Windows are snapped outward to whole pixels (offsets floored, far
edges ceiled — georeader's ``round_outer_window`` convention) and the
returned transform describes the snapped window exactly.

Codec fidelity: lossless codecs (deflate, LZW, zstd, none, ...) decode
bit-for-bit identical to GDAL. JPEG tiles are decoded by async-tiff's
own JPEG decoder rather than GDAL's libjpeg, so values can differ from
``rasterio.read`` by a few DN — up to ±3 for ``PHOTOMETRIC=YCbCr`` (the
YCbCr→RGB conversion differs) and ±1 for plain-RGB JPEG in our checks.
Compare JPEG reads with a tolerance.

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
import json
import math
import threading
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor
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
        import async_tiff
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

# GeoTIFF "user-defined" sentinel for CRS geokeys (ProjectedCSTypeGeoKey,
# GeographicTypeGeoKey): the CRS is spelled out by other keys rather than
# named by an EPSG code, so it must not be looked up as ``EPSG:32767``.
_GEOKEY_USER_DEFINED = 32767
# GTRasterTypeGeoKey values: 1 = PixelIsArea (default), 2 = PixelIsPoint.
_RASTER_PIXEL_IS_POINT = 2
# Decimal places used to absorb float noise before snapping window edges
# to whole pixels (same as georeader's ``PIXEL_PRECISION``).
_PIXEL_PRECISION = 3


@dataclass(frozen=True)
class ObstoreCogDomain:
    """RasterDomain-shaped view over a remote COG IFD.

    Exposes the attributes the geopatcher samplers and geometries
    expect from a raster domain (``crs``, ``transform``, ``shape``,
    ``bounds``, ``res``) without holding the IFD object — the domain
    is the I/O-free metadata twin, by Protocol contract.

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
            f"ObstoreCogField: unparseable GDAL_NODATA {text!r} in {url!r}; "
            "ignoring it.",
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
                f"ObstoreCogField: GDAL_NODATA {text!r} in {url!r} is not "
                f"representable as {dtype}; ignoring it.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        return int(value)
    return value


def _geo_transform(geo_ifd: Any) -> Any:
    """Full-resolution affine transform from an IFD's GeoTIFF tags.

    Mirrors GDAL's ``GTiffDataset`` logic: a ``ModelTransformationTag``
    wins when present; otherwise the first ``ModelTiepointTag`` +
    ``ModelPixelScaleTag`` pair, honouring the raster-space tiepoint
    ``(I, J)``. ``GTRasterTypeGeoKey = PixelIsPoint`` shifts the origin
    by half a pixel up-left (GDAL default).

    Raises:
        ValueError: Neither a ModelTransformationTag nor a
            ModelTiepointTag + ModelPixelScaleTag pair is present.
    """
    from rasterio.transform import Affine

    matrix = getattr(geo_ifd, "model_transformation", None)
    tiepoint = getattr(geo_ifd, "model_tiepoint", None)
    pixel_scale = getattr(geo_ifd, "model_pixel_scale", None)
    if matrix is not None and len(matrix) >= 8:
        # Row-major 4x4: [a b 0 c; d e 0 f; 0 0 0 0; 0 0 0 1].
        m = [float(v) for v in matrix]
        transform = Affine(m[0], m[1], m[3], m[4], m[5], m[7])
    elif tiepoint is not None and pixel_scale is not None:
        i, j, _k, x, y, _z = (float(v) for v in list(tiepoint)[:6])
        sx, sy = (float(v) for v in list(pixel_scale)[:2])
        transform = Affine(sx, 0.0, x - i * sx, 0.0, -sy, y + j * sy)
    else:
        raise ValueError(
            "ObstoreCogField: COG IFD lacks ModelTransformationTag and "
            "ModelTiepointTag + ModelPixelScaleTag; cannot derive an affine "
            "transform. Use RasterField (rasterio) for GCP-only or "
            "non-georeferenced TIFFs."
        )

    raster_type = getattr(
        getattr(geo_ifd, "geo_key_directory", None), "raster_type", None
    )
    if raster_type is not None and int(raster_type) == _RASTER_PIXEL_IS_POINT:
        transform = transform * Affine.translation(-0.5, -0.5)
    return transform


def _build_domain(
    ifd: Any, *, geo_ifd: Any = None, url: str = "<unknown>"
) -> ObstoreCogDomain:
    """Read transform, CRS and nodata for ``ifd``.

    Args:
        ifd: The IFD being read (full resolution or an overview).
        geo_ifd: The IFD that carries the GeoTIFF tags — IFD 0 when
            ``ifd`` is an overview (GDAL writes geo tags only on the
            first IFD). Defaults to ``ifd`` itself.
        url: The COG's URL, for error / warning messages.
    """
    from rasterio.transform import Affine

    geo = ifd if geo_ifd is None else geo_ifd
    width = int(ifd.image_width)
    height = int(ifd.image_height)
    samples = int(ifd.samples_per_pixel)

    crs = _crs_from_geokeys(getattr(geo, "geo_key_directory", None))
    transform = _geo_transform(geo)
    if geo is not ifd:
        # Overview: same footprint, coarser pixels (GDAL scales each axis
        # by its own full-res / overview size ratio).
        transform = transform * Affine.scale(
            int(geo.image_width) / width, int(geo.image_height) / height
        )

    dtype = _dtype_from_ifd(ifd, url=url)
    raw_nodata = getattr(ifd, "gdal_nodata", None)
    if raw_nodata is None and geo is not ifd:
        raw_nodata = getattr(geo, "gdal_nodata", None)
    nodata = _parse_nodata(raw_nodata, dtype, url=url)

    corners = [transform * (c, r) for c in (0, width) for r in (0, height)]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    a, b, _c, d, e, _f = transform[:6]
    if b == 0 and d == 0:
        res = (abs(a), abs(e))
    else:
        res = (math.hypot(a, d), math.hypot(b, e))

    return ObstoreCogDomain(
        crs=crs,
        transform=transform,
        shape=(samples, height, width),
        bounds=(min(xs), min(ys), max(xs), max(ys)),
        res=res,
        nodata=nodata,
    )


def _crs_from_geokeys(geo_keys: Any) -> Any:
    """Best-effort CRS extraction from an async-tiff GeoKeyDirectory.

    Handles the common cases: an EPSG ProjectedCSTypeGeoKey
    (``projected_type``) or GeographicTypeGeoKey (``geographic_type``).
    Falls back to ``None`` for exotic GeoTIFFs — with a
    ``RuntimeWarning`` when an EPSG code was present but unusable, or
    when the CRS is user-defined (code ``32767``: spelled out by
    individual geokeys, which this reader does not reassemble) — so
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
        code = int(epsg)
    except (TypeError, ValueError):
        code = None
    if code == _GEOKEY_USER_DEFINED:
        warnings.warn(
            "ObstoreCogField: the COG declares a user-defined CRS "
            f"(geokey {_GEOKEY_USER_DEFINED}); building it from the individual "
            "geokeys is not supported, so domain.crs will be None. Use "
            "RasterField (rasterio) to read it with its full CRS.",
            RuntimeWarning,
            stacklevel=2,
        )
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


def _reopen(url: str, options: dict[str, Any]) -> ObstoreCogField:
    """Unpickle hook: re-open the COG by URL (see `ObstoreCogField.__reduce__`)."""
    return ObstoreCogField.from_url(url, **options)


@dataclass(eq=False)
class ObstoreCogField:
    """Tiled-COG `Field` with batched range-fetch reads.

    Open via :meth:`from_url`; constructor takes the parsed handles.
    ``select`` / ``select_many`` return `georeader.GeoTensor` chips with
    the window's transform, the CRS and the COG's nodata as
    ``fill_value_default``; out-of-image pixels are filled with it.
    The field pickles by URL (see ``__reduce__``).

    Note:
        Lossless codecs match ``rasterio.read`` exactly. JPEG tiles are
        decoded by async-tiff, not GDAL's libjpeg: expect differences of
        a few DN (up to ±3 for ``PHOTOMETRIC=YCbCr``, ±1 for RGB JPEG).

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
        ifd_index: Which IFD ``ifd`` is (``0`` = full resolution).
            Recorded so pickling re-opens the same overview.
        storage_options: Options the pooled store was built with;
            recorded for pickling.
        store: The explicit obstore store passed to :meth:`from_url`,
            if any (``None`` = the process-global pool). Recorded for
            pickling; obstore stores pickle by configuration.
        path: Object key inside ``store`` (only with an explicit store).
    """

    url: str
    tiff: Any  # async_tiff.TIFF
    ifd: Any  # async_tiff.ImageFileDirectory
    domain: ObstoreCogDomain
    timeout: float | None = 120.0
    ifd_index: int = 0
    storage_options: dict[str, Any] | None = None
    store: Any = None
    path: str | None = None

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
                image, ``1+`` for overviews. Overviews inherit the CRS
                and origin of IFD 0 with the pixel size scaled by the
                size ratio. v1 doesn't auto-select by resolution; if you
                need overview pyramid selection, open multiple fields
                and dispatch in user code.
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
            ValueError: COG is striped (not tiled), ``ifd_index`` names
                a mask IFD, or the GeoTIFF tags needed to derive an
                affine transform are missing.
            TimeoutError: Opening the COG took longer than ``timeout``
                seconds.
        """
        async_tiff = _require_async_tiff()

        explicit_store = store
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
        subfile_type = getattr(ifd, "new_subfile_type", None)
        if subfile_type is not None and int(subfile_type) & 4:
            raise ValueError(
                f"ObstoreCogField: IFD {ifd_index} of {url!r} is a "
                "transparency mask, not image data."
            )
        geo_ifd = tiff.ifd(0) if ifd_index != 0 else None
        domain = _build_domain(ifd, geo_ifd=geo_ifd, url=url)
        return cls(
            url=url,
            tiff=tiff,
            ifd=ifd,
            domain=domain,
            timeout=timeout,
            ifd_index=ifd_index,
            storage_options=storage_options,
            store=explicit_store,
            path=path if explicit_store is not None else None,
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Pickle by URL: the parsed TIFF handle is a Rust object.

        Unpickling calls :meth:`from_url` with the recorded
        ``ifd_index`` / ``storage_options`` / ``timeout`` (and the
        explicit ``store`` + ``path`` when one was supplied), so a worker
        process re-reads only the COG header.
        """
        options: dict[str, Any] = {
            "ifd_index": self.ifd_index,
            "storage_options": self.storage_options,
            "timeout": self.timeout,
        }
        if self.store is not None:
            options["store"] = self.store
            options["path"] = self.path
        return (_reopen, (self.url, options))

    def cache_id(self) -> str:
        """`PatchCache` identity: the object read, its version and the IFD.

        With the process pool the object is the ``url``; with an explicit
        ``store`` the ``url`` is only a label, so the identity is the
        store's printable configuration (e.g. ``LocalStore("/data")``)
        plus ``path``. A store without one (``MemoryStore``, whose
        contents live only in that instance) has no stable identity:
        `PatchCache` then requires an explicit ``field_id``.

        The version is one ``HEAD`` of the object (one request per
        `split`, not per patch): its ETag, else its size and
        last-modified time, so an object overwritten in place
        invalidates its cache entries. When the store refuses the
        ``HEAD``, or answers with neither an ETag nor a last-modified
        time, a `RuntimeWarning` says remote changes will not be
        detected and the version is left out.

        Raises:
            UnstableIdentityError: The explicit ``store`` has no printable
                configuration; ``partial`` carries the rest of the
                identity (``path``, ``ifd_index``).

        Examples:
            >>> ObstoreCogField.from_url("s3://b/k.tif").cache_id()
            '{"ifd_index": 0, ..., "url": "s3://b/k.tif", "version": "etag:..."}'
            >>> ObstoreCogField.from_url(
            ...     "file:///d/k.tif", store=LocalStore("/d"), path="k.tif", ifd_index=1
            ... ).cache_id()
            '{"ifd_index": 1, "path": "k.tif", "store": "LocalStore(...)", ...}'
        """
        explicit = self.store is not None
        store_id = _store_identity(self.store) if explicit else None
        identity = {
            "url": None if explicit else self.url,
            # Pooled stores are keyed on their options (endpoint, region,
            # …): the same url can name different objects under different
            # options. A digest, so credentials never reach the key.
            "options": None if explicit else _options_digest(self.storage_options),
            "store": store_id,
            "path": self.path if explicit else None,
            "ifd_index": int(self.ifd_index),
            "version": self._object_version(),
        }
        if explicit and store_id is None:
            from geopatcher._src.cache import UnstableIdentityError

            raise UnstableIdentityError(
                f"its {type(self.store).__qualname__} has no configuration that "
                f"names its contents.",
                partial=json.dumps(identity, sort_keys=True),
            )
        return json.dumps(identity, sort_keys=True)

    def _object_version(self) -> str | None:
        """ETag (else ``size:last_modified``) of the object, or ``None``."""
        if self.store is not None:
            store, key = self.store, self.path
        else:
            from geopatcher._src.objstore import get_obstore

            store = get_obstore(self.url, storage_options=self.storage_options)
            key = _uri_path(self.url)
        try:
            meta = store.head(key)
        except Exception as exc:
            warnings.warn(
                f"ObstoreCogField: HEAD of {self.url!r} failed ({exc}); PatchCache "
                f"entries for it will not notice the object being overwritten.",
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
                f"ObstoreCogField: HEAD of {self.url!r} returned neither an ETag "
                f"nor a Last-Modified time; PatchCache entries for it will not "
                f"notice the object being overwritten.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        return f"size:{meta.get('size')}:{last_modified}"

    @property
    def fill_value_default(self) -> float | int:
        """Fill for out-of-image pixels and padding: nodata, else ``0``."""
        return self.domain.fill_value_default

    def select(self, window: Window) -> GeoTensor:
        """Read one window via the COG's tile grid.

        Implemented as a thin wrapper around ``select_many([window])`` —
        keeps the single-window path going through the same
        tile-coalescing code as the batched path, so there's no
        divergence in semantics.
        """
        return self.select_many([window])[0]

    def select_many(self, windows: list[Window]) -> list[GeoTensor]:
        """Bulk-read every window via one batched tile fetch.

        The headline path: collect every unique tile coordinate
        across all windows, dispatch a single ``ifd.fetch_tiles``
        call, then assemble per-window arrays by cropping each
        decoded tile to its window's intersection.

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

    def with_data(self, array: np.ndarray) -> GeoTensor:
        """Wrap an operator output as a `georeader.GeoTensor`.

        Carries the domain transform, CRS and nodata fill value.
        """
        from georeader.geotensor import GeoTensor

        return GeoTensor(
            values=array,
            transform=self.domain.transform,
            crs=self.domain.crs,
            fill_value_default=self.fill_value_default,
        )


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
            f"ObstoreCogField: window {window!r} has a negative width or height."
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
    ifd: Any, coords: list[tuple[int, int]]
) -> list[np.ndarray]:
    """One batched fetch + per-tile async decode.

    ``ifd.fetch_tiles(xy)`` pipelines all tile range requests over the
    pooled HTTP/2 connection — this is where the wall-clock win lives.
    Decode is per-tile because async-tiff's decoder API takes one tile
    at a time; we ``asyncio.gather`` the decodes so they overlap.

    async-tiff returns pixel-interleaved tiles (``PlanarConfiguration``
    1) as ``(H, W, samples)`` and band-interleaved ones (2) already
    band-first as ``(samples, H, W)``; only the former is transposed,
    so every tile lands band-first for the assembly step.

    Raises:
        ValueError: A decoded tile's band count disagrees with the
            IFD's ``SamplesPerPixel``.
    """
    if not coords:
        return []
    planar = int(getattr(ifd, "planar_configuration", None) or 1)
    samples = int(getattr(ifd, "samples_per_pixel", 1) or 1)
    tiles = await ifd.fetch_tiles(coords)
    decoded = await asyncio.gather(*(t.decode() for t in tiles))
    out: list[np.ndarray] = []
    for d in decoded:
        arr = np.asarray(d)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.ndim == 3 and planar == 1:
            # Chunky: (H, W, samples) → (samples, H, W).
            arr = np.transpose(arr, (2, 0, 1))
        if arr.ndim != 3 or arr.shape[0] != samples:
            raise ValueError(
                f"ObstoreCogField: decoded tile has shape {arr.shape}, expected "
                f"{samples} band(s) (PlanarConfiguration={planar})."
            )
        out.append(arr)
    return out


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

    ``bands`` and ``dtype`` come from the IFD via :func:`_dtype_from_ifd`
    + ``samples_per_pixel``, so the empty-tile-range fallback (window
    entirely outside the image) returns an array of the right shape
    even when no tile was decoded — preserving the documented
    ``(bands, h, w)`` contract regardless of batch composition.

    Copies are clamped to the image extent: edge tiles are stored at
    full tile size, and their padding past the image edge must not leak
    into the window — those pixels keep ``fill`` (the nodata value).
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
            out[..., dst_r0:dst_r1, dst_c0:dst_c1] = tile[
                ..., src_r0:src_r1, src_c0:src_c1
            ]
    return out
