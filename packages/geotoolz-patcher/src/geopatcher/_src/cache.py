"""`PatchCache` — content-addressed, on-disk patch cache (gh #24).

Cross-run sibling of `IndexedPatchView(cache=True)` (which only avoids
re-reads *within* a process). A `PatchCache` keys each patch by

    sha256( format ‖ field_id ‖ config_id ‖ anchor_id )

so a second *process* — a rerun after editing the operator — skips the
source read entirely and consults the field only for its ``domain``
metadata.

- ``field_id`` — what the field reads (see `PatchCache.field_id_for`):
  the source identity, the adapter's own ``cache_id()``, the domain
  (CRS, transform, shape, dtype / grid coordinates) and the reader's
  band selection (``indexes``) and boundless fill.
- ``config_id`` — ``json.dumps`` of the geometry + window configs
  (sampler / aggregation excluded: they don't change patch bytes).
- ``anchor_id`` — `PatchJournal`'s anchor normaliser, reused verbatim.

Each entry is one ``<hash>.npz`` holding ``values``, the carrier
``kind`` (``"geotensor"``, ``"dataarray"`` or ``"ndarray"``), a JSON
``meta`` record with everything needed to rebuild the carrier exactly
(`GeoTensor` transform / CRS / ``fill_value_default`` / ``attrs``;
`DataArray` dims / coords / attrs / encoding, which carry the rioxarray
transform, CRS and nodata), and ``weights`` when present. A hit is
therefore bit-identical to an uncached read. A carrier that cannot be
rebuilt exactly (a `GeoDataFrame`, an object-dtype array, an attribute
value with no lossless encoding) is refused with a `TypeError` rather
than silently degraded.

`Patch.anchor` / `Patch.indices` are recomputed by the patcher, not
stored. Writes are atomic (unique temp file + ``os.replace``), so a
reader never sees a torn entry written by this class; an entry that
fails to load anyway (zero-byte, truncated, foreign) is a miss, is
deleted, and is rewritten by the next `put`. Eviction is LRU by file
``st_atime`` when ``max_bytes`` is exceeded, checked on write.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from geopatcher._src.journal import _anchor_key
from geopatcher._src.patch import Patch


# Bumped whenever the key derivation or the entry layout changes, so
# entries written by an older layout can never be served.
_FORMAT = "geopatcher.PatchCache/2"


@dataclass
class PatchCache:
    """A local, content-addressed directory cache of read patches.

    Args:
        root: Directory the cache writes entries under (created if
            absent). The layout is fsspec-friendly (two-level shard) but
            v1 is a local filesystem only.
        max_bytes: Soft cap on the cache size. When exceeded after a
            write, least-recently-accessed entries are evicted until the
            total is back under the cap. ``None`` (default) is unbounded.
        field_id: Explicit source identity for in-memory fields that
            have none (a bare `GeoTensor`-backed `RasterField`). Leave
            ``None`` for path- or URL-backed fields, whose identity is
            derived automatically. It replaces only the *source* part of
            the key: the domain, band selection and the adapter's
            ``cache_id()`` are still folded in.
    """

    root: str | Path
    max_bytes: int | None = None
    field_id: str | None = None
    _hits: int = field(default=0, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.max_bytes is not None and self.max_bytes < 1:
            raise ValueError("max_bytes must be >= 1 (or None for unbounded).")

    # -- key derivation ---------------------------------------------------

    def field_id_for(self, field: Any) -> str:
        """Resolve the identity of everything ``field`` reads.

        The result is ``"<source>|<json>"``:

        - ``source`` — this cache's explicit ``field_id``, else the
          first of: a ``url``, a reader / field file path
          (``realpath`` + ``st_mtime_ns`` + ``st_size``), the
          ``encoding["source"]`` file of an xarray-backed field —
          looked up on the field and then on its ``reader`` / ``da`` /
          ``array``. A field exposing only a ``cache_id()`` uses that as
          its source.
        - ``json`` — the field's ``cache_id()`` (per-adapter identity,
          e.g. `ObstoreCogField`'s store / path / ``ifd_index`` or
          `ReprojectingRasterField`'s ``dst_crs`` / ``resolution`` /
          ``resampling``), the domain (CRS, transform, shape, dtype, or
          a digest of the grid coordinates) and the reader's band
          selection (``indexes``) and boundless ``fill_value_default``.

        An xarray-backed field also keys on its variable name (two
        variables of one file share everything else). An adapter whose
        ``cache_id()`` raises `UnstableIdentityError` (an in-memory
        object store) needs an explicit ``field_id``.

        In-memory fields with no source identity raise, since caching
        against nothing would silently serve stale data — including a
        field that wraps an unnamed in-memory reader, whatever its
        ``cache_id()`` says.

        Examples:
            >>> cache = PatchCache("./.cache")
            >>> cache.field_id_for(RasterField(RasterioReader("a.tif")))
            'path:/data/a.tif:1700000000000000000:4096|{"adapter": null, ...}'
            >>> PatchCache("./.cache", field_id="scene").field_id_for(
            ...     RasterField(geotensor)
            ... )
            'scene|{"adapter": null, "domain": {...}, "read": {...}}'
        """
        try:
            adapter_id = _adapter_id(field)
        except UnstableIdentityError as exc:
            if self.field_id is None:
                raise ValueError(
                    f"PatchCache cannot derive a stable identity for a "
                    f"{type(field).__name__}: {exc} Pass field_id=... to "
                    f"PatchCache, distinct for every object it caches."
                ) from exc
            adapter_id = exc.partial
        source = self.field_id if self.field_id is not None else _source_id(field)
        if source is None:
            if adapter_id is None or _wraps_unnamed_source(field):
                raise ValueError(
                    f"PatchCache cannot derive a stable identity for a "
                    f"{type(field).__name__}: it exposes no cache_id(), url, or "
                    f"file path. Pass field_id=... to PatchCache for in-memory "
                    f"fields."
                )
            source = f"id:{adapter_id}"
        signature = {
            "adapter": adapter_id,
            "domain": _domain_signature(field),
            "read": _read_signature(field),
        }
        return f"{source}|{json.dumps(signature, sort_keys=True)}"

    @staticmethod
    def config_id_for(geometry: Any, window: Any) -> str:
        """JSON config key from the geometry + window (byte-affecting axes)."""
        for axis in (geometry, window):
            if getattr(axis, "forbid_in_yaml", False):
                raise ValueError(
                    f"{type(axis).__name__} carries a Python closure and cannot "
                    f"be used as a cache key; give it a serialisable config or "
                    f"drop the cache for this run."
                )
        return json.dumps(
            {"geometry": geometry.get_config(), "window": window.get_config()},
            sort_keys=True,
        )

    def _key(self, field_id: str, config_id: str, anchor: Any) -> str:
        h = hashlib.sha256()
        for part in (_FORMAT, field_id, config_id, _anchor_key(anchor)):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return Path(self.root) / key[:2] / f"{key}.npz"

    # -- get / put --------------------------------------------------------

    def get(self, field_id: str, config_id: str, anchor: Any) -> dict[str, Any] | None:
        """Return the stored payload for a key, or ``None`` on a miss.

        An entry that cannot be decoded — zero-byte, truncated, not a
        zip, or missing the layout's fields — is a miss and is deleted,
        so the following `put` rewrites it.
        """
        path = self._path(self._key(field_id, config_id, anchor))
        if not path.exists():
            self._misses += 1
            return None
        try:
            payload = _load_entry(path)
            # Rebuild the carrier here, inside the repair guard: an entry
            # with an unknown kind or inconsistent metadata is as unusable
            # as a torn zip and must not fail later, on every run.
            payload["decoded"] = _decode_carrier(payload)
        except Exception:
            # Any decode failure (zipfile.BadZipFile, EOFError, KeyError,
            # a JSON error, a carrier that won't rebuild, …) means the
            # entry is unusable: repair it.
            with suppress(OSError):
                path.unlink()
            self._misses += 1
            return None
        self._hits += 1
        if self.max_bytes is not None:
            # Refresh the access time so eviction sees true recency even on
            # relatime/noatime mounts.
            with suppress(OSError):
                os.utime(path)
        return payload

    def put(self, field_id: str, config_id: str, anchor: Any, patch: Patch) -> None:
        """Store ``patch``'s reconstructable payload under its key.

        Always (re)writes the entry — `put` follows a miss, so an
        existing file at the key is one `get` could not use. The write
        is atomic: a unique temp file in the shard directory is
        published with ``os.replace``.

        Raises:
            TypeError: The patch carrier cannot be rebuilt bit-identically
                from disk (see `_encode_patch`).
        """
        path = self._path(self._key(field_id, config_id, anchor))
        arrays = _encode_patch(patch)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A unique temp per writer (mkstemp is atomic) so concurrent writers
        # of the same key — same PID across threads included — never share
        # a scratch file; the os.replace then publishes atomically.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **arrays)
                f.flush()
                with suppress(OSError):
                    os.fsync(f.fileno())
            os.replace(tmp_name, path)
        finally:
            with suppress(OSError):
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        if self.max_bytes is not None:
            self._evict()

    def build_patch(self, payload: dict[str, Any], anchor: Any, indices: Any) -> Patch:
        """Rebuild a `Patch` from a stored ``payload`` at ``anchor``/``indices``."""
        data = payload["decoded"] if "decoded" in payload else _decode_carrier(payload)
        weights = payload.get("weights")
        return Patch(data=data, anchor=anchor, indices=indices, weights=weights)

    # -- introspection ----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return ``{"hits", "misses", "bytes", "entries"}`` for the cache."""
        total = 0
        entries = 0
        for path in self.root.rglob("*.npz"):
            with suppress(OSError):
                total += path.stat().st_size
                entries += 1
        return {
            "hits": self._hits,
            "misses": self._misses,
            "bytes": total,
            "entries": entries,
        }

    def clear(self) -> None:
        """Delete every cached entry; reset hit / miss counters."""
        for path in self.root.rglob("*.npz"):
            with suppress(OSError):
                path.unlink()
        self._hits = 0
        self._misses = 0

    def _evict(self) -> None:
        """Drop least-recently-accessed entries until under ``max_bytes``."""
        if self.max_bytes is None:
            return
        entries: list[tuple[float, int, Path]] = []
        for path in self.root.rglob("*.npz"):
            with suppress(OSError):
                st = path.stat()
                entries.append((st.st_atime, st.st_size, path))
        total = sum(size for _, size, _ in entries)
        if total <= self.max_bytes:
            return
        entries.sort()  # oldest access time first
        for _atime, size, path in entries:
            if total <= self.max_bytes:
                break
            with suppress(OSError):
                path.unlink()
                total -= size


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

# Attributes through which a field adapter exposes the object it reads.
_SOURCE_ATTRS = ("reader", "da", "array")


class UnstableIdentityError(ValueError):
    """A field's ``cache_id()`` cannot name what it reads stably.

    Raised by an adapter whose source has no identity that survives the
    process (an in-memory object store, say). `PatchCache` then requires
    an explicit ``field_id`` and keys on it plus ``partial`` — the rest
    of the adapter's identity (IFD, reprojection parameters, …).
    """

    def __init__(self, message: str, *, partial: str) -> None:
        super().__init__(message)
        self.partial = partial


def _adapter_id(field: Any) -> str | None:
    """The field's own ``cache_id()``, or ``None`` when it defines none."""
    cache_id = getattr(field, "cache_id", None)
    if not callable(cache_id):
        return None
    return str(cache_id())


def _source_id(obj: Any, _depth: int = 0) -> str | None:
    """Stable identity of the data behind ``obj``, or ``None`` if in-memory.

    Checks ``obj`` itself — a ``url``, a file ``paths`` / ``path``, an
    xarray ``encoding["source"]`` — then recurses into the object it
    wraps (``reader`` / ``da`` / ``array``).
    """
    if obj is None or _depth > 4:
        return None
    url = getattr(obj, "url", None)
    if isinstance(url, str) and url:
        return f"url:{url}"
    encoding = getattr(obj, "encoding", None)
    for path in (
        getattr(obj, "paths", None),
        getattr(obj, "path", None),
        encoding.get("source") if isinstance(encoding, dict) else None,
    ):
        if isinstance(path, (list, tuple)):
            path = path[0] if path else None
        if not isinstance(path, (str, os.PathLike)):
            continue
        if _is_remote(path):
            # ``open_rasterio("s3://…")`` / ``open_dataset("https://…")``:
            # no local stat, but the URL still names the object.
            return f"url:{path}"
        with suppress(OSError):
            st = os.stat(path)
            return f"path:{os.path.realpath(path)}:{st.st_mtime_ns}:{st.st_size}"
    for attr in _SOURCE_ATTRS:
        inner = getattr(obj, attr, None)
        if inner is not None and inner is not obj:
            found = _source_id(inner, _depth + 1)
            if found is not None:
                return found
    return None


def _is_remote(path: Any) -> bool:
    """``True`` for a ``scheme://`` URL other than ``file://``."""
    text = os.fspath(path)
    scheme, sep, _ = text.partition("://")
    return bool(sep) and scheme.isidentifier() and scheme.lower() != "file"


def _wraps_unnamed_source(field: Any) -> bool:
    """``True`` when ``field`` wraps a reader / array (that had no identity)."""
    return any(getattr(field, attr, None) is not None for attr in _SOURCE_ATTRS)


def _crs_text(crs: Any) -> str | None:
    """Canonical text of a CRS for keying (WKT when available)."""
    if crs is None:
        return None
    if isinstance(crs, str):
        return crs
    to_wkt = getattr(crs, "to_wkt", None)
    if callable(to_wkt):
        return str(to_wkt())
    return str(crs)


def _domain_signature(field: Any) -> dict[str, Any] | None:
    """CRS, transform, shape, dtype and grid-coordinate digest of the domain."""
    domain = getattr(field, "domain", None)
    if domain is None:
        return None
    sig: dict[str, Any] = {"crs": _crs_text(getattr(domain, "crs", None))}
    transform = getattr(domain, "transform", None)
    if transform is not None and not callable(transform):
        sig["transform"] = [float(v) for v in tuple(transform)[:6]]
    shape = getattr(domain, "shape", None)
    if shape is not None:
        sig["shape"] = [int(d) for d in shape]
    coords = getattr(domain, "coords", None)
    if isinstance(coords, dict):
        sig["coords"] = {str(name): _array_digest(v) for name, v in coords.items()}
    dtype = getattr(domain, "dtype", None)
    if dtype is None:
        for attr in _SOURCE_ATTRS:
            dtype = getattr(getattr(field, attr, None), "dtype", None)
            if dtype is not None:
                break
    if dtype is not None:
        with suppress(TypeError):
            sig["dtype"] = np.dtype(dtype).str
    return sig


def _read_signature(field: Any) -> dict[str, Any]:
    """Reader-level selection: band ``indexes`` and the boundless fill."""
    reader = getattr(field, "reader", None)
    if reader is None:
        reader = getattr(field, "domain", None)
    sig: dict[str, Any] = {}
    indexes = getattr(reader, "indexes", None)
    if indexes is not None:
        sig["indexes"] = np.asarray(indexes).tolist()
    fill = getattr(reader, "fill_value_default", None)
    if fill is not None:
        sig["fill"] = _to_json(fill, what="fill_value_default")
    # Two variables of one file share ``encoding["source"]``, dims, coords,
    # shape and dtype; only the variable name tells them apart.
    for attr in ("da", "array"):
        data = getattr(field, attr, None)
        if data is not None and hasattr(data, "dims") and hasattr(data, "name"):
            sig["variable"] = _to_json(data.name, what="DataArray.name")
            # ``open_rasterio(p).sel(band=1)`` and ``.sel(band=2)`` differ
            # only in a (scalar or non-spatial) coordinate.
            sig["coords"] = {
                str(name): _array_digest(coord.values)
                for name, coord in data.coords.items()
            }
            # Two views of one file can differ only in metadata the chips
            # carry: ``rio.write_nodata`` (which also pads out-of-range
            # windows) or other attrs / encoding.
            rio = getattr(data, "rio", None)
            if rio is not None:
                with suppress(Exception):
                    sig["nodata"] = _to_json(rio.nodata, what="rio.nodata")
            sig["attrs"] = _metadata_digest(data.attrs, data.encoding)
            break
    return sig


def _metadata_digest(*parts: Any) -> str:
    """sha256 over ``repr`` of mappings, keys sorted (any value type)."""
    h = hashlib.sha256()
    for part in parts:
        for key in sorted(part, key=repr):
            h.update(repr(key).encode("utf-8"))
            value = part[key]
            if isinstance(value, np.ndarray):
                h.update(_array_digest(value).encode())
            else:
                h.update(repr(value).encode("utf-8"))
    return h.hexdigest()


def _array_digest(values: Any) -> str:
    """sha256 of an array's dtype, shape and bytes (object dtype via ``repr``)."""
    arr = np.asarray(values)
    h = hashlib.sha256(f"{arr.dtype.str}{arr.shape}".encode())
    if arr.dtype.hasobject:
        h.update(repr(arr.tolist()).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Entry encoding
# ---------------------------------------------------------------------------


def _load_entry(path: Path) -> dict[str, Any]:
    """Read and validate one entry; raises on anything unusable."""
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    if str(arrays["format"]) != _FORMAT:
        raise ValueError(f"foreign cache entry format {arrays['format']!r}")
    payload: dict[str, Any] = {
        "kind": str(arrays["kind"]),
        "values": arrays["values"],
        "meta": json.loads(str(arrays["meta"])),
    }
    if "weights" in arrays:
        payload["weights"] = arrays["weights"]
    for i in range(len(payload["meta"].get("coords", []))):
        payload[f"coord_{i}"] = arrays[f"coord_{i}"]
    return payload


def _encode_patch(patch: Patch) -> dict[str, Any]:
    """Arrays + JSON metadata that rebuild ``patch.data`` exactly.

    Raises:
        TypeError: The carrier is not a `GeoTensor`, an xarray
            `DataArray` or a plain ndarray, or holds a value (object
            dtype, an attribute with no lossless JSON form) that cannot
            round-trip.
    """
    from georeader.geotensor import GeoTensor

    data = patch.data
    arrays: dict[str, Any] = {"format": np.array(_FORMAT)}
    if isinstance(data, GeoTensor):
        kind = "geotensor"
        values = np.asarray(data)
        meta: dict[str, Any] = {
            "transform": [float(v) for v in tuple(data.transform)[:6]],
            "crs": _encode_crs(data.crs),
            "fill_value_default": _to_json(
                data.fill_value_default, what="GeoTensor.fill_value_default"
            ),
            "attrs": _to_json(data.attrs, what="GeoTensor.attrs"),
        }
    elif _is_dataarray(data):
        kind = "dataarray"
        values = np.asarray(data.values)
        meta = _dataarray_meta(data, arrays)
    elif type(data) is np.ndarray:
        kind = "ndarray"
        values = data
        meta = {}
    else:
        raise TypeError(
            f"PatchCache cannot store a {type(data).__module__}."
            f"{type(data).__qualname__} patch: only GeoTensor, xarray.DataArray "
            f"and numpy.ndarray carriers round-trip bit-identically. Drop "
            f"cache= for this field."
        )
    _check_storable(values, what=f"{kind} values")
    arrays["kind"] = np.array(kind)
    arrays["values"] = values
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    if patch.weights is not None:
        weights = np.asarray(patch.weights)
        _check_storable(weights, what="patch weights")
        arrays["weights"] = weights
    return arrays


def _decode_carrier(payload: dict[str, Any]) -> Any:
    """Inverse of `_encode_patch` for the carrier (``patch.data``)."""
    kind = payload["kind"]
    meta = payload["meta"]
    values = payload["values"]
    if kind == "geotensor":
        from georeader.geotensor import GeoTensor
        from rasterio import Affine

        return GeoTensor(
            values=values,
            transform=Affine(*meta["transform"]),
            crs=_decode_crs(meta["crs"]),
            fill_value_default=_from_json(meta["fill_value_default"]),
            attrs=_from_json(meta["attrs"]),
        )
    if kind == "dataarray":
        return _dataarray_from(payload)
    if kind == "ndarray":
        return values
    raise ValueError(f"unknown cache entry kind {kind!r}")


def _is_dataarray(data: Any) -> bool:
    try:
        import xarray as xr  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - xarray absent → no DataArray
        return False
    return isinstance(data, xr.DataArray)


def _dataarray_meta(da: Any, arrays: dict[str, Any]) -> dict[str, Any]:
    """Dims, coords, attrs and encoding of ``da``; coord values go in ``arrays``.

    The rioxarray state (``rio.transform()`` / ``rio.crs`` /
    ``rio.nodata``) lives in the ``spatial_ref`` coordinate's attrs, the
    spatial coordinates and ``attrs`` / ``encoding["_FillValue"]``, so it
    round-trips with them.
    """
    coords = []
    for i, (name, coord) in enumerate(da.coords.items()):
        index = da.xindexes.get(name)
        if index is not None and type(index).__name__ != "PandasIndex":
            raise TypeError(
                f"PatchCache cannot store a DataArray with a "
                f"{type(index).__name__} index on {name!r}; only default "
                f"pandas indexes round-trip."
            )
        values = np.asarray(coord.values)
        _check_storable(values, what=f"DataArray coordinate {name!r}")
        arrays[f"coord_{i}"] = values
        coords.append(
            {
                "name": _to_json(name, what="DataArray coordinate name"),
                "dims": _to_json(list(coord.dims), what="dimension names"),
                "attrs": _to_json(coord.attrs, what=f"attrs of coordinate {name!r}"),
                "encoding": _to_json(
                    coord.encoding, what=f"encoding of coordinate {name!r}"
                ),
            }
        )
    return {
        "name": _to_json(da.name, what="DataArray.name"),
        "dims": _to_json(list(da.dims), what="dimension names"),
        "coords": coords,
        "attrs": _to_json(da.attrs, what="DataArray.attrs"),
        "encoding": _to_json(da.encoding, what="DataArray.encoding"),
    }


def _dataarray_from(payload: dict[str, Any]) -> Any:
    import xarray as xr  # type: ignore[import-untyped]

    meta = payload["meta"]
    coords = {}
    for i, c in enumerate(meta["coords"]):
        coords[_from_json(c["name"])] = xr.Variable(
            _from_json(c["dims"]),
            payload[f"coord_{i}"],
            attrs=_from_json(c["attrs"]),
            encoding=_from_json(c["encoding"]),
        )
    da = xr.DataArray(
        payload["values"],
        coords=coords,
        dims=_from_json(meta["dims"]),
        name=_from_json(meta["name"]),
        attrs=_from_json(meta["attrs"]),
    )
    da.encoding = _from_json(meta["encoding"])
    return da


def _check_storable(values: np.ndarray, *, what: str) -> None:
    if values.dtype.hasobject:
        raise TypeError(
            f"PatchCache cannot store {what} with object dtype: it would "
            f"need pickling and could not be rebuilt bit-identically."
        )
    _check_plain_dtype(values.dtype, what=what)


def _check_plain_dtype(dtype: np.dtype, *, what: str) -> None:
    """Refuse structured / void dtypes: ``dtype.str`` (``"|V8"``) drops fields."""
    if dtype.kind == "V":
        raise TypeError(
            f"PatchCache cannot store {what} with structured or void dtype "
            f"{dtype!r}: its field layout would not survive the round trip."
        )


def _encode_crs(crs: Any) -> Any:
    if crs is None:
        return None
    if isinstance(crs, str):
        return {"str": crs}
    from rasterio.crs import CRS

    if isinstance(crs, CRS):
        return {"rasterio": crs.to_wkt()}
    try:
        import pyproj
    except ImportError:  # pragma: no cover - pyproj ships with rasterio stacks
        pyproj = None  # type: ignore[assignment]
    if pyproj is not None and isinstance(crs, pyproj.CRS):
        return {"pyproj": crs.to_wkt()}
    raise TypeError(
        f"PatchCache cannot store a CRS of type {type(crs).__qualname__}; "
        f"use a str, rasterio.crs.CRS or pyproj.CRS."
    )


def _decode_crs(encoded: Any) -> Any:
    if encoded is None:
        return None
    if "str" in encoded:
        return encoded["str"]
    if "rasterio" in encoded:
        from rasterio.crs import CRS

        return CRS.from_wkt(encoded["rasterio"])
    import pyproj

    return pyproj.CRS.from_wkt(encoded["pyproj"])


def _to_json(value: Any, *, what: str) -> Any:
    """Lossless JSON form of ``value`` (tagged for numpy / tuple / dict).

    Plain ``bool`` / ``int`` / ``float`` / ``str`` / ``None`` / ``list``
    pass through; numpy scalars and arrays keep dtype and bytes (base64);
    tuples and dicts are tagged so they come back as tuples and dicts
    with their key order.

    Raises:
        TypeError: ``value`` (or something inside it) has no lossless
            encoding — e.g. a ``datetime``, a non-string dict key or an
            object-dtype array.
    """
    if type(value) in (bool, int, float, str, type(None)):
        return value
    if isinstance(value, np.dtype):
        _check_plain_dtype(value, what=what)
        return {"__dtype__": value.str}
    if isinstance(value, (np.ndarray, np.generic)):
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            raise TypeError(f"PatchCache cannot store {what}: object-dtype array.")
        _check_plain_dtype(arr.dtype, what=what)
        return {
            "__np__": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode(),
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
            "scalar": isinstance(value, np.generic),
        }
    if isinstance(value, list):
        return [_to_json(v, what=what) for v in value]
    if isinstance(value, tuple):
        return {"__tuple__": [_to_json(v, what=what) for v in value]}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError(f"PatchCache cannot store {what}: non-string dict keys.")
        return {"__dict__": [[k, _to_json(v, what=what)] for k, v in value.items()]}
    raise TypeError(
        f"PatchCache cannot store {what}: a value of type "
        f"{type(value).__qualname__} has no lossless on-disk form. Drop "
        f"cache= for this field or convert the value to a numpy / plain "
        f"Python type."
    )


def _from_json(value: Any) -> Any:
    """Inverse of `_to_json`."""
    if isinstance(value, list):
        return [_from_json(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "__dtype__" in value:
        return np.dtype(value["__dtype__"])
    if "__np__" in value:
        arr = np.frombuffer(
            base64.b64decode(value["__np__"]), dtype=np.dtype(value["dtype"])
        ).reshape(value["shape"])
        return arr[()] if value["scalar"] else arr.copy()
    if "__tuple__" in value:
        return tuple(_from_json(v) for v in value["__tuple__"])
    return {k: _from_json(v) for k, v in value["__dict__"]}


__all__ = ["PatchCache"]
