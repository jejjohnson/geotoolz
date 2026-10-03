"""Internal path resolution helpers for local paths and cloud URIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geocatalog._src._extras import missing_extra
from geocatalog._src.uri import FSSPEC_SCHEMES, parse_uri


def _gdal_vsi_path(path: str | Path) -> str | None:
    """GDAL ``/vsi*/`` path for a cloud/HTTP URI, or ``None`` if GDAL can't read it."""
    return parse_uri(path).gdal_path()


def _uri_scheme(path: str | Path) -> str:
    """Return the lower-case URI scheme for ``path`` (``""`` for local paths)."""
    return parse_uri(path).scheme


def _is_fsspec_uri(path: str | Path) -> bool:
    """Return True for cloud/HTTP URI schemes handled through fsspec."""
    return parse_uri(path).scheme in FSSPEC_SCHEMES


def _uri_name(path: str | Path) -> str:
    """Return the filename component without mangling URI schemes."""
    return parse_uri(path).name


def _resolve_uri(
    path: str | Path,
    *,
    storage_options: dict[str, Any] | None = None,
    prefer_gdal: bool = False,
) -> str | Path | Any:
    """Resolve ``path`` to a local path, GDAL VSI path, fsspec mapper or file handle.

    Dispatch:

    * Local paths (``str`` / ``Path``) pass through unchanged.
    * With ``prefer_gdal=True`` (rasterio readers) and no
      ``storage_options``, a cloud/HTTP URI GDAL understands becomes its
      ``/vsi*/`` path (see `_gdal_vsi_path`). An fsspec file object would
      make rasterio copy the *whole* file into memory before reading even
      the header (#220). ``storage_options`` keeps the fsspec route, since
      those credentials don't reach GDAL.
    * Recognised cloud/HTTP URIs ending in ``.zarr`` return an
      ``fsspec.get_mapper(...)`` — Zarr stores are directory-/mapping-based
      and can't be represented by a single binary file handle. The mapper
      is safe to drop on the floor (no resource to close) so
      `_close_resolved_uri` treats it as a no-op.
    * Other recognised cloud/HTTP URIs return an fsspec file-like object;
      callers should pass the returned value to `_close_resolved_uri` when
      finished.
    """
    if not _is_fsspec_uri(path):
        return path
    if prefer_gdal and not storage_options:
        vsi = _gdal_vsi_path(path)
        if vsi is not None:
            return vsi
    try:
        import fsspec
    except ImportError as exc:
        raise missing_extra(f"Reading {_uri_scheme(path)!r} URIs", "fsspec") from exc
    uri = str(path)
    if parse_uri(uri).is_zarr:
        return fsspec.get_mapper(uri, **(storage_options or {}))
    return fsspec.open(uri, mode="rb", **(storage_options or {})).open()


def _close_resolved_uri(resolved: Any) -> None:
    """Close a resolved fsspec handle; local str/Path inputs are no-ops."""
    close = getattr(resolved, "close", None)
    if callable(close):
        close()
