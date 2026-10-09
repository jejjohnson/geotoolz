"""Internal path resolution helpers for local paths and cloud URIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geocatalog._src._extras import missing_extra
from geocatalog._src.utils.uri import FSSPEC_SCHEMES, parse_uri


def _gdal_vsi_path(path: str | Path) -> str | None:
    """GDAL ``/vsi*/`` path for a cloud/HTTP URI, or ``None`` if GDAL can't read it."""
    return parse_uri(path).gdal_path()


def _uri_scheme(path: str | Path) -> str:
    """Return the lower-case URI scheme for ``path`` (``""`` for local paths)."""
    return parse_uri(path).scheme


def _is_fsspec_uri(path: str | Path) -> bool:
    """Return True for cloud/HTTP URI schemes read through `geocloud.fs`."""
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
    """Resolve ``path`` to a local path, GDAL VSI path, Zarr mapper or file handle.

    Dispatch:

    * Local paths (``str`` / ``Path``) pass through unchanged.
    * With ``prefer_gdal=True`` (rasterio readers) and no
      ``storage_options``, a cloud/HTTP URI GDAL understands becomes its
      ``/vsi*/`` path (see `_gdal_vsi_path`). A Python file object would
      make rasterio copy the *whole* file into memory before reading even
      the header (#220). ``storage_options`` keeps the `geocloud.fs`
      route, since those options don't reach GDAL.
    * Other remote URIs are read through `geocloud.fs` (the ``[cloud]``
      extra): geotoolz-cloud's fsspec filesystem on the stack's shared
      obstore pool, with the credentials registered in
      `geocloud.credentials`. ``storage_options`` are obstore store options
      (``{"skip_signature": True}`` for a public bucket), forwarded to
      `geocloud.store.get_obstore`.
    * URIs ending in ``.zarr`` return a mapper (``fs.get_mapper``) — Zarr
      stores are directory-/mapping-based and can't be represented by a
      single binary file handle. The mapper has nothing to close, so
      `_close_resolved_uri` treats it as a no-op.
    * Other remote URIs return a seekable file object; callers pass it to
      `_close_resolved_uri` when finished.
    """
    if not _is_fsspec_uri(path):
        return path
    if prefer_gdal and not storage_options:
        vsi = _gdal_vsi_path(path)
        if vsi is not None:
            return vsi
    try:
        from geocloud.fs import filesystem

        fs = filesystem(storage_options)
    except ImportError as exc:
        raise missing_extra(
            f"Reading {_uri_scheme(path)!r} URIs",
            "cloud",
            packages="'geotoolz-cloud[fsspec]'",
        ) from exc
    uri = str(path)
    if parse_uri(uri).is_zarr:
        return fs.get_mapper(uri)
    return fs.open(uri, mode="rb")


def _close_resolved_uri(resolved: Any) -> None:
    """Close a resolved file handle; paths and mappers are no-ops."""
    close = getattr(resolved, "close", None)
    if callable(close):
        close()
