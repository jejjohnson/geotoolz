"""Internal path resolution helpers for local paths and cloud URIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


# URI schemes resolved through fsspec instead of GDAL VSI prefixes.
_FSSPEC_SCHEMES = frozenset(
    {
        "s3",
        "gs",
        "gcs",
        "az",
        "azure",
        "http",
        "https",
        "hf",
    }
)


# Cloud/HTTP schemes GDAL reads natively through a virtual file system,
# with ranged requests (a COG header read fetches kilobytes, not the file).
_GDAL_VSI_PREFIXES = {
    "s3": "/vsis3/",
    "gs": "/vsigs/",
    "gcs": "/vsigs/",
    "az": "/vsiaz/",
    "azure": "/vsiaz/",
}


def _gdal_vsi_path(path: str | Path) -> str | None:
    """GDAL ``/vsi*/`` path for a cloud/HTTP URI, or ``None`` if GDAL can't read it.

    ``s3://bucket/key`` -> ``/vsis3/bucket/key``; ``http(s)://…`` ->
    ``/vsicurl/http(s)://…``. Credentials come from GDAL's usual
    configuration (``AWS_*``, ``GOOGLE_APPLICATION_CREDENTIALS``,
    ``AZURE_STORAGE_*`` environment variables, …).
    """
    uri = str(path)
    scheme = _uri_scheme(uri)
    if scheme in ("http", "https"):
        return f"/vsicurl/{uri}"
    prefix = _GDAL_VSI_PREFIXES.get(scheme)
    if prefix is None:
        return None
    parsed = urlsplit(uri)
    return f"{prefix}{parsed.netloc}{parsed.path}"


def _uri_scheme(path: str | Path) -> str:
    """Return the lower-case URI scheme for ``path`` if it has one."""
    return urlsplit(str(path)).scheme.lower()


def _is_fsspec_uri(path: str | Path) -> bool:
    """Return True for cloud/HTTP URI schemes handled through fsspec."""
    return _uri_scheme(path) in _FSSPEC_SCHEMES


def _uri_name(path: str | Path) -> str:
    """Return the filename component without mangling URI schemes."""
    parsed = urlsplit(str(path))
    if parsed.scheme:
        return Path(parsed.path).name
    return Path(path).name


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
        scheme = _uri_scheme(path)
        raise ImportError(
            f"Reading {scheme!r} URIs requires the [fsspec] extra; install via "
            "`pip install 'geocatalog[fsspec]'`."
        ) from exc
    uri = str(path)
    if urlsplit(uri).path.endswith(".zarr"):
        return fsspec.get_mapper(uri, **(storage_options or {}))
    return fsspec.open(uri, mode="rb", **(storage_options or {})).open()


def _close_resolved_uri(resolved: Any) -> None:
    """Close a resolved fsspec handle; local str/Path inputs are no-ops."""
    close = getattr(resolved, "close", None)
    if callable(close):
        close()
