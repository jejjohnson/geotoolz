"""Process-global ``obstore`` client pool for geopatcher's COG field.

Mirrors the pool in ``geocatalog._src.objstore``; when geocatalog is
installed alongside geopatcher we soft-import the shared pool so a
process talking to the same bucket through both packages reuses one
client (and one HTTP/2 connection pool). When geocatalog isn't
available we fall back to a local pool with the same surface.

Why two copies: geopatcher is published independently and shouldn't
hard-depend on geocatalog. The 80 LOC of duplication is cheaper than
either a new shared PyPI package or a tight cross-repo dep.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit


if TYPE_CHECKING:
    from obstore.store import ObjectStore


_OBSTORE_INSTALL_HINT = (
    "The ObstoreCogField pool requires the [obstore-cog] extra; install via "
    "`pip install 'geopatcher[obstore-cog]'`."
)


def _try_geocatalog_pool() -> Any:
    """Return ``geocatalog._src.objstore.get_obstore`` if available, else ``None``.

    When both packages are co-installed this lets them share a single
    pool — opens of the same bucket from either side hit the same
    HTTP/2 connection. Soft-imported so geopatcher remains
    installable without geocatalog.
    """
    try:
        from geocatalog._src.objstore import (
            get_obstore,  # ty: ignore[unresolved-import]
        )
    except ImportError:
        return None
    return get_obstore


_POOL: OrderedDict[tuple[str, str, str | None, str | None], Any] = OrderedDict()
_POOL_MAXSIZE = 64
_POOL_LOCK = threading.Lock()


def _pool_key(uri: str) -> tuple[str, str, str | None, str | None]:
    """Return ``(scheme, bucket, region, endpoint)`` for ``uri``.

    Mirrors :func:`geocatalog._src.objstore._pool_key`; kept in sync so
    the two pools cache by the same key when both packages are
    installed.
    """
    parsed = urlsplit(uri)
    scheme = parsed.scheme.lower()
    bucket = parsed.netloc.split("@", 1)[-1]
    region: str | None
    endpoint: str | None
    if scheme in ("s3", "s3a"):
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        endpoint = os.environ.get("AWS_S3_ENDPOINT") or os.environ.get(
            "AWS_ENDPOINT_URL"
        )
    elif scheme in ("gs", "gcs"):
        region = None
        endpoint = os.environ.get("GOOGLE_SERVICE_ENDPOINT")
    elif scheme in ("az", "azure", "abfs"):
        region = None
        endpoint = os.environ.get("AZURE_STORAGE_ENDPOINT")
    else:
        region = None
        endpoint = None
    return scheme, bucket, region, endpoint


def _build_store(uri: str, storage_options: dict[str, Any] | None) -> ObjectStore:
    try:
        from obstore.store import (  # ty: ignore[unresolved-import]
            AzureStore,
            GCSStore,
            HTTPStore,
            S3Store,
        )
    except ImportError as exc:
        raise ImportError(_OBSTORE_INSTALL_HINT) from exc

    parsed = urlsplit(uri)
    scheme = parsed.scheme.lower()
    bucket = parsed.netloc.split("@", 1)[-1]
    options = dict(storage_options or {})

    if scheme in ("s3", "s3a"):
        return S3Store(bucket, **options)
    if scheme in ("gs", "gcs"):
        return GCSStore(bucket, **options)
    if scheme in ("az", "azure", "abfs"):
        # Azure URIs are ``az://account/container/blob`` — the container
        # name is the FIRST path segment, not the netloc, and obstore's
        # ``AzureStore(container, ...)`` constructor expects the
        # container name. Use ``from_url`` so obstore's parser handles
        # the account/container/blob split correctly; per-request reads
        # then use the blob key (the rest of the path) — see
        # `object_key` below.
        return AzureStore.from_url(uri, **options)
    if scheme in ("http", "https"):
        origin = f"{scheme}://{parsed.netloc}"
        return HTTPStore.from_url(origin, **options)
    raise ValueError(
        f"obstore client pool: unsupported scheme {scheme!r} for URI {uri!r}. "
        "Supported: s3, gs, gcs, az, azure, abfs, http, https."
    )


def object_key(uri: str) -> str:
    """Return the key inside the pooled store for ``uri``.

    Most schemes: the path component minus the leading ``/``. Azure
    is the exception — the first path segment is the container name
    (already baked into the pooled `AzureStore` instance) and the
    rest is the blob key.
    """
    parsed = urlsplit(uri)
    scheme = parsed.scheme.lower()
    path = parsed.path.lstrip("/")
    if scheme in ("az", "azure", "abfs"):
        # Drop the container segment; what remains is the blob key.
        _, _, blob = path.partition("/")
        return blob
    return path


def get_obstore(
    uri: str,
    *,
    storage_options: dict[str, Any] | None = None,
) -> ObjectStore:
    """Return the pooled :class:`ObjectStore` for ``uri``.

    When ``geocatalog`` is installed the call is forwarded to its pool
    so both packages share clients. Otherwise the local pool is used
    with identical LRU + fork-safety semantics.
    """
    geocatalog_pool = _try_geocatalog_pool()
    if geocatalog_pool is not None:
        return geocatalog_pool(uri, storage_options=storage_options)

    key = _pool_key(uri)
    with _POOL_LOCK:
        existing = _POOL.get(key)
        if existing is not None:
            _POOL.move_to_end(key)
            return existing
        store = _build_store(uri, storage_options)
        _POOL[key] = store
        while len(_POOL) > _POOL_MAXSIZE:
            _POOL.popitem(last=False)
        return store


def clear_obstore_pool() -> None:
    """Drop every pooled client. No-op when geocatalog owns the pool."""
    with _POOL_LOCK:
        _POOL.clear()


def _clear_after_fork() -> None:
    clear_obstore_pool()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_clear_after_fork)
