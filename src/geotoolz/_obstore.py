"""Process-global ``obstore`` client pool for geotoolz sensor readers.

Mirrors the pools in ``geocatalog._src.objstore`` and
``geopatcher._src.objstore``: a ``dict[(scheme, bucket, region,
endpoint) -> ObjectStore]`` held once per process so HTTP/2
connection pooling and TLS-session reuse survives across every cloud
read in one Python session. The whole point of having a sibling pool
here — rather than just importing the geocatalog one — is that
geotoolz must remain installable without either of those packages;
when they *are* installed, the soft-import below funnels through the
shared cache so all three libraries share one set of clients.

Surfaces:

- :func:`get_obstore` — return a pooled ``ObjectStore`` for a URI.
- :func:`read_byte_range` — async helper that resolves a URI to its
  pool entry and fetches a byte range. The cloud-backed `SensorReader`
  subclasses use this for header / overview / tile reads when an
  obstore client is wired in.
- :func:`clear_obstore_pool` — explicit pool teardown for long-lived
  processes (notebooks).

Gated on the ``[obstore]`` extra. Importing this module without
``obstore`` installed succeeds; any call raises ``ImportError`` with
the install hint.

Adapted from the ``openEO-RuSTAC`` Rust pattern in
``crates/orbit-geo/src/async_download.rs:76-140``.
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
    "The geotoolz obstore client pool requires the [obstore] extra; "
    "install via `pip install 'geotoolz[obstore]'`."
)


def _try_shared_pool() -> Any:
    """Return a foreign pool's ``get_obstore`` if available, else ``None``.

    Preference order: geocatalog → geopatcher → local. When either
    sibling package is installed we forward through their pool so the
    three libraries share one process-global client cache. Otherwise
    we fall through to the local copy below.
    """
    try:
        from geocatalog._src.objstore import (
            get_obstore,  # ty: ignore[unresolved-import]
        )

        return get_obstore
    except ImportError:
        pass
    try:
        from geopatcher._src.objstore import (
            get_obstore,  # ty: ignore[unresolved-import]
        )

        return get_obstore
    except ImportError:
        pass
    return None


_POOL: OrderedDict[tuple[str, str, str | None, str | None], Any] = OrderedDict()
_POOL_MAXSIZE = 64
_POOL_LOCK = threading.Lock()


def _pool_key(uri: str) -> tuple[str, str, str | None, str | None]:
    """Return ``(scheme, bucket, region, endpoint)`` for ``uri``.

    Kept byte-identical to the sibling pools so the three packages
    cache by the same key.
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
        return AzureStore(bucket, **options)
    if scheme in ("http", "https"):
        origin = f"{scheme}://{parsed.netloc}"
        return HTTPStore.from_url(origin, **options)
    raise ValueError(
        f"obstore client pool: unsupported scheme {scheme!r} for URI {uri!r}. "
        "Supported: s3, gs, gcs, az, azure, abfs, http, https."
    )


def get_obstore(
    uri: str,
    *,
    storage_options: dict[str, Any] | None = None,
) -> ObjectStore:
    """Return the pooled :class:`ObjectStore` for ``uri``.

    When ``geocatalog`` or ``geopatcher`` is installed the call is
    forwarded to their pool so all three libraries share clients.
    Otherwise the local pool is used with identical LRU + fork-safety
    semantics.

    Raises:
        ImportError: ``[obstore]`` extra not installed.
        ValueError: URI scheme not handled by an obstore backend.
    """
    shared = _try_shared_pool()
    if shared is not None:
        return shared(uri, storage_options=storage_options)

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


async def read_byte_range(
    uri: str,
    start: int,
    length: int,
    *,
    storage_options: dict[str, Any] | None = None,
    store: Any = None,
) -> bytes:
    """Fetch ``length`` bytes from ``start`` on ``uri``.

    Args:
        uri: Cloud URI of the object.
        start: Byte offset of the read.
        length: Number of bytes.
        storage_options: Forwarded to :func:`get_obstore` on the first
            call for a given pool key.
        store: Optional pre-built ``ObjectStore`` instance to use
            instead of the pool. Useful for tests with ``LocalStore``
            or ``MemoryStore`` — and for advanced users whose
            credential / endpoint config doesn't fit the
            environment-driven pool key.

    Returns:
        The requested byte range as a ``bytes`` object.
    """
    if store is None:
        store = get_obstore(uri, storage_options=storage_options)
    path = urlsplit(uri).path.lstrip("/")
    blob = await store.get_range_async(path, start=start, length=length)
    return bytes(blob)


def clear_obstore_pool() -> None:
    """Drop every pooled client. No-op when a sibling pool owns the cache."""
    with _POOL_LOCK:
        _POOL.clear()


def _clear_after_fork() -> None:
    clear_obstore_pool()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_clear_after_fork)
