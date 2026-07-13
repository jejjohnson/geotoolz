"""Process-global ``obstore`` client pool for fast COG range reads.

The single point of object-store client construction for any catalog
code path that wants connection reuse. A
``dict[(scheme, bucket, region, endpoint) -> ObjectStore]`` is held
once per process so the underlying HTTP/2 connection pool — and thus
connection / TLS reuse — survives across every file we open in one
Python session.

Surfaces:

- `get_obstore` — return a pooled :class:`obstore.store.ObjectStore`
  for a URI.
- :func:`get_range_bytes` — async helper that resolves a URI to its
  pool entry and fetches a byte range. Not currently called by the
  catalog builder — rasterio uses its own GDAL CURL stack for the
  bounds-extraction path — but exposed for downstream packages
  (``geopatcher``'s upcoming ``ObstoreCogField`` is the first
  consumer) and for users who want to drive header prefetch directly.
- `clear_obstore_pool` — explicit pool teardown for long-lived
  processes (notebooks).
- :func:`set_obstore_pool_maxsize` — LRU cap.

The pool is gated on the ``[obstore]`` extra; importing this module
without it installed succeeds, but any call raises ``ImportError``
with the install hint. This keeps the rest of the catalog importable
on a slim install.

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
    "The obstore client pool requires the [obstore] extra; install via "
    "`pip install 'geocatalog[obstore]'`."
)


# ``OrderedDict`` so we can implement LRU eviction without a second
# data structure — long-lived notebook sessions can otherwise leak
# clients (one per unique bucket + region they touch). 64 is the
# default cap; tune via :func:`set_obstore_pool_maxsize`.
_POOL: OrderedDict[tuple[str, str, str | None, str | None], Any] = OrderedDict()
_POOL_MAXSIZE = 64
_POOL_LOCK = threading.Lock()


def _pool_key(uri: str) -> tuple[str, str, str | None, str | None]:
    """Return the dict key ``(scheme, bucket, region, endpoint)`` for ``uri``.

    Region and endpoint are resolved from the surrounding environment
    (``AWS_REGION`` / ``AWS_S3_ENDPOINT`` and equivalents) rather than
    the URI itself; storing them in the key means a process talking to
    two regions of the same bucket gets two pool entries (the right
    answer — they're different HTTP/2 connections).
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
    """Construct a fresh `ObjectStore` for ``uri``.

    Dispatches on URI scheme to the corresponding obstore backend
    (S3, GCS, Azure, HTTP). Storage options are passed through; nothing
    is interpreted here.
    """
    try:
        from obstore.store import (
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
        # `_object_key` below.
        return AzureStore.from_url(uri, **options)
    if scheme in ("http", "https"):
        # HTTPStore takes the full origin (scheme://host[:port]); the
        # path of the URI is the per-request key.
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
    """Return the pooled `ObjectStore` for ``uri``.

    Calls share clients across the process when the
    ``(scheme, bucket, region, endpoint)`` key matches. The first call
    for a key constructs the client; subsequent calls return the same
    instance, so HTTP/2 connection pooling and TLS sessions survive.

    Args:
        uri: A cloud URI (``s3://...``, ``gs://...``, ``https://...``,
            etc.). Local paths raise ``ValueError`` — they don't need
            an obstore client.
        storage_options: Passed through to the underlying obstore
            constructor on the *first* call for a given key; subsequent
            calls ignore it (the client is already built). For
            per-call overrides, call `clear_obstore_pool` first.

    Raises:
        ImportError: ``[obstore]`` extra not installed.
        ValueError: URI scheme is not handled by an obstore backend.
    """
    key = _pool_key(uri)
    with _POOL_LOCK:
        existing = _POOL.get(key)
        if existing is not None:
            # Move to end for LRU semantics.
            _POOL.move_to_end(key)
            return existing
        store = _build_store(uri, storage_options)
        _POOL[key] = store
        while len(_POOL) > _POOL_MAXSIZE:
            _POOL.popitem(last=False)
        return store


def _object_key(uri: str) -> str:
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


async def get_range_bytes(
    uri: str,
    start: int,
    length: int,
    *,
    storage_options: dict[str, Any] | None = None,
) -> bytes:
    """Fetch ``length`` bytes starting at ``start`` from ``uri``.

    Async wrapper around `ObjectStore.get_range_async`. Resolves
    the URI's path component to the obstore key (the bucket+region is
    in the pool entry; the path is the per-request key).

    Args:
        uri: Cloud URI of the object to read.
        start: Byte offset of the read.
        length: Number of bytes to read.
        storage_options: Forwarded to `get_obstore` on the first
            call for a given pool key.

    Returns:
        The requested byte range as a ``bytes`` object.
    """
    store = get_obstore(uri, storage_options=storage_options)
    blob = await store.get_range_async(_object_key(uri), start=start, length=length)
    return bytes(blob)


def clear_obstore_pool() -> None:
    """Drop every pooled client.

    Use after changing AWS / GCS / Azure credentials in a long-running
    process, or before a fork (the registered ``after_in_child`` hook
    does this automatically).
    """
    with _POOL_LOCK:
        _POOL.clear()


def set_obstore_pool_maxsize(maxsize: int) -> None:
    """Set the LRU cap for the pool.

    Lower the cap on memory-constrained servers; raise it for
    notebooks talking to many endpoints. Default 64.
    """
    global _POOL_MAXSIZE
    if maxsize < 1:
        raise ValueError(f"maxsize must be >= 1, got {maxsize}")
    with _POOL_LOCK:
        _POOL_MAXSIZE = maxsize
        while len(_POOL) > _POOL_MAXSIZE:
            _POOL.popitem(last=False)


# Fork-safety: reqwest connection pools (the engine inside obstore) are
# not fork-safe. Clear the pool in the child after fork so each child
# rebuilds its own clients on first use. ``os.register_at_fork`` is only
# present on POSIX; Windows (no fork) skips the registration silently.
def _clear_after_fork() -> None:
    clear_obstore_pool()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_clear_after_fork)
