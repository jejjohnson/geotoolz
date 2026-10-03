"""Process-global ``obstore`` client pool — the one pool of the geotoolz stack.

geopatcher owns the pool and `ObstoreCogField` reads through it;
``geotoolz`` (sensor-reader byte path) imports it from the public
`geopatcher.objstore` module, and ``geotoolz-catalog[obstore]`` installs
it. A process that talks to one bucket through the whole stack therefore
builds one client and one HTTP/2 connection pool for it, and
`clear_obstore_pool` empties it for every package.

Surfaces:

- `get_obstore` — the pooled ``ObjectStore`` for a URI.
- `object_key` — the key to request from that store for the same URI.
- `get_range_bytes` — async byte-range fetch through the pool.
- `clear_obstore_pool` — drop every pooled client (also runs after fork).
- `set_obstore_pool_maxsize` — LRU cap (default 64).

Every pooled store is built **without a prefix**: a client is shared by
every object under its bucket / container, so baking the first object's
path into it would corrupt every later request. `object_key` therefore
always returns the full key within the bucket / container, and
``get_obstore(uri)`` + ``object_key(uri)`` always address ``uri``.

Supported URI forms (store → key):

- ``s3://bucket/key`` (``s3a``), ``gs://bucket/key`` (``gcs``) →
  ``S3Store(bucket)`` / ``GCSStore(bucket)``, ``key``.
- ``az://account/container/key`` (``azure``),
  ``abfs[s]://container@account.dfs.core.windows.net/key`` and
  ``https://account.blob.core.windows.net/container/key`` →
  ``AzureStore(container_name=container, account_name=account)``, ``key``.
- ``http[s]://host/path[?query]`` → ``HTTPStore(origin[?query])``,
  ``path``.
- ``hf://[datasets/|spaces/]org/repo[@revision]/path`` (Hugging Face Hub)
  → ``HTTPStore`` on ``$HF_ENDPOINT`` (default ``https://huggingface.co``),
  ``[datasets/|spaces/]org/repo/resolve/<revision or main>/path``.

The query string of an ``http(s)`` URI (pre-signed S3 / GCS URLs, Azure
SAS tokens) lives in the store's base URL, because obstore
percent-encodes a ``?`` that appears in a request key; ``HTTPStore``
keeps its base URL's query on every request it makes. Each distinct
query therefore gets its own pooled client. An Azure ``https://`` URL
*with* a query is treated as a plain signed HTTP URL for the same
reason.

Adapted from the ``openEO-RuSTAC`` Rust pattern in
``crates/orbit-geo/src/async_download.rs:76-140``.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit


if TYPE_CHECKING:
    from obstore.store import ObjectStore


_OBSTORE_INSTALL_HINT = (
    "The obstore client pool requires the `obstore` package; install via "
    "`pip install 'geotoolz-patcher[obstore]'`."
)

_S3_SCHEMES = frozenset({"s3", "s3a"})
_GCS_SCHEMES = frozenset({"gs", "gcs"})
_AZ_SCHEMES = frozenset({"az", "azure"})
_ABFS_SCHEMES = frozenset({"abfs", "abfss"})
_HTTP_SCHEMES = frozenset({"http", "https"})
_HF_SCHEMES = frozenset({"hf"})
_AZURE_HOST_SUFFIXES = (".blob.core.windows.net", ".dfs.core.windows.net")

SUPPORTED_SCHEMES: frozenset[str] = (
    _S3_SCHEMES
    | _GCS_SCHEMES
    | _AZ_SCHEMES
    | _ABFS_SCHEMES
    | _HTTP_SCHEMES
    | _HF_SCHEMES
)
"""URI schemes the pool can build a client for."""


@dataclass(frozen=True)
class _Location:
    """A URI resolved into ``(backend, bucket, container/query, key)``.

    ``bucket`` is the S3 / GCS bucket, the Azure storage account, or the
    ``host[:port]`` of an HTTP origin. ``scope`` is the Azure container
    or the HTTP query string (``None`` otherwise).
    """

    backend: str  # "s3" | "gcs" | "azure" | "http"
    scheme: str
    bucket: str
    scope: str | None
    key: str


def _azure_account_from_host(host: str, uri: str) -> str:
    account = host.split(".", 1)[0]
    if not account:
        raise ValueError(f"obstore client pool: no Azure account in {uri!r}.")
    return account


def _split_container(path: str, uri: str) -> tuple[str, str]:
    container, _, key = path.partition("/")
    if not container:
        raise ValueError(
            f"obstore client pool: no Azure container in {uri!r}; expected "
            "the container as the first path segment."
        )
    return container, key


def _locate_hf(uri: str, netloc: str, path: str) -> _Location:
    """``hf://`` → the Hub's ``resolve`` URL, served by an `HTTPStore`."""
    parts = [p for p in f"{netloc}/{path}".split("/") if p]
    kind = parts.pop(0) + "/" if parts and parts[0] in ("datasets", "spaces") else ""
    if len(parts) < 3:
        raise ValueError(
            f"obstore client pool: {uri!r} must name `org/repo/path` (after an "
            "optional `datasets/` or `spaces/`)."
        )
    org, repo, rest = parts[0], parts[1], "/".join(parts[2:])
    repo, _, revision = repo.partition("@")
    endpoint = urlsplit(os.environ.get("HF_ENDPOINT") or "https://huggingface.co")
    return _Location(
        "http",
        endpoint.scheme or "https",
        endpoint.netloc,
        None,
        f"{kind}{org}/{repo}/resolve/{revision or 'main'}/{rest}",
    )


def _locate(uri: str) -> _Location:
    """Resolve ``uri`` to the backend, bucket / container and object key."""
    parsed = urlsplit(uri)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if scheme in _S3_SCHEMES:
        return _Location("s3", scheme, netloc, None, path)
    if scheme in _GCS_SCHEMES:
        return _Location("gcs", scheme, netloc, None, path)
    if scheme in _AZ_SCHEMES or scheme in _ABFS_SCHEMES:
        if "@" in netloc:
            # ``container@account.<dfs|blob>.core.windows.net/key``
            container, _, host = netloc.partition("@")
            if not container:
                raise ValueError(
                    f"obstore client pool: empty Azure container in {uri!r}."
                )
            return _Location(
                "azure", scheme, _azure_account_from_host(host, uri), container, path
            )
        if scheme in _ABFS_SCHEMES:
            raise ValueError(
                f"obstore client pool: {scheme}:// URIs must name the container "
                "and account as `container@account.dfs.core.windows.net`; got "
                f"{uri!r}."
            )
        # ``az://account/container/key``
        if not netloc:
            raise ValueError(f"obstore client pool: no Azure account in {uri!r}.")
        container, key = _split_container(path, uri)
        return _Location("azure", scheme, netloc, container, key)
    if scheme in _HF_SCHEMES:
        return _locate_hf(uri, netloc, path)
    if scheme in _HTTP_SCHEMES:
        host = (parsed.hostname or "").lower()
        if host.endswith(_AZURE_HOST_SUFFIXES) and not parsed.query:
            # ``https://account.blob.core.windows.net/container/key`` —
            # unsigned, so go through AzureStore and its credential chain.
            container, key = _split_container(path, uri)
            return _Location(
                "azure", scheme, _azure_account_from_host(host, uri), container, key
            )
        return _Location("http", scheme, netloc, parsed.query or None, path)
    raise ValueError(
        f"obstore client pool: unsupported scheme {scheme!r} for URI {uri!r}. "
        f"Supported: {', '.join(sorted(SUPPORTED_SCHEMES))}."
    )


def _freeze(value: Any) -> Hashable:
    """Return a hashable, order-independent stand-in for ``value``."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, list | tuple | set | frozenset):
        items = [_freeze(v) for v in value]
        if isinstance(value, set | frozenset):
            items.sort(key=repr)
        return tuple(items)
    try:
        hash(value)
    except TypeError:
        # Unhashable custom object (e.g. a credential provider): pool by
        # identity — a different provider object is a different client.
        return ("<id>", id(value))
    return value


# Environment variables a backend reads for its endpoint. obstore takes
# the S3 endpoint from ``AWS_ENDPOINT`` as well as ``AWS_ENDPOINT_URL``;
# keying on all of them keeps two endpoints from sharing one client.
_ENDPOINT_VARS: dict[str, tuple[str, ...]] = {
    "s3": (
        "AWS_ENDPOINT",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_S3_ENDPOINT",
    ),
    "gcs": ("GOOGLE_SERVICE_ENDPOINT",),
    "azure": ("AZURE_STORAGE_ENDPOINT", "AZURE_ENDPOINT"),
}


def _env_region_endpoint(
    backend: str,
) -> tuple[str | None, str | None, str | None]:
    """Region, endpoint and profile the backend picks up from the environment.

    The endpoint is every set endpoint variable, joined (``None`` when
    none is set), so changing any of them changes the pool key.
    """
    endpoints = [
        f"{name}={value}"
        for name in _ENDPOINT_VARS.get(backend, ())
        if (value := os.environ.get(name))
    ]
    endpoint = ";".join(endpoints) or None
    if backend == "s3":
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        return region, endpoint, os.environ.get("AWS_PROFILE")
    return None, endpoint, None


_PoolKey = tuple[str, str, str | None, str | None, str | None, str | None, Hashable]


def _pool_key(uri: str, storage_options: Mapping[str, Any] | None = None) -> _PoolKey:
    """Return ``(backend, bucket, scope, region, endpoint, profile, options)``.

    ``scope`` is the Azure container (two containers of one account are
    two clients) or the HTTP query string (each signed URL is its own
    client). Region, endpoint and (S3) profile come from the
    environment, so a process talking to two regions or two
    S3-compatible endpoints of one bucket name gets two entries. The frozen
    ``storage_options`` are part of the key: asking for the same bucket
    with different options returns a different client instead of
    silently reusing the first one's configuration.
    """
    loc = _locate(uri)
    region, endpoint, profile = _env_region_endpoint(loc.backend)
    return (
        loc.backend,
        loc.bucket,
        loc.scope,
        region,
        endpoint,
        profile,
        _freeze(dict(storage_options or {})),
    )


def _azure_options(loc: _Location, options: dict[str, Any]) -> dict[str, Any]:
    """Merge the URI's account / container into ``options``; reject conflicts."""
    if "prefix" in options:
        raise ValueError(
            "obstore client pool: `prefix` is not allowed in storage_options — "
            "pooled stores are shared across objects and must be prefix-free."
        )
    for name, value in (("account_name", loc.bucket), ("container_name", loc.scope)):
        given = options.pop(name, None)
        if given is not None and given != value:
            raise ValueError(
                f"obstore client pool: storage_options[{name!r}]={given!r} "
                f"conflicts with {value!r} from the URI."
            )
    return options


def _build_store(uri: str, storage_options: Mapping[str, Any] | None) -> ObjectStore:
    """Construct a fresh, prefix-free ``ObjectStore`` for ``uri``."""
    loc = _locate(uri)
    try:
        from obstore.store import (
            AzureStore,
            GCSStore,
            HTTPStore,
            S3Store,
        )
    except ImportError as exc:
        raise ImportError(_OBSTORE_INSTALL_HINT) from exc

    options = dict(storage_options or {})
    if loc.backend == "s3":
        return S3Store(loc.bucket, **options)
    if loc.backend == "gcs":
        return GCSStore(loc.bucket, **options)
    if loc.backend == "azure":
        # Built explicitly: ``AzureStore.from_url("az://acct/cont/key")``
        # takes the netloc as the *container* and the whole path as the
        # store prefix, which is wrong for this URI convention.
        options = _azure_options(loc, options)
        return AzureStore(container_name=loc.scope, account_name=loc.bucket, **options)
    base = f"{loc.scheme}://{loc.bucket}"
    if loc.scope is not None:
        base = f"{base}/?{loc.scope}"
    return HTTPStore.from_url(base, **options)


def object_key(uri: str) -> str:
    """Return the key to request from ``get_obstore(uri)`` for ``uri``.

    The path inside the bucket / container, without a leading ``/``.
    For Azure the container is bound into the pooled store, so it is
    not part of the key (``az://acct/cont/a/b.tif`` → ``a/b.tif``). For
    ``http(s)`` the query string is carried by the pooled store, not the
    key.

    Raises:
        ValueError: unsupported scheme or malformed Azure URI.
    """
    return _locate(uri).key


# ``OrderedDict`` so LRU eviction needs no second structure — long-lived
# notebook sessions would otherwise leak one client per bucket they touch.
_POOL: OrderedDict[_PoolKey, Any] = OrderedDict()
_POOL_MAXSIZE = 64
_POOL_LOCK = threading.Lock()


def get_obstore(
    uri: str,
    *,
    storage_options: Mapping[str, Any] | None = None,
) -> ObjectStore:
    """Return the pooled ``ObjectStore`` for ``uri``.

    The first call for a pool key builds the client; later calls with
    the same key return the same instance, so HTTP/2 connection pooling
    and TLS sessions survive across files. Request objects from it with
    `object_key`.

    Args:
        uri: A cloud URI (see the module docstring for the forms).
        storage_options: Keyword arguments for the obstore store
            constructor (``client_options``, ``retry_config``,
            credentials, ...). They are part of the pool key, so
            different options give a different client. ``prefix`` is
            rejected, and Azure ``account_name`` / ``container_name``
            must agree with the URI.

    Raises:
        ImportError: ``obstore`` is not installed.
        ValueError: unsupported scheme, malformed URI, or conflicting
            ``storage_options``.
    """
    key = _pool_key(uri, storage_options)
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


async def get_range_bytes(
    uri: str,
    start: int,
    length: int,
    *,
    storage_options: Mapping[str, Any] | None = None,
    store: ObjectStore | None = None,
) -> bytes:
    """Fetch ``length`` bytes starting at ``start`` from ``uri``.

    Args:
        uri: Cloud URI of the object.
        start: Byte offset of the read.
        length: Number of bytes.
        storage_options: Forwarded to `get_obstore`.
        store: A pre-built store to use instead of the pool (tests with
            ``LocalStore`` / ``MemoryStore``, or credential setups the
            pool key cannot express). It must be laid out like the pooled
            store: rooted at the bucket / container, without a prefix.

    Returns:
        The requested bytes.
    """
    if store is None:
        store = get_obstore(uri, storage_options=storage_options)
    blob = await store.get_range_async(object_key(uri), start=start, length=length)
    return bytes(blob)


def clear_obstore_pool() -> None:
    """Drop every pooled client.

    Use after rotating credentials in a long-running process. Runs
    automatically in a forked child (reqwest connection pools are not
    fork-safe).
    """
    with _POOL_LOCK:
        _POOL.clear()


def set_obstore_pool_maxsize(maxsize: int) -> None:
    """Set the pool's LRU cap (default 64), evicting the oldest entries.

    Raises:
        ValueError: ``maxsize < 1``.
    """
    global _POOL_MAXSIZE
    if maxsize < 1:
        raise ValueError(f"maxsize must be >= 1, got {maxsize}")
    with _POOL_LOCK:
        _POOL_MAXSIZE = maxsize
        while len(_POOL) > _POOL_MAXSIZE:
            _POOL.popitem(last=False)


def _clear_after_fork() -> None:
    clear_obstore_pool()


# ``os.register_at_fork`` is POSIX-only; Windows (no fork) skips it.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_clear_after_fork)
