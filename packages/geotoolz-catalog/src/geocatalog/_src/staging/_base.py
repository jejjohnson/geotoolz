"""`stage()` + `LocalCache` — resolve remote URIs into a local cache.

The staging layer is the bytes-on-disk side of the
discovery/matchup/staging trio. Catalog ingestion records URIs
(``s3://``, ``gs://``, ``https://``, …); ``stage()`` materialises
those URIs into a `LocalCache` and returns a new catalog whose
``filepath`` (and asset map, when present) points at the cached
copies.

Design points:

* Local paths (no scheme, ``file://``) are used in place — no copy and
  no fsspec, so staging a local catalog works on a base install.
  fsspec handles every remote scheme (``pip install
  'geotoolz-catalog[fsspec]'``).
* The cache key is the SHA-256 of the URI (with expiring signature
  parameters removed, so a re-signed URL hits the same slot) plus the
  file extension. It is keyed by *location*, not content: two URIs
  holding the same bytes get two slots.
* Downloads are written to a temporary file and renamed into place,
  so a cache slot only ever holds a complete download; each distinct
  URI is fetched once per call however many rows share it.

Asset-aware: when a catalog row's ``assets`` is an asset map (the
JSON-encoded dict produced by `CatalogBundle.ingest`, or a dict),
each named asset is staged independently and the map is rewritten to
local paths. When ``assets`` is absent (the row came from
`build_raster_catalog` or similar), only ``filepath`` is staged.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import re
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import url2pathname

import geopandas as gpd
from loguru import logger

from geocatalog._src.retry import retry_transient_io


if TYPE_CHECKING:
    from os import PathLike

    from geocatalog._src.base import GeoCatalog


# Default cache root resolved at first use rather than import
# time, so a process that overrides $GEOCATALOG_CACHE just before
# calling `stage()` still sees it.
_DEFAULT_CACHE_SUBDIR = ".cache/geocatalog"

# Suffix of in-progress downloads; `LocalCache.prune` removes leftovers.
# Their names are `.<slot>.<uuid4 hex>.part`, which no cache slot (a
# 64-hex digest plus the source extension) can match.
_PART_SUFFIX = ".part"
_PART_NAME = re.compile(r"^\..+\.[0-9a-f]{32}\.part$")

#: Column holding each staged row's original URIs (JSON, keyed like
#: ``assets``; ``{"filepath": uri}`` for rows without an asset map).
STAGED_FROM_COLUMN = "staged_from"

_CHUNK = 8 * 1024 * 1024  # 8 MB


@dataclasses.dataclass
class LocalCache:
    """fsspec-backed cache for staged remote files.

    Files land at ``{root}/{key[:2]}/{key}{ext}``, where ``key`` is the
    SHA-256 of the URI with expiring signature parameters removed. The
    two-letter prefix keeps any one directory under a few thousand
    entries on a large catalog — friendly to filesystems that
    paginate big directories.

    Args:
        root: Directory the cache lives under. ``None`` resolves
            ``$GEOCATALOG_CACHE`` (when set) or
            ``~/.cache/geocatalog``. The resolution is lazy so
            tests can override the env var before each call.
        ttl_days: Optional lifetime. When set, cached files older
            than this many days are re-downloaded on their next use,
            and `prune` deletes them. ``None`` means cache forever.
        timeout: Per-download timeout in seconds, forwarded to
            ``fsspec.open`` so a stalled remote read cannot hang a
            worker slot forever. ``None`` disables the timeout.
            Enforcement is filesystem-dependent: the keyword is
            passed through to the fsspec backend, and backends that
            do not understand it typically ignore it.
    """

    root: PathLike[str] | str | None = None
    ttl_days: int | None = None
    timeout: float | None = 60.0

    def resolve_root(self) -> Path:
        """Return the resolved cache root (creates it on first call)."""
        if self.root is not None:
            r = Path(self.root)
        else:
            env_root = os.environ.get("GEOCATALOG_CACHE")
            r = Path(env_root) if env_root else Path.home() / _DEFAULT_CACHE_SUBDIR
        r.mkdir(parents=True, exist_ok=True)
        return r

    def path_for(self, uri: str) -> Path:
        """Deterministic cache path for a URI."""
        digest = hashlib.sha256(cache_key(uri).encode("utf-8")).hexdigest()
        return self.resolve_root() / digest[:2] / f"{digest}{_ext_for(uri)}"

    def is_fresh(self, path: Path) -> bool:
        """Is the cached file present and within TTL?"""
        if not path.exists():
            return False
        if self.ttl_days is None:
            return True
        return not self._expired(path)

    def prune(self) -> int:
        """Delete expired files and abandoned partial downloads.

        Returns:
            The number of files removed. Without ``ttl_days`` only
            partial downloads (``*.part``) are removed.
        """
        removed = 0
        for path in self.resolve_root().glob("*/*"):
            if not path.is_file():
                continue
            if _PART_NAME.match(path.name) or (
                self.ttl_days is not None and self._expired(path)
            ):
                with contextlib.suppress(OSError):
                    path.unlink()
                    removed += 1
        return removed

    def _expired(self, path: Path) -> bool:
        assert self.ttl_days is not None
        age = datetime.now(tz=UTC) - datetime.fromtimestamp(
            path.stat().st_mtime, tz=UTC
        )
        return age >= timedelta(days=self.ttl_days)


def stage(
    catalog: GeoCatalog,
    *,
    dest: PathLike[str] | str | None = None,
    assets: list[str] | None = None,
    parallel: int = 8,
    cache: LocalCache | None = None,
    retries: int = 3,
    on_error: str = "raise",
) -> GeoCatalog:
    """Resolve every URI in ``catalog`` into a local file.

    Args:
        catalog: Catalog whose rows reference remote URIs. Currently
            only `InMemoryGeoCatalog` is supported — the function
            returns a fresh in-memory catalog rather than mutating
            the input.
        dest: Override for the cache root. ``None`` defers to
            ``cache.resolve_root()``; if ``cache`` is also None,
            falls back to ``$GEOCATALOG_CACHE`` /
            ``~/.cache/geocatalog``.
        assets: When rows carry an asset map (see
            `CatalogBundle.ingest`), only fetch these keys. ``None``
            stages every asset present on each row. Rows that have
            no asset map stage ``filepath`` regardless.
        parallel: Max concurrent fetches via a
            `ThreadPoolExecutor`. The fsspec backends release the
            GIL on I/O so threads scale well even in pure Python.
        cache: Reuse an existing cache instance. ``None`` builds a
            default one bound to ``dest`` (or the env-var default).
        retries: Per-URI retry budget for *transient* failures only
            (network blips, partial reads — the shared policy of
            `geocatalog._src.retry.retry_transient_io`). Fatal errors
            such as `FileNotFoundError` or `PermissionError` fail the
            URI immediately; either way a failed URI is then subject
            to ``on_error``.
        on_error: ``"raise"`` (default) — the first failure cancels
            the downloads not yet started and propagates. ``"skip"``
            — keep the original URI in the asset map and continue;
            the row is emitted with whatever did succeed.

    Returns:
        A new catalog of the same backend type. Each row's
        ``filepath`` points at the local copy of its primary asset
        (the asset whose URI was the row's ``filepath``); it keeps
        its original URI when that asset was not staged. An asset map
        is rewritten to local paths. The original URIs are kept in
        the ``staged_from`` column (``extras["staged_from"]``), a JSON
        dict keyed like ``assets``.

    Raises:
        ValueError: If ``assets`` is empty or names a key that no row
            carries, or an argument is out of range.
        ModuleNotFoundError: If a remote URI needs fsspec and it is
            not installed.
    """
    from geocatalog._src.memory import InMemoryGeoCatalog

    if not isinstance(catalog, InMemoryGeoCatalog):
        raise TypeError(
            "stage() currently supports InMemoryGeoCatalog only; got "
            f"{type(catalog).__name__}. Convert via "
            "`from_geoparquet(...)` first if you have a "
            "DuckDB-backed catalog."
        )
    if on_error not in {"raise", "skip"}:
        raise ValueError(f"on_error must be 'raise' or 'skip'; got {on_error!r}")
    if retries < 0:
        raise ValueError(f"retries must be >= 0; got {retries!r}")
    if assets is not None and not list(assets):
        raise ValueError("stage(assets=[]) stages nothing; pass assets=None for all")

    cache = cache or LocalCache(root=dest)

    plans = [
        _plan_row(row, idx, asset_filter=assets)
        for idx, row in enumerate(catalog.gdf.itertuples())
    ]
    known = {k for plan in plans for k in plan.all_keys}
    # Rows without an asset map stage `filepath` whatever `assets` says,
    # so a filter only has keys to check when some row carries a map.
    if assets is not None and known:
        unknown = sorted(set(assets) - known)
        if unknown:
            raise ValueError(
                f"stage(assets=...) names keys no row carries: {unknown}; "
                f"known keys: {sorted(known)}"
            )

    # One unit of work per distinct URI, however many rows share it.
    users: dict[str, list[tuple[int, str]]] = {}
    for plan in plans:
        for key, uri in plan.assets.items():
            users.setdefault(uri, []).append((plan.row_idx, key))

    failures: dict[tuple[int, str], Exception] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, parallel))
    try:
        futures: dict[Future[Path], str] = {
            pool.submit(_fetch_one, uri, cache, retries): uri for uri in users
        }
        for fut in as_completed(futures):
            uri = futures[fut]
            try:
                local_path = fut.result()
            except Exception as exc:
                # `Exception`, not `BaseException`: KeyboardInterrupt /
                # SystemExit still stop staging immediately.
                if on_error == "raise":
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
                logger.warning("stage: skipping {!r}: {}", uri, exc)
                for user in users[uri]:
                    failures[user] = exc
                continue
            for row_idx, key in users[uri]:
                plans[row_idx].results[key] = str(local_path)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    new_gdf = _rewrite_gdf(catalog.gdf, plans, failures=failures)
    return InMemoryGeoCatalog(new_gdf, backend=catalog.backend)


# ---------------------------------------------------------------------------
# Plan + rewrite helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _RowPlan:
    """Per-row plan: which URIs to stage + the results."""

    row_idx: int
    primary_uri: str
    assets: dict[str, str]  # key -> uri (subset filtered by `assets=...`)
    has_asset_map: bool
    all_keys: tuple[str, ...] = ()  # every key of the row's map, unfiltered
    map_is_json: bool = True
    results: dict[str, str] = dataclasses.field(default_factory=dict)


def _decode_asset_map(blob: Any) -> tuple[dict[str, str] | None, bool]:
    """``(map, was_json)`` for an asset-map cell; ``(None, _)`` if none."""
    if isinstance(blob, dict):
        decoded: Any = blob
        was_json = False
    elif isinstance(blob, str) and blob.lstrip().startswith("{"):
        try:
            decoded = json.loads(blob)
        except json.JSONDecodeError:
            return None, True
        was_json = True
    else:
        return None, True
    if not isinstance(decoded, dict) or not decoded:
        return None, was_json
    return {str(k): str(v) for k, v in decoded.items()}, was_json


def _plan_row(row: Any, idx: int, *, asset_filter: list[str] | None) -> _RowPlan:
    """Build a `_RowPlan` from one `itertuples` row.

    Handles both shapes:
    * Rows with an asset map (CatalogBundle ingest output) — stage
      every named asset (or the filtered subset).
    * Rows with only ``filepath`` (legacy / build_raster_catalog) —
      stage just the filepath under the synthetic key ``"filepath"``.
    """
    fields = row._asdict() if hasattr(row, "_asdict") else dict(row.__dict__)
    primary = str(fields.get("filepath", "") or "")
    asset_map, was_json = _decode_asset_map(fields.get("assets"))
    if asset_map is None:
        return _RowPlan(
            row_idx=idx,
            primary_uri=primary,
            assets={"filepath": primary} if primary else {},
            has_asset_map=False,
        )
    selected = (
        asset_map
        if asset_filter is None
        else {k: v for k, v in asset_map.items() if k in asset_filter}
    )
    return _RowPlan(
        row_idx=idx,
        primary_uri=primary,
        assets=selected,
        has_asset_map=True,
        all_keys=tuple(asset_map),
        map_is_json=was_json,
    )


def _rewrite_gdf(
    src: gpd.GeoDataFrame,
    plans: list[_RowPlan],
    *,
    failures: dict[tuple[int, str], Exception],
) -> gpd.GeoDataFrame:
    """Build a new GeoDataFrame with rewritten filepath + asset map columns.

    Under ``on_error="skip"``, failed assets keep their original URI
    in the rewritten asset map (matching the documented contract).
    Assets filtered out by ``assets=`` are omitted from the map.
    """
    new_filepaths: list[str] = []
    new_assets: list[Any] = []
    staged_from: list[str] = []

    for plan in plans:
        if not plan.has_asset_map:
            new_filepaths.append(plan.results.get("filepath", plan.primary_uri))
            new_assets.append(None)  # keep the row's original cell
            staged_from.append(json.dumps({"filepath": plan.primary_uri}))
            continue
        local_map: dict[str, str] = {}
        for key, uri in plan.assets.items():
            local = plan.results.get(key)
            if local is not None:
                local_map[key] = local
            elif (plan.row_idx, key) in failures:
                local_map[key] = uri
        # `filepath` follows the primary asset — any staged key holding
        # the row's `filepath` URI (aliases share one download). When it
        # was not staged (filtered out, failed) the row keeps its URI.
        primary_local = next(
            (
                plan.results[k]
                for k, uri in plan.assets.items()
                if uri == plan.primary_uri and k in plan.results
            ),
            None,
        )
        new_filepaths.append(primary_local or plan.primary_uri)
        new_assets.append(json.dumps(local_map) if plan.map_is_json else local_map)
        staged_from.append(json.dumps(plan.assets))

    new_gdf = src.copy()
    new_gdf["filepath"] = new_filepaths
    if "assets" in new_gdf.columns:
        column = new_gdf["assets"].astype(object).tolist()
        new_gdf["assets"] = [
            old if new is None else new
            for old, new in zip(column, new_assets, strict=True)
        ]
    new_gdf[STAGED_FROM_COLUMN] = staged_from
    return new_gdf


# ---------------------------------------------------------------------------
# Per-URI fetch
# ---------------------------------------------------------------------------


def _local_path(uri: str) -> Path | None:
    """The filesystem path a local URI names, ``None`` for remote URIs."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        # `file://server/share/x` is a UNC path; `file:///C:/x` a drive
        # path, which `url2pathname` resolves on Windows.
        path = parsed.path
        if parsed.netloc and parsed.netloc != "localhost":
            path = f"//{parsed.netloc}{path}"
        return Path(url2pathname(path))
    # No scheme, or a Windows drive letter (`C:\\...` parses as scheme "c").
    if parsed.scheme == "" or len(parsed.scheme) == 1:
        return Path(uri)
    return None


def _fetch_one(uri: str, cache: LocalCache, retries: int) -> Path:
    """Resolve a single URI to a local file; return its path.

    Local URIs (no scheme or ``file://``) are returned in place —
    nothing is copied and fsspec is not needed; a missing local file
    raises `FileNotFoundError`. Remote URIs are served from the cache
    when fresh, else downloaded through fsspec into a temporary file
    that is renamed into place only once complete. Transient failures
    are retried with the shared `retry_transient_io` policy.
    """
    local = _local_path(uri)
    if local is not None:
        if not local.exists():
            raise FileNotFoundError(f"stage: local file not found: {uri}")
        return local

    dest = cache.path_for(uri)
    if cache.is_fresh(dest):
        logger.debug("stage: cache hit {!r} → {}", uri, dest)
        return dest
    try:
        import fsspec  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"stage: fetching {uri!r} needs fsspec; install it with "
            "`pip install 'geotoolz-catalog[fsspec]'`."
        ) from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    retry_transient_io(_download, uri, dest, cache.timeout, retries=retries)
    return dest


def _download(uri: str, dest: Path, timeout: float | None) -> None:
    """One download attempt of ``uri`` into ``dest``, atomically."""
    import fsspec

    # Only forward `timeout` when set: fsspec passes unknown kwargs
    # through to the backend, and omitting the key entirely is the
    # safest "disabled" spelling across filesystem implementations.
    open_kwargs: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}{_PART_SUFFIX}")
    try:
        written = 0
        with fsspec.open(uri, mode="rb", **open_kwargs) as src, tmp.open("wb") as dst:
            expected = getattr(src, "size", None)
            # Stream in chunks so a 5 GB asset never sits in memory.
            while chunk := src.read(_CHUNK):
                dst.write(chunk)
                written += len(chunk)
        if isinstance(expected, int) and expected >= 0 and written != expected:
            # A plain OSError is transient: the retry policy re-fetches.
            raise OSError(
                f"stage: short read for {uri!r}: {written} of {expected} bytes"
            )
        os.replace(tmp, dest)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------

# Query parameters that sign a URL rather than select content: a
# re-signed URL for the same object must hit the same cache slot.
_AZURE_SAS = {
    "sig", "se", "st", "sp", "sv", "sr", "spr", "si", "srt", "ss", "sdd",
    "skoid", "sktid", "skt", "ske", "sks", "skv", "saoid", "suoid", "scid",
}  # fmt: skip
_CLOUDFRONT = {"expires", "signature", "key-pair-id", "policy"}


def cache_key(uri: str) -> str:
    """``uri`` without the parameters of an expiring signature.

    Azure SAS (when ``sig`` is present), AWS / GCS query signing
    (``X-Amz-*`` / ``X-Goog-*``) and CloudFront signed-URL parameters
    are dropped; every other query parameter is kept, so URLs that
    select different content keep different keys.
    """
    parsed = urlparse(uri)
    if not parsed.query:
        return uri
    params = parse_qsl(parsed.query, keep_blank_values=True)
    names = {k.lower() for k, _ in params}

    def signing(name: str) -> bool:
        n = name.lower()
        return (
            n.startswith(("x-amz-", "x-goog-"))
            or ("sig" in names and n in _AZURE_SAS)
            or ({"signature", "key-pair-id"} <= names and n in _CLOUDFRONT)
        )

    kept = [(k, v) for k, v in params if not signing(k)]
    if len(kept) == len(params):
        return uri
    return urlunparse(parsed._replace(query=urlencode(kept)))


def _ext_for(uri: str) -> str:
    """Return the file extension (with dot) for a URI; empty string if none."""
    local = _local_path(uri)
    # A local path is not a URL: `#` and `?` are ordinary characters.
    path = local.as_posix() if local is not None else urlparse(uri).path
    leaf = path.rsplit("/", 1)[-1]
    if "." not in leaf:
        return ""
    return "." + leaf.rsplit(".", 1)[-1]


__all__ = ["LocalCache", "cache_key", "stage"]
