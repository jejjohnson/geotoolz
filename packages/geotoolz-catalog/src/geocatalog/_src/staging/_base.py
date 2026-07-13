"""`stage()` + `LocalCache` — resolve remote URIs into a local cache.

The staging layer is the bytes-on-disk side of the
discovery/matchup/staging trio. Catalog ingestion records URIs
(``s3://``, ``gs://``, ``https://``, …); ``stage()`` materialises
those URIs into a `LocalCache` and returns a new catalog whose
``filepath`` (and asset map, when present) points at the cached
copies.

Two key design points:

* fsspec handles the URI scheme dispatch. Any URI scheme fsspec
  knows about works transparently — S3, GCS, HTTPS, Azure Blob,
  the local filesystem itself (as a no-op clone).
* Cache key is the SHA-256 of the URI plus the original file
  extension. Two URIs that resolve to "the same file" by content
  are intentionally not deduped here — staging is about
  reproducibility (same URI → same cache slot), not deduplication.

Asset-aware: when a catalog row's ``extras["assets"]`` is the
JSON-encoded dict produced by `CatalogBundle.ingest`, each named
asset is staged independently and the row's asset map is
rewritten to local paths. When ``assets`` is absent (i.e. the
row came from `build_raster_catalog` or similar), only the
top-level ``filepath`` is staged.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import warnings
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import geopandas as gpd
from loguru import logger

from geocatalog._src.retry import _is_transient


if TYPE_CHECKING:
    from os import PathLike

    from geocatalog._src.base import GeoCatalog


# Default cache root resolved at first use rather than import
# time, so a process that overrides $GEOCATALOG_CACHE just before
# calling `stage()` still sees it.
_DEFAULT_CACHE_SUBDIR = ".cache/geocatalog"


@dataclasses.dataclass
class LocalCache:
    """fsspec-backed cache for staged remote files.

    Files land at ``{root}/{hash(uri)[:2]}/{hash(uri)}{ext}``. The
    two-letter prefix keeps any one directory under a few thousand
    entries on a large catalog — friendly to filesystems that
    paginate big directories.

    Args:
        root: Directory the cache lives under. ``None`` resolves
            ``$GEOCATALOG_CACHE`` (when set) or
            ``~/.cache/geocatalog``. The resolution is lazy so
            tests can override the env var before each call.
        ttl_days: Optional lifetime. When set, cached files older
            than this many days are re-downloaded. ``None`` means
            cache forever.
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
        digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
        ext = _ext_for(uri)
        root = self.resolve_root()
        return root / digest[:2] / f"{digest}{ext}"

    def is_fresh(self, path: Path) -> bool:
        """Is the cached file present and within TTL?"""
        if not path.exists():
            return False
        if self.ttl_days is None:
            return True
        age = datetime.now(tz=UTC) - datetime.fromtimestamp(
            path.stat().st_mtime, tz=UTC
        )
        return age < timedelta(days=self.ttl_days)


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
        assets: When the row carries a JSON-encoded asset map (see
            `CatalogBundle.ingest`), only fetch these keys. ``None``
            stages every asset present on each row. Ignored for
            rows that have no asset map (only ``filepath`` is
            staged).
        parallel: Max concurrent fetches via a
            `ThreadPoolExecutor`. The fsspec backends release the
            GIL on I/O so threads scale well even in pure Python.
        cache: Reuse an existing cache instance. ``None`` builds a
            default one bound to ``dest`` (or the env-var default).
        retries: Per-asset retry budget for *transient* failures
            only (network blips, partial reads — the classification
            in `geocatalog._src.retry`). Fatal errors such as
            `FileNotFoundError` or `PermissionError` fail the asset
            immediately without burning the retry budget; either way
            a failed asset is then subject to ``on_error``.
        on_error: ``"raise"`` (default) — any failed asset stops
            the stage and propagates. ``"skip"`` — keep the
            original URI in the asset map and continue; the row
            is emitted with whatever did succeed.

    Returns:
        A new catalog of the same backend type. Each row's
        ``filepath`` points at the cached primary asset; when the
        row had an asset map, that map is rewritten to local
        paths. The original URIs are preserved under
        ``extras["_staged_from"]`` (a JSON dict mirroring
        ``assets``).
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

    cache = cache or LocalCache(root=dest)

    # Materialise the work list (one row → many assets) so we can
    # parallelise the downloads.
    plans: list[_RowPlan] = []
    for idx, row in enumerate(catalog.gdf.itertuples()):
        plan = _plan_row(row, idx, asset_filter=assets)
        plans.append(plan)

    # Flatten to a flat list of (row_idx, asset_key, uri) tuples
    # so the pool sees independent units of work.
    work: list[tuple[int, str, str]] = []
    for plan in plans:
        for key, uri in plan.assets.items():
            work.append((plan.row_idx, key, uri))

    # Fetch in parallel. Each future returns (row_idx, key, local_path
    # or Exception); the main thread sorts results back into the plans.
    # Catch `Exception` (not `BaseException`) so KeyboardInterrupt /
    # SystemExit still stop staging immediately.
    failures: dict[tuple[int, str], Exception] = {}
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {
            pool.submit(_fetch_one, uri, cache, retries): (row_idx, key, uri)
            for row_idx, key, uri in work
        }
        for fut in as_completed(futures):
            row_idx, key, uri = futures[fut]
            try:
                local_path = fut.result()
            except Exception as exc:
                if on_error == "raise":
                    raise
                logger.warning(
                    "stage: skipping {!r} for row {} key {!r}: {}",
                    uri,
                    row_idx,
                    key,
                    exc,
                )
                failures[(row_idx, key)] = exc
                continue
            plans[row_idx].results[key] = str(local_path)

    # Build the new GeoDataFrame from the plans.
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
    results: dict[str, str] = dataclasses.field(default_factory=dict)


def _plan_row(row: Any, idx: int, *, asset_filter: list[str] | None) -> _RowPlan:
    """Build a `_RowPlan` from one `itertuples` row.

    Handles both shapes:
    * Rows with a JSON-encoded ``assets`` column (CatalogBundle
      ingest output) — stage every named asset (or the filtered
      subset).
    * Rows with only ``filepath`` (legacy / build_raster_catalog) —
      stage just the filepath under the synthetic key
      ``"_filepath"``.
    """
    fields = row._asdict() if hasattr(row, "_asdict") else dict(row.__dict__)
    primary = str(fields.get("filepath", ""))
    asset_blob = fields.get("assets")
    assets_map: dict[str, str] = {}
    has_map = False
    if isinstance(asset_blob, str) and asset_blob.startswith("{"):
        try:
            decoded = json.loads(asset_blob)
        except json.JSONDecodeError:
            decoded = {}
        if isinstance(decoded, dict) and decoded:
            assets_map = {str(k): str(v) for k, v in decoded.items()}
            has_map = True
    if not has_map and primary:
        assets_map = {"_filepath": primary}
    if asset_filter is not None and has_map:
        assets_map = {k: v for k, v in assets_map.items() if k in asset_filter}
    return _RowPlan(
        row_idx=idx,
        primary_uri=primary,
        assets=assets_map,
        has_asset_map=has_map,
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
    ``failures`` is the set of ``(row_idx, key)`` pairs the executor
    captured; anything else absent from ``plan.results`` is treated
    as a fetch that simply wasn't attempted.
    """
    new_filepaths: list[str] = []
    new_assets: list[str] = []
    staged_from: list[str] = []

    for plan in plans:
        if plan.has_asset_map:
            # Preserve the original dict's key order. For each asset:
            #   - success → local path
            #   - failure under on_error="skip" → original URI
            #   - never attempted → omitted from the map
            local_map: dict[str, str] = {}
            for key, uri in plan.assets.items():
                local = plan.results.get(key)
                if local is not None:
                    local_map[key] = local
                elif (plan.row_idx, key) in failures:
                    local_map[key] = uri
            # Prefer a real local path for the primary; fall back to
            # the first surviving entry (which may itself be a URI
            # under "skip"), then to the row's original primary.
            primary_local = next(
                (v for k, v in local_map.items() if k in plan.results),
                next(iter(local_map.values()), plan.primary_uri),
            )
            new_filepaths.append(primary_local)
            new_assets.append(json.dumps(local_map))
            staged_from.append(json.dumps(plan.assets))
        else:
            # Legacy: just `_filepath` under the synthetic key.
            local = plan.results.get("_filepath", plan.primary_uri)
            new_filepaths.append(local)
            new_assets.append("")
            staged_from.append(json.dumps({"_filepath": plan.primary_uri}))

    new_gdf = src.copy()
    new_gdf["filepath"] = new_filepaths
    # Preserve / overwrite the `assets` column.
    new_gdf["assets"] = new_assets
    new_gdf["_staged_from"] = staged_from
    return new_gdf


# ---------------------------------------------------------------------------
# Per-asset fetch with retry
# ---------------------------------------------------------------------------


def _fetch_one(uri: str, cache: LocalCache, retries: int) -> Path:
    """Download a single URI into the cache; return the local path.

    Skips the download when the cached file already exists and is
    within TTL. For local URIs (no scheme or ``file://``) we
    avoid the fsspec copy and link / return the existing path.

    Only *transient* failures (per `geocatalog._src.retry`'s
    classification — network blips, partial reads, non-fatal
    `OSError`) are retried, up to ``retries`` times with bounded
    exponential backoff. Fatal errors (`FileNotFoundError`,
    `PermissionError`, …) propagate immediately without retrying.
    When ``cache.timeout`` is set it is forwarded to ``fsspec.open``
    (enforcement is filesystem-dependent — see `LocalCache`).
    """
    import fsspec

    dest = cache.path_for(uri)
    if cache.is_fresh(dest):
        logger.debug("stage: cache hit {!r} → {}", uri, dest)
        return dest

    # Local-file fast path — already on disk, no network round-trip.
    parsed = urlparse(uri)
    if parsed.scheme in {"", "file"}:
        local = Path(parsed.path or uri)
        if local.exists():
            return local

    # Only forward `timeout` when set: fsspec passes unknown kwargs
    # through to the backend, and omitting the key entirely is the
    # safest "disabled" spelling across filesystem implementations.
    open_kwargs: dict[str, Any] = {}
    if cache.timeout is not None:
        open_kwargs["timeout"] = cache.timeout

    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        try:
            with (
                fsspec.open(uri, mode="rb", **open_kwargs) as src,
                dest.open("wb") as dst,
            ):
                # Stream in chunks so we don't materialise a 5 GB
                # asset into memory. fsspec's `open` returns a
                # file-like; `read1`/`readinto` would be marginally
                # faster but `read(chunk)` is portable across backends.
                while True:
                    chunk = src.read(8 * 1024 * 1024)  # 8 MB
                    if not chunk:
                        break
                    dst.write(chunk)
            return dest
        except Exception as exc:
            # On any failure, scrub a partial file so the next
            # attempt starts clean.
            if dest.exists():
                with contextlib.suppress(OSError):
                    dest.unlink()
            # Fatal errors (404-style missing objects, auth /
            # permission problems, …) will not heal on retry —
            # propagate immediately instead of burning the budget.
            if not _is_transient(exc) or attempt >= retries:
                raise
            # Exponential-ish backoff: 0.5 / 1.0 / 2.0 / ...
            # Bounded at 16s to keep total retry time predictable.
            import time

            sleep_for = min(0.5 * (2**attempt), 16.0)
            logger.debug(
                "stage: retry {}/{} for {!r} after {}s ({})",
                attempt + 1,
                retries,
                uri,
                sleep_for,
                exc,
            )
            time.sleep(sleep_for)
    raise AssertionError("unreachable")  # pragma: no cover


def _ext_for(uri: str) -> str:
    """Return the file extension (with dot) for a URI; empty string if none."""
    leaf = urlparse(uri).path.rsplit("/", 1)[-1]
    if "." not in leaf:
        return ""
    return "." + leaf.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# Module-level helpers used by tests
# ---------------------------------------------------------------------------


def _normalize_assets_for_filter(
    assets: Iterable[str] | None,
) -> list[str] | None:
    """Coerce ``None`` / iterable to a concrete list-or-None."""
    if assets is None:
        return None
    out = list(assets)
    if not out:
        warnings.warn(
            "stage(assets=[]) requested with no keys; the result will "
            "have empty asset maps. Pass `assets=None` to stage all.",
            stacklevel=2,
        )
    return out


__all__ = ["LocalCache", "stage"]
