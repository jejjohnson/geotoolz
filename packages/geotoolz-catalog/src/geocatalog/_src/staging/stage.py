"""`stage()` — resolve a catalog's remote URIs into a local cache.

The staging layer is the bytes-on-disk side of the
discovery/matchup/staging trio. Catalog ingestion records URIs
(``s3://``, ``gs://``, ``https://``, …); ``stage()`` materialises
those URIs into a `geocloud.cache.LocalCache` and returns a new catalog
whose ``filepath`` (and asset map, when present) points at the cached
copies.

Design points:

* Local paths (no scheme, ``file://``) are used in place — no copy, so
  staging a local catalog works on a base install. Remote URIs
  (``s3://``, ``gs://``, ``az://`` / ``abfs[s]://``, ``http(s)://``,
  ``hf://``) go through `geocloud.cache.LocalCache.fetch` — a download
  with `geocloud.files` on the stack's shared obstore client pool, with
  the credentials registered in `geocloud.credentials` (``pip install
  'geotoolz-catalog[cloud]'``).
* The cache (slot layout, signature-stripped keys, TTL, ``.part``
  downloads renamed into place) is geotoolz-cloud's; each distinct URI is
  fetched once per call however many rows share it.

Asset-aware: when a catalog row's ``assets`` is an asset map (the
JSON-encoded dict produced by `CatalogBundle.ingest`, or a dict),
each named asset is staged independently and the map is rewritten to
local paths. When ``assets`` is absent (the row came from
`build_raster_catalog` or similar), only ``filepath`` is staged.
"""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import geopandas as gpd
from loguru import logger

from geocatalog._src._extras import missing_extra
from geocatalog._src.utils.retry import retry_transient_io
from geocatalog._src.utils.uri import parse_uri


if TYPE_CHECKING:
    from os import PathLike

    from geocloud.cache import LocalCache

    from geocatalog._src.base import GeoCatalog


#: Column holding each staged row's original URIs (JSON, keyed like
#: ``assets``; ``{"filepath": uri}`` for rows without an asset map).
STAGED_FROM_COLUMN = "staged_from"


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
        dest: The cache root when ``cache`` is ``None``. ``None`` uses
            `geocloud.cache.LocalCache`'s default (``$GEOCLOUD_CACHE``,
            else ``~/.cache/geocloud``).
        assets: When rows carry an asset map (see
            `CatalogBundle.ingest`), only fetch these keys. ``None``
            stages every asset present on each row. Rows that have
            no asset map stage ``filepath`` regardless.
        parallel: Max concurrent fetches via a
            `ThreadPoolExecutor`. obstore releases the GIL on I/O, so
            threads scale well.
        cache: The `geocloud.cache.LocalCache` to fetch into. ``None``
            builds one bound to ``dest`` when a remote URI needs it.
        retries: Per-URI retry budget for *transient* failures only
            (network failures left after the object-store client's own
            retries, short reads — the shared policy of
            `geocatalog._src.utils.retry.retry_transient_io`). Fatal errors
            such as `FileNotFoundError` or `PermissionError` fail the
            URI immediately; either way a failed URI is then subject
            to ``on_error``.
        on_error: ``"raise"`` (default) — the first failure cancels
            the downloads not yet started and propagates. ``"skip"``
            — keep the original URI in the asset map and continue;
            the row is emitted with whatever did succeed.

    Returns:
        A new `InMemoryGeoCatalog` (the input is not mutated). Each row's
        ``filepath`` points at the local copy of its primary asset
        (the asset whose URI was the row's ``filepath``); it keeps
        its original URI when that asset was not staged. An asset map
        is rewritten to local paths. The original URIs are kept in
        the ``staged_from`` column (``extras["staged_from"]``), a JSON
        dict keyed like ``assets``.

    Raises:
        ValueError: If ``assets`` is empty or names a key that no row
            carries, or an argument is out of range.
        ModuleNotFoundError: If a remote URI needs geotoolz-cloud (the
            ``[cloud]`` extra) and it is not installed.
    """
    from geocatalog._src.backends.memory import InMemoryGeoCatalog

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

    if cache is None and any(_local_path(uri) is None for uri in users):
        cache = _cloud_cache().LocalCache(root=dest)

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
    return InMemoryGeoCatalog(new_gdf, kind=catalog.kind)


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
    return parse_uri(uri).local_path()


def _cloud_cache() -> ModuleType:
    """`geocloud.cache`, or an error naming the ``[cloud]`` extra."""
    try:
        from geocloud import cache
    except ImportError as exc:
        raise missing_extra(
            "stage: fetching remote URIs", "cloud", packages="geotoolz-cloud"
        ) from exc
    return cache


def _fetch_one(uri: str, cache: LocalCache | None, retries: int) -> Path:
    """Resolve a single URI to a local file; return its path.

    Local URIs (no scheme or ``file://``) are returned in place —
    nothing is copied and geotoolz-cloud is not needed; a missing local
    file raises `FileNotFoundError`. Remote URIs go through
    `geocloud.cache.LocalCache.fetch` (a cache hit, or a download into the
    cache), with transient failures retried by the shared
    `retry_transient_io` policy.
    """
    local = _local_path(uri)
    if local is not None:
        if not local.exists():
            raise FileNotFoundError(f"stage: local file not found: {uri}")
        return local
    assert cache is not None  # `stage` builds one when a URI is remote
    return retry_transient_io(cache.fetch, uri, retries=retries)


__all__ = ["stage"]
