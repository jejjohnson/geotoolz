"""`CatalogBundle` implementation.

A bundle holds three things:

* an `InMemoryGeoCatalog` (the items)
* an in-memory list of `QueryRecord` (provenance — which `Source.query`
  call produced which items)
* an in-memory list of `MatchupRow` (matched-row tuples)

Persisted to disk as a directory of three Parquet files plus a JSON
metadata sidecar — see ``to_directory`` / ``from_directory``.

`InMemoryGeoCatalog` itself stays immutable: the bundle mutates its
own internal state and rebuilds a fresh catalog on each ingest so
the catalog's invariants (single CRS, IntervalIndex on time) hold.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import pandas as pd
import pyproj
from loguru import logger
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from geocatalog._src._deprecation import deprecated_alias, renamed_kwargs
from geocatalog._src._schema import CatalogKind, crs_config_string, empty_frame
from geocatalog._src._stac_item import (
    ASSET_CRS_PROPERTY,
    LONLAT,
    is_signed_href,
    reproject_geometry,
)
from geocatalog._src._timeutil import to_naive_utc
from geocatalog._src.memory import InMemoryGeoCatalog


if TYPE_CHECKING:
    from geocatalog._src.matchup.engine import MatchupRow
    from geocatalog._src.sources._base import Source, SourceRow


# The bundle's own schema version, distinct from the catalog's
# `_schema_version` column. Bump on changes to the bundle directory
# layout or the queries/matchups parquet schemas; carried in `_meta.json`.
#
# v2: `matchups.parquet` gains `member_collections`. v1 bundles still
# load (their matchups read back with no collections).
BUNDLE_SCHEMA_VERSION: int = 2

#: The files `to_directory` owns; anything else in the directory is kept.
_BUNDLE_FILES = ("items.parquet", "queries.parquet", "matchups.parquet", "_meta.json")


@dataclasses.dataclass(frozen=True)
class QueryRecord:
    """One row of ``queries.parquet``.

    Persisted alongside the items so a user can answer "which call
    produced these rows?" without storing per-row provenance copies.

    Attributes:
        query_id: Stable identifier (uuid4 hex). Matches the
            ``_provenance['query_id']`` field on every ingested
            `SourceRow` for cross-table joins.
        source: ``SourceRow.source`` value of the rows that came
            from this call (``"earthaccess"``, ``"stac.pc"``, etc.).
        collection: Upstream collection identifier searched.
        bounds_wkt: WKT representation of the search bbox (WGS84).
        time_start, time_end: Time-window endpoints; ``pd.NaT`` if
            the call had no temporal filter.
        filters_json: JSON-encoded filters dict; ``""`` if empty.
        created_at: When the call completed (UTC).
        n_returned: How many rows the call yielded.
        tag: User-supplied label (e.g. ``"iberia_summer24"``); ``None``
            if the call had no tag.
        notes: Free-form text.
    """

    query_id: str
    source: str
    collection: str | None
    bounds_wkt: str
    time_start: datetime | pd.Timestamp | None
    time_end: datetime | pd.Timestamp | None
    filters_json: str
    created_at: datetime
    n_returned: int
    tag: str | None = None
    notes: str | None = None


@renamed_kwargs(target_crs="crs")
def source_row_to_gdf_row(
    row: SourceRow,
    *,
    crs: pyproj.CRS,
    primary_asset: str | None = None,
) -> dict[str, Any]:
    """Map a `SourceRow` to a flat dict suitable for a `GeoDataFrame` row.

    Output keys:
    * ``geometry``: the footprint, reprojected from EPSG:4326 to
      ``crs`` if necessary.
    * ``start_time`` / ``end_time``: pulled from the interval.
    * ``filepath``: chosen asset URL (see ``primary_asset`` resolution).
    * ``id``, ``source``, ``collection``: as-is from the SourceRow.
    * ``assets``, ``properties``, ``provenance``: JSON-encoded
      dicts (Parquet can't store nested dicts directly without
      schema acrobatics; JSON-as-string is the boring-but-robust path).

    Args:
        row: The source row to convert.
        crs: CRS to reproject the geometry into. SourceRow
            geometries come in EPSG:4326 by convention.
        primary_asset: Asset key to promote to ``filepath``. If
            ``None``, the first key in ``row.assets`` is used (Python
            dicts preserve insertion order); if the assets dict is
            empty, ``filepath`` is an empty string.

    Raises:
        KeyError: ``primary_asset`` is given but the row has no such
            asset — silently indexing another band instead would be a
            wrong-data bug, not a fallback.
    """
    # Reproject if needed. The shapely geometry doesn't carry its
    # CRS — we use `crs` as the authoritative target and assume
    # `SourceRow.geometry` is in EPSG:4326 (the `Source` Protocol
    # convention).
    geometry = reproject_geometry(row.geometry, LONLAT, crs)

    # Pick a filepath. For STAC items the "first asset" convention
    # is usually a sensible default (it's typically the lowest-res
    # overview or the canonical band); users who care can pass
    # `primary_asset` explicitly.
    # Strict on an explicit key (#240): silently indexing another band
    # instead would be a wrong-data bug, not a fallback.
    if primary_asset is not None:
        if primary_asset not in row.assets:
            raise KeyError(
                f"source row {row.id!r} has no asset {primary_asset!r}; "
                f"available: {sorted(row.assets)}"
            )
        key: str | None = primary_asset
    else:
        key = next(iter(row.assets), None)
    filepath = row.assets[key] if key is not None else ""
    # Native CRS of the promoted asset, when the source knows it (STAC
    # `proj:*`); granule sources such as CMR leave it unset.
    asset_crs = row.properties.get(ASSET_CRS_PROPERTY)
    crs = asset_crs.get(key) if isinstance(asset_crs, Mapping) else None

    return {
        "geometry": geometry,
        # Naive UTC, the catalog's stored time form: source adapters emit
        # tz-aware intervals, existing catalogs hold naive ones (#231).
        "start_time": to_naive_utc(row.interval.left),
        "end_time": to_naive_utc(row.interval.right),
        "filepath": filepath,
        "crs": crs,
        # Planetary Computer SAS signatures expire after ~1 h; flag them
        # so a persisted bundle says which hrefs need re-signing (#238).
        "href_signed": is_signed_href(filepath),
        "id": row.id,
        "source": row.source,
        "collection": row.collection,
        "assets": json.dumps(dict(row.assets)),
        "properties": json.dumps(dict(row.properties), default=str),
        "provenance": json.dumps(dict(row.provenance), default=str),
    }


# An item's identity in the bundle. Upstream ids are only unique within
# their source and collection (STAC scopes item ids to a collection), so
# the primary key is the triple, not the bare id.
_ItemKey = tuple[str, str, str]


def _row_key(row: SourceRow) -> _ItemKey:
    return (str(row.source), str(row.collection), str(row.id))


def _gdf_keys(gdf: gpd.GeoDataFrame) -> list[_ItemKey]:
    if not {"source", "collection", "id"} <= set(gdf.columns):
        return []
    return [
        (str(s), str(c), str(i))
        for s, c, i in zip(gdf["source"], gdf["collection"], gdf["id"], strict=True)
    ]


def _item_keys(gdf: gpd.GeoDataFrame) -> set[_ItemKey]:
    return set(_gdf_keys(gdf))


def _resolve_duplicates(
    rows: list[SourceRow],
    existing: set[_ItemKey],
    on_duplicate: str,
) -> list[SourceRow]:
    """Apply the ``on_duplicate`` policy to freshly queried rows.

    Rows are keyed by (source, collection, id). Returns the rows to add.
    ``"skip"`` keeps the first occurrence and drops keys already in the
    bundle; ``"replace"`` keeps the last occurrence (the caller removes
    the existing rows it supersedes).
    """
    if on_duplicate == "error":
        seen: set[_ItemKey] = set()
        dups = []
        for r in rows:
            key = _row_key(r)
            if key in existing or key in seen:
                dups.append(key)
            seen.add(key)
        if dups:
            raise ValueError(
                f"ingest: {len(dups)} row(s) already in the bundle or repeated "
                f"in this ingest (e.g. source/collection/id {dups[0]!r}); pass "
                "on_duplicate='skip' to keep the existing rows or "
                "'replace' to overwrite them."
            )
        return rows
    if on_duplicate == "skip":
        out, seen = [], set(existing)
        for r in rows:
            key = _row_key(r)
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out
    latest = {_row_key(r): r for r in rows}
    return list(latest.values())


class CatalogBundle:
    """Wraps an `InMemoryGeoCatalog` + queries + matchups + persistence.

    See module docstring for the directory layout and lifecycle.

    Attributes:
        catalog: The items table as an `InMemoryGeoCatalog`. Updated
            in place when `ingest()` adds new rows.
        crs: Authoritative CRS for the items table's geometry column.
            Set at construction; reprojection happens on ingest, never
            on lookup. (``target_crs`` is the deprecated name.)
        kind: The catalog kind forwarded to the items table.
            (``backend`` is the deprecated name.)
        queries: In-memory list of `QueryRecord`s. Persisted to
            ``queries.parquet`` on `to_directory`.
        matchups: In-memory list of `MatchupRow`s. Persisted to
            ``matchups.parquet``.
    """

    target_crs = deprecated_alias("crs")
    backend = deprecated_alias("kind")

    @renamed_kwargs(target_crs="crs", backend="kind")
    def __init__(
        self,
        catalog: InMemoryGeoCatalog,
        *,
        crs: pyproj.CRS,
        kind: CatalogKind = "raster",
        queries: list[QueryRecord] | None = None,
        matchups: list[MatchupRow] | None = None,
    ) -> None:
        self.catalog = catalog
        self.crs = pyproj.CRS.from_user_input(crs)
        self.kind = kind
        self.queries: list[QueryRecord] = list(queries) if queries else []
        self.matchups: list[MatchupRow] = list(matchups) if matchups else []
        # When the bundle was first written; kept across re-saves.
        self.created_at: str | None = None

    @classmethod
    @renamed_kwargs(target_crs="crs", backend="kind")
    def empty(
        cls,
        *,
        crs: str | pyproj.CRS = "EPSG:4326",
        kind: CatalogKind = "raster",
    ) -> CatalogBundle:
        """Create a fresh bundle with an empty items table.

        Use this as the starting point when ingesting from a `Source`:

            >>> bundle = CatalogBundle.empty(crs="EPSG:4326")
            >>> bundle.ingest(STACSource.planetary_computer(), ...)
            >>> bundle.to_directory("my_catalog/")
        """
        dst_crs = pyproj.CRS.from_user_input(crs)
        # The same columns `source_row_to_gdf_row` writes, so an empty
        # bundle has the schema of an ingested one.
        empty_gdf = empty_frame(
            dst_crs,
            {
                "filepath": "object",
                "crs": "object",
                "href_signed": "bool",
                "id": "object",
                "source": "object",
                "collection": "object",
                "assets": "object",
                "properties": "object",
                "provenance": "object",
            },
        )
        catalog = InMemoryGeoCatalog(empty_gdf, kind=kind)
        return cls(catalog, crs=dst_crs, kind=kind)

    @classmethod
    @renamed_kwargs(backend="kind")
    def from_catalog(
        cls,
        catalog: InMemoryGeoCatalog,
        *,
        kind: CatalogKind | None = None,
    ) -> CatalogBundle:
        """Wrap an already-built `InMemoryGeoCatalog`.

        Useful when the user constructed a catalog via
        ``build_raster_catalog`` / ``build_vector_catalog`` and now
        wants to add queries/matchups state.
        """
        return cls(
            catalog,
            crs=pyproj.CRS.from_user_input(catalog.gdf.crs),
            kind=kind if kind is not None else catalog.kind,
        )

    def ingest(
        self,
        source: Source,
        *,
        bounds: tuple[float, float, float, float],
        interval: pd.Interval | None = None,
        collection: str | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int | None = None,
        primary_asset: str | None = None,
        tag: str | None = None,
        notes: str | None = None,
        on_duplicate: Literal["error", "skip", "replace"] = "error",
    ) -> str:
        """Query a `Source` and append matching rows to the items table.

        Records the call in ``self.queries`` with a fresh UUID so
        every row can be traced back to the call that produced it.

        Args:
            source: Any `Source` Protocol implementer (STACSource,
                EarthAccessSource, ...).
            bounds: ``(xmin, ymin, xmax, ymax)`` in EPSG:4326.
            interval: Optional time window.
            collection: Upstream collection id (passed through to
                ``Source.query``).
            filters: Adapter-specific filter dict.
            limit: Cap on the number of rows returned. ``0`` ingests
                nothing (no request); ``None`` means no cap.
            primary_asset: Asset key to promote to the catalog's
                ``filepath`` column. ``None`` uses the first asset
                key (dict insertion order).
            tag: User label persisted in ``QueryRecord.tag`` and
                propagated to every row via
                ``provenance['query_tag']``.
            notes: Free-form notes recorded in ``QueryRecord.notes``.
            on_duplicate: What to do with a row whose
                ``(source, collection, id)`` — the items table's primary
                key; upstream ids are only unique within a collection —
                is already in the bundle or repeats within this ingest.
                ``"error"`` raises before anything is changed; ``"skip"``
                keeps the existing row (so re-running an ingest is a
                no-op) and never inspects the discarded one;
                ``"replace"`` keeps the newest one and drops the
                matchups that referenced the replaced item, since they
                describe the old data.

        Returns:
            The query_id (uuid4 hex) of this ingest call.

        Raises:
            ValueError: A duplicate ``id`` with ``on_duplicate="error"``,
                or an unknown ``on_duplicate``.
            KeyError: ``primary_asset`` is missing from a row.
        """
        if on_duplicate not in ("error", "skip", "replace"):
            raise ValueError(
                "on_duplicate must be 'error', 'skip' or 'replace'; "
                f"got {on_duplicate!r}"
            )
        from shapely.geometry import box as _shapely_box

        query_id = uuid.uuid4().hex
        created_at = datetime.now(tz=UTC)
        stamped_rows: list[SourceRow] = []
        for row in source.query(
            bounds,
            interval,
            collection=collection,
            filters=filters,
            limit=limit,
        ):
            # Stamp the bundle's query_id onto the row's provenance so
            # downstream tooling can join items ↔ queries without an
            # extra DataFrame merge. We preserve any provenance the
            # adapter already set (e.g. earthaccess might add its own
            # granule UR or doi).
            prov = dict(row.provenance)
            prov.setdefault("query_id", query_id)
            # Use setdefault so an adapter that already set
            # `query_tag` on the row's provenance wins (consistent
            # with the documented "do not overwrite" contract for
            # query_id). If the user's tag and the adapter's
            # disagree, prefer the more-specific (adapter-set) one.
            if tag is not None:
                prov.setdefault("query_tag", tag)
            stamped_rows.append(dataclasses.replace(row, provenance=prov))
        n = len(stamped_rows)

        # Resolve duplicates before mapping, so rows the policy discards
        # are never validated (a skipped row may lack `primary_asset`).
        existing = _item_keys(self.catalog.gdf)
        stamped_rows = _resolve_duplicates(stamped_rows, existing, on_duplicate)
        new_rows = [
            source_row_to_gdf_row(row, crs=self.crs, primary_asset=primary_asset)
            for row in stamped_rows
        ]
        base = self.catalog.gdf
        if on_duplicate == "replace":
            replaced = {_row_key(r) for r in stamped_rows} & existing
            if replaced:
                keep = [k not in replaced for k in _gdf_keys(base)]
                base = base[keep]
                self._drop_matchups_of(replaced)

        if new_rows:
            new_gdf = gpd.GeoDataFrame(new_rows, crs=self.crs)
            new_gdf.index = pd.IntervalIndex.from_arrays(
                new_gdf.pop("start_time"),
                new_gdf.pop("end_time"),
                closed="both",
                name="datetime",
            )
            merged = pd.concat([base, new_gdf], axis=0)
            self.catalog = InMemoryGeoCatalog(
                gpd.GeoDataFrame(merged, crs=self.crs),
                kind=self.kind,
            )

        self.queries.append(
            QueryRecord(
                query_id=query_id,
                source=getattr(source, "name", "unknown"),
                collection=collection,
                bounds_wkt=_shapely_box(*bounds).wkt,
                time_start=(
                    pd.Timestamp(interval.left) if interval is not None else None
                ),
                time_end=(
                    pd.Timestamp(interval.right) if interval is not None else None
                ),
                filters_json=json.dumps(dict(filters), default=str) if filters else "",
                created_at=created_at,
                n_returned=n,
                tag=tag,
                notes=notes,
            )
        )
        return query_id

    def _drop_matchups_of(self, items: set[tuple[str, str, str]]) -> None:
        """Drop matchups with a member in ``items`` ((source, collection, id)).

        A matchup that records its members' collections is matched on
        the full key, so replacing one collection's item keeps the
        matchups of a same-id item in another collection. Older rows
        without collections can only be matched on ``(source, id)``
        and are dropped whenever that pair is replaced.
        """
        pairs = {(src, rid) for src, _, rid in items}

        def stale(m: MatchupRow) -> bool:
            if len(m.member_collections) == len(m.member_ids):
                return any(
                    (str(s), str(c), str(i)) in items
                    for s, c, i in zip(
                        m.member_sources,
                        m.member_collections,
                        m.member_ids,
                        strict=True,
                    )
                )
            return any(
                (str(s), str(i)) in pairs
                for s, i in zip(m.member_sources, m.member_ids, strict=True)
            )

        kept = [m for m in self.matchups if not stale(m)]
        if len(kept) < len(self.matchups):
            logger.warning(
                "ingest: dropped {} matchup(s) referencing replaced items",
                len(self.matchups) - len(kept),
            )
        self.matchups = kept

    def write_matchups(
        self,
        rows: Iterable[MatchupRow],
        *,
        tag: str | None = None,
    ) -> int:
        """Persist a stream of `MatchupRow`s into the bundle.

        Args:
            rows: Iterable of `MatchupRow` (typically the output of
                ``geocatalog.matchup.matchup(...)``).
            tag: Optional ``query_set`` override applied to every
                row. When ``None``, each row keeps its own
                ``query_set``.

        ``matchup_id`` is the table's key: a row whose id is already in
        the bundle replaces the stored one in place (so re-running a
        deterministic matchup does not duplicate it, and a re-run under
        another ``tag`` moves it to that ``query_set``).

        Returns:
            How many rows were written (added or replaced).
        """
        position = {m.matchup_id: i for i, m in enumerate(self.matchups)}
        written = 0
        for row in rows:
            mr = dataclasses.replace(row, query_set=tag) if tag is not None else row
            if mr.matchup_id in position:
                self.matchups[position[mr.matchup_id]] = mr
            else:
                position[mr.matchup_id] = len(self.matchups)
                self.matchups.append(mr)
            written += 1
        return written

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_directory(self, path: str | Path) -> None:
        """Persist the bundle as a directory of Parquet files + metadata.

        Layout::

            <path>/
              items.parquet
              queries.parquet     (omitted when empty)
              matchups.parquet    (omitted when empty)
              _meta.json

        Existing files inside ``path`` are overwritten. Stale
        ``queries.parquet`` / ``matchups.parquet`` from a previous
        write are *removed* when the corresponding in-memory list is
        empty — otherwise `from_directory()` would silently
        resurrect rows that should have been dropped.

        Every component is first written to a hidden staging directory
        next to ``path``; only once all of them exist is the bundle
        published, by swapping that directory in for ``path`` (files the
        bundle does not own are moved across). A reader therefore sees
        either the previous generation or the new one — never new items
        beside old queries or metadata — and a failure while writing
        leaves the previous bundle untouched.
        ``created_at`` records the first save and survives later ones;
        ``updated_at`` is the latest.
        """
        from geocatalog._src.parquet import to_geoparquet

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(tz=UTC).isoformat()
        created_at = self.created_at or _existing_created_at(dest) or now
        meta = {
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            # On-disk keys keep their names (no bundle-schema change).
            "target_crs": crs_config_string(self.crs),
            "backend": self.kind,
            "created_at": created_at,
            "updated_at": now,
        }
        token = uuid.uuid4().hex
        staging = dest.with_name(f".{dest.name}.{token}.new")
        staging.mkdir()
        try:
            to_geoparquet(self.catalog, staging / "items.parquet")
            # Sidecar tables are written only when non-empty, so the
            # directory matches the in-memory contract (empty = omitted).
            if self.queries:
                _queries_to_parquet(self.queries, staging / "queries.parquet")
            if self.matchups:
                _matchups_to_parquet(self.matchups, staging / "matchups.parquet")
            (staging / "_meta.json").write_text(json.dumps(meta, indent=2))
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        _publish_directory(staging, dest, owned=_BUNDLE_FILES, token=token)
        self.created_at = created_at

    @classmethod
    def from_directory(cls, path: str | Path) -> CatalogBundle:
        """Load a bundle from disk. Inverse of `to_directory`."""
        from geocatalog._src.parquet import from_geoparquet

        src = Path(path)
        if not src.is_dir():
            raise NotADirectoryError(
                f"CatalogBundle.from_directory expects a directory; got {src!r}. "
                "If the path is a single GeoParquet file, use "
                "`geocatalog.from_geoparquet(path)` to get an "
                "`InMemoryGeoCatalog`, then wrap it with "
                "`CatalogBundle.from_catalog`."
            )
        meta_path = src / "_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"CatalogBundle directory {src!r} is missing `_meta.json`. "
                "Was it produced by `to_directory`?"
            )
        meta = json.loads(meta_path.read_text())
        # Schema-version gate: reject artifacts written by a newer
        # bundle layout we don't understand. Forward migrations live
        # alongside the catalog's `_schema_version` chain (parquet.py);
        # for now there's only one version and no migrations.
        artifact_version = meta.get("bundle_schema_version")
        if artifact_version is None:
            raise ValueError(
                f"CatalogBundle directory {src!r} `_meta.json` is missing "
                "`bundle_schema_version`; this bundle predates the version "
                "field. Inspect / rewrite via the geocatalog CLI's "
                "migration tools."
            )
        if int(artifact_version) > BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"CatalogBundle directory {src!r} has bundle_schema_version="
                f"{artifact_version!r}, exceeds reader v{BUNDLE_SCHEMA_VERSION}. "
                "Upgrade `geocatalog` to read this bundle."
            )
        try:
            target_crs = pyproj.CRS.from_user_input(meta["target_crs"])
        except (KeyError, pyproj.exceptions.CRSError) as exc:
            raise ValueError(
                f"CatalogBundle directory {src!r} `_meta.json` has no valid "
                f"`target_crs`: {exc}"
            ) from exc
        backend = meta.get("backend", "raster")
        if backend not in ("raster", "xarray", "vector"):
            raise ValueError(
                f"CatalogBundle directory {src!r} `_meta.json` has unknown "
                f"backend {backend!r}"
            )

        catalog = from_geoparquet(src / "items.parquet", kind=backend)
        # Checked for empty bundles too: a later ingest builds rows in
        # `target_crs` and concatenates them onto this table.
        if catalog.gdf.crs is not None and not pyproj.CRS.from_user_input(
            catalog.gdf.crs
        ).equals(target_crs):
            raise ValueError(
                f"CatalogBundle directory {src!r}: items.parquet CRS "
                f"{catalog.gdf.crs} does not match `_meta.json` target_crs "
                f"{target_crs.to_string()}"
            )
        queries = (
            _queries_from_parquet(src / "queries.parquet")
            if (src / "queries.parquet").exists()
            else []
        )
        matchups = (
            _matchups_from_parquet(src / "matchups.parquet")
            if (src / "matchups.parquet").exists()
            else []
        )
        bundle = cls(
            catalog,
            crs=target_crs,
            kind=backend,
            queries=queries,
            matchups=matchups,
        )
        bundle.created_at = meta.get("created_at")
        return bundle

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_items(self) -> int:
        """Number of rows in the items table."""
        return len(self.catalog)

    def queries_df(self) -> pd.DataFrame:
        """Return the queries table as a `pd.DataFrame` for ad-hoc analysis."""
        if not self.queries:
            return pd.DataFrame(
                columns=[f.name for f in dataclasses.fields(QueryRecord)]
            )
        return pd.DataFrame([dataclasses.asdict(q) for q in self.queries])

    def matchups_df(self) -> pd.DataFrame:
        """Return the matchups table as a `pd.DataFrame`."""
        from geocatalog._src.matchup.engine import MatchupRow

        if not self.matchups:
            return pd.DataFrame(
                columns=[f.name for f in dataclasses.fields(MatchupRow)]
            )
        # geometry_intersect is a shapely geometry — serialize to WKT
        # for the dataframe view so the DataFrame is JSON/Parquet-friendly.
        rows = []
        for m in self.matchups:
            d = dataclasses.asdict(m)
            d["geometry_intersect"] = m.geometry_intersect.wkt
            rows.append(d)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# queries.parquet  /  matchups.parquet  serialization helpers
# ---------------------------------------------------------------------------


def _existing_created_at(dest: Path) -> str | None:
    """``created_at`` from a bundle already saved at ``dest``, if any."""
    try:
        value = json.loads((dest / "_meta.json").read_text()).get("created_at")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def _queries_to_parquet(queries: list[QueryRecord], path: Path) -> None:
    """Write a `QueryRecord` list as flat parquet (no geometry)."""
    df = pd.DataFrame([dataclasses.asdict(q) for q in queries])
    df.to_parquet(path)


def _queries_from_parquet(path: Path) -> list[QueryRecord]:
    """Read back a list of `QueryRecord` from a flat parquet."""
    df = pd.read_parquet(path)
    out = []
    for _, row in df.iterrows():
        out.append(
            QueryRecord(
                query_id=str(row["query_id"]),
                source=str(row["source"]),
                collection=(
                    None if pd.isna(row["collection"]) else str(row["collection"])
                ),
                bounds_wkt=str(row["bounds_wkt"]),
                time_start=(
                    None
                    if pd.isna(row["time_start"])
                    else pd.Timestamp(row["time_start"])
                ),
                time_end=(
                    None if pd.isna(row["time_end"]) else pd.Timestamp(row["time_end"])
                ),
                filters_json=str(row["filters_json"])
                if not pd.isna(row["filters_json"])
                else "",
                created_at=pd.Timestamp(row["created_at"]).to_pydatetime(),
                n_returned=int(row["n_returned"]),
                tag=(None if pd.isna(row.get("tag")) else str(row["tag"])),
                notes=(None if pd.isna(row.get("notes")) else str(row["notes"])),
            )
        )
    return out


def _publish_directory(
    staging: Path, dest: Path, *, owned: tuple[str, ...], token: str
) -> None:
    """Make ``staging`` the new ``dest`` in one rename.

    The previous ``dest`` is set aside under a hidden name first and
    restored if the swap fails; entries in it that the bundle does not
    own (``owned``) are then moved into the new directory, and the rest
    of it is deleted. Between the two renames ``dest`` briefly does not
    exist — a reader gets "not found", never a mix of generations.
    """
    if not dest.exists():
        os.replace(staging, dest)
        return
    previous = dest.with_name(f".{dest.name}.{token}.old")
    os.replace(dest, previous)
    try:
        os.replace(staging, dest)
    except BaseException:
        os.replace(previous, dest)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for entry in previous.iterdir():
        if entry.name not in owned:
            os.replace(entry, dest / entry.name)
    shutil.rmtree(previous, ignore_errors=True)


def _matchups_to_parquet(matchups: list[MatchupRow], path: Path) -> None:
    """Write a `MatchupRow` list as parquet (geometry → WKT column)."""
    rows = []
    for m in matchups:
        d = dataclasses.asdict(m)
        # Geometry → WKT; tolerance → JSON. Tuples → lists for
        # arrow.
        d["geometry_intersect_wkt"] = m.geometry_intersect.wkt
        d.pop("geometry_intersect")
        d["member_ids"] = list(m.member_ids)
        d["member_sources"] = list(m.member_sources)
        d["member_collections"] = list(m.member_collections)
        d["member_roles"] = list(m.member_roles)
        d["time_offset_sec"] = list(m.time_offset_sec)
        d["tolerance_json"] = json.dumps(dict(m.tolerance), default=str)
        d.pop("tolerance")
        rows.append(d)
    pd.DataFrame(rows).to_parquet(path)


def _matchups_from_parquet(path: Path) -> list[MatchupRow]:
    """Read back a list of `MatchupRow` from parquet."""
    from geocatalog._src.matchup.engine import MatchupRow

    df = pd.read_parquet(path)
    out = []
    for _, row in df.iterrows():
        geom = _wkt_to_geometry(row["geometry_intersect_wkt"])
        tolerance_raw = row.get("tolerance_json", "")
        tolerance = (
            json.loads(tolerance_raw)
            if isinstance(tolerance_raw, str) and tolerance_raw
            else {}
        )
        out.append(
            MatchupRow(
                matchup_id=str(row["matchup_id"]),
                strategy=str(row["strategy"]),
                member_ids=tuple(row["member_ids"]),
                member_sources=tuple(row["member_sources"]),
                member_roles=tuple(row["member_roles"]),
                geometry_intersect=geom,
                time_reference=pd.Timestamp(row["time_reference"]).to_pydatetime(),
                time_offset_sec=tuple(float(x) for x in row["time_offset_sec"]),
                tolerance=tolerance,
                query_set=(
                    None if pd.isna(row.get("query_set")) else str(row["query_set"])
                ),
                member_collections=_member_collections(row),
            )
        )
    return out


def _member_collections(row: pd.Series) -> tuple[str, ...]:
    """``member_collections`` of a persisted row; ``()`` for older tables."""
    value = row.get("member_collections")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    return tuple(str(c) for c in value)


def _wkt_to_geometry(wkt: str) -> BaseGeometry:
    """Parse a WKT string back into a shapely geometry."""
    from shapely import wkt as _wkt_mod

    # Tolerate the helper being passed a non-WKT GeoJSON fallback —
    # `shape` handles dict-shaped inputs that pandas might unpack.
    if isinstance(wkt, dict):
        return shape(wkt)
    return _wkt_mod.loads(wkt)
