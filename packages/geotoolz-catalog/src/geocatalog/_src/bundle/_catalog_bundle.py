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
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import pandas as pd
import pyproj
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from geocatalog._src.memory import InMemoryGeoCatalog


if TYPE_CHECKING:
    from geocatalog._src.matchup.engine import MatchupRow
    from geocatalog._src.sources._base import Source, SourceRow


_BACKEND_T = Literal["raster", "xarray", "vector"]


# The bundle's own schema version, distinct from the catalog's
# `_schema_version` column. Bump on changes to the bundle directory
# layout or the queries/matchups parquet schemas; carried in `_meta.json`.
BUNDLE_SCHEMA_VERSION: int = 1


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


def source_row_to_gdf_row(
    row: SourceRow,
    *,
    target_crs: pyproj.CRS,
    primary_asset: str | None = None,
) -> dict[str, Any]:
    """Map a `SourceRow` to a flat dict suitable for a `GeoDataFrame` row.

    Output keys:
    * ``geometry``: the footprint, reprojected from EPSG:4326 to
      ``target_crs`` if necessary.
    * ``start_time`` / ``end_time``: pulled from the interval.
    * ``filepath``: chosen asset URL (see ``primary_asset`` resolution).
    * ``id``, ``source``, ``collection``: as-is from the SourceRow.
    * ``assets``, ``properties``, ``provenance``: JSON-encoded
      dicts (Parquet can't store nested dicts directly without
      schema acrobatics; JSON-as-string is the boring-but-robust path).

    Args:
        row: The source row to convert.
        target_crs: CRS to reproject the geometry into. SourceRow
            geometries come in EPSG:4326 by convention.
        primary_asset: Asset key to promote to ``filepath``. If
            ``None``, the first key in ``row.assets`` is used (Python
            dicts preserve insertion order). If the assets dict is
            empty, ``filepath`` is an empty string.
    """
    # Reproject if needed. The shapely geometry doesn't carry its
    # CRS — we use `target_crs` as the authoritative target and assume
    # `SourceRow.geometry` is in EPSG:4326 (the `Source` Protocol
    # convention).
    src_crs = pyproj.CRS.from_epsg(4326)
    dst_crs = pyproj.CRS.from_user_input(target_crs)
    geometry = row.geometry
    if not src_crs.equals(dst_crs):
        from shapely.ops import transform as shapely_transform

        transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        geometry = shapely_transform(transformer.transform, row.geometry)

    # Pick a filepath. For STAC items the "first asset" convention
    # is usually a sensible default (it's typically the lowest-res
    # overview or the canonical band); users who care can pass
    # `primary_asset` explicitly.
    filepath = ""
    if row.assets:
        if primary_asset is not None and primary_asset in row.assets:
            filepath = row.assets[primary_asset]
        else:
            filepath = next(iter(row.assets.values()))

    return {
        "geometry": geometry,
        "start_time": pd.Timestamp(row.interval.left),
        "end_time": pd.Timestamp(row.interval.right),
        "filepath": filepath,
        "id": row.id,
        "source": row.source,
        "collection": row.collection,
        "assets": json.dumps(dict(row.assets)),
        "properties": json.dumps(dict(row.properties), default=str),
        "provenance": json.dumps(dict(row.provenance), default=str),
    }


class CatalogBundle:
    """Wraps an `InMemoryGeoCatalog` + queries + matchups + persistence.

    See module docstring for the directory layout and lifecycle.

    Attributes:
        catalog: The items table as an `InMemoryGeoCatalog`. Updated
            in place when `ingest()` adds new rows.
        target_crs: Authoritative CRS for the items table's geometry
            column. Set at construction; reprojection happens on
            ingest, never on lookup.
        backend: Backend tag forwarded to the catalog.
        queries: In-memory list of `QueryRecord`s. Persisted to
            ``queries.parquet`` on `to_directory`.
        matchups: In-memory list of `MatchupRow`s. Persisted to
            ``matchups.parquet``.
    """

    def __init__(
        self,
        catalog: InMemoryGeoCatalog,
        *,
        target_crs: pyproj.CRS,
        backend: _BACKEND_T = "raster",
        queries: list[QueryRecord] | None = None,
        matchups: list[MatchupRow] | None = None,
    ) -> None:
        self.catalog = catalog
        self.target_crs = pyproj.CRS.from_user_input(target_crs)
        self.backend = backend
        self.queries: list[QueryRecord] = list(queries) if queries else []
        self.matchups: list[MatchupRow] = list(matchups) if matchups else []

    @classmethod
    def empty(
        cls,
        *,
        target_crs: str | pyproj.CRS = "EPSG:4326",
        backend: _BACKEND_T = "raster",
    ) -> CatalogBundle:
        """Create a fresh bundle with an empty items table.

        Use this as the starting point when ingesting from a `Source`:

            >>> bundle = CatalogBundle.empty(target_crs="EPSG:4326")
            >>> bundle.ingest(STACSource.planetary_computer(), ...)
            >>> bundle.to_directory("my_catalog/")
        """
        dst_crs = pyproj.CRS.from_user_input(target_crs)
        empty_gdf = gpd.GeoDataFrame(
            {
                "filepath": pd.Series(dtype="object"),
                "id": pd.Series(dtype="object"),
                "source": pd.Series(dtype="object"),
                "collection": pd.Series(dtype="object"),
                "assets": pd.Series(dtype="object"),
                "properties": pd.Series(dtype="object"),
                "provenance": pd.Series(dtype="object"),
            },
            geometry=gpd.GeoSeries([], crs=dst_crs),
        )
        empty_gdf.index = pd.IntervalIndex.from_arrays(
            pd.to_datetime([]),
            pd.to_datetime([]),
            closed="both",
            name="datetime",
        )
        catalog = InMemoryGeoCatalog(empty_gdf, backend=backend)
        return cls(catalog, target_crs=dst_crs, backend=backend)

    @classmethod
    def from_catalog(
        cls,
        catalog: InMemoryGeoCatalog,
        *,
        backend: _BACKEND_T | None = None,
    ) -> CatalogBundle:
        """Wrap an already-built `InMemoryGeoCatalog`.

        Useful when the user constructed a catalog via
        ``build_raster_catalog`` / ``build_vector_catalog`` and now
        wants to add queries/matchups state.
        """
        bk: _BACKEND_T = backend if backend is not None else catalog.backend
        return cls(
            catalog,
            target_crs=pyproj.CRS.from_user_input(catalog.gdf.crs),
            backend=bk,
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
            limit: Cap on the number of rows returned.
            primary_asset: Asset key to promote to the catalog's
                ``filepath`` column. ``None`` uses the first asset
                key (dict insertion order).
            tag: User label persisted in ``QueryRecord.tag`` and
                propagated to every row via
                ``provenance['query_tag']``.
            notes: Free-form notes recorded in ``QueryRecord.notes``.

        Returns:
            The query_id (uuid4 hex) of this ingest call.
        """
        from shapely.geometry import box as _shapely_box

        query_id = uuid.uuid4().hex
        created_at = datetime.now(tz=UTC)
        new_rows: list[dict[str, Any]] = []
        n = 0
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
            stamped = dataclasses.replace(row, provenance=prov)
            new_rows.append(
                source_row_to_gdf_row(
                    stamped,
                    target_crs=self.target_crs,
                    primary_asset=primary_asset,
                )
            )
            n += 1

        if new_rows:
            new_gdf = gpd.GeoDataFrame(new_rows, crs=self.target_crs)
            new_gdf.index = pd.IntervalIndex.from_arrays(
                new_gdf.pop("start_time"),
                new_gdf.pop("end_time"),
                closed="both",
                name="datetime",
            )
            merged = pd.concat([self.catalog.gdf, new_gdf], axis=0)
            self.catalog = InMemoryGeoCatalog(
                gpd.GeoDataFrame(merged, crs=self.target_crs),
                backend=self.backend,
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

        Returns:
            How many rows were added.
        """
        added = 0
        for row in rows:
            mr = dataclasses.replace(row, query_set=tag) if tag is not None else row
            self.matchups.append(mr)
            added += 1
        return added

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
        """
        from geocatalog._src.parquet import to_geoparquet

        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        to_geoparquet(self.catalog, dest / "items.parquet")
        # Sidecar tables: write when non-empty, otherwise delete any
        # leftover from a previous write so the on-disk state
        # matches the in-memory contract (empty = omitted).
        if self.queries:
            _queries_to_parquet(self.queries, dest / "queries.parquet")
        else:
            (dest / "queries.parquet").unlink(missing_ok=True)
        if self.matchups:
            _matchups_to_parquet(self.matchups, dest / "matchups.parquet")
        else:
            (dest / "matchups.parquet").unlink(missing_ok=True)
        meta = {
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "target_crs": self.target_crs.to_string(),
            "backend": self.backend,
            "created_at": datetime.now(tz=UTC).isoformat(),
        }
        (dest / "_meta.json").write_text(json.dumps(meta, indent=2))

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
        target_crs = pyproj.CRS.from_user_input(meta["target_crs"])
        backend: _BACKEND_T = meta.get("backend", "raster")

        catalog = from_geoparquet(src / "items.parquet")
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
        return cls(
            catalog,
            target_crs=target_crs,
            backend=backend,
            queries=queries,
            matchups=matchups,
        )

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
            )
        )
    return out


def _wkt_to_geometry(wkt: str) -> BaseGeometry:
    """Parse a WKT string back into a shapely geometry."""
    from shapely import wkt as _wkt_mod

    # Tolerate the helper being passed a non-WKT GeoJSON fallback —
    # `shape` handles dict-shaped inputs that pandas might unpack.
    if isinstance(wkt, dict):
        return shape(wkt)
    return _wkt_mod.loads(wkt)
