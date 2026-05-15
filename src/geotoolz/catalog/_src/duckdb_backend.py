"""`DuckDBGeoCatalog` — Phase 2 SQL-backed catalog over GeoParquet.

The DuckDB backend swaps Phase 1's in-RAM `GeoDataFrame` for a lazy SQL
relation on top of a GeoParquet artifact (a single file, a directory of
shards, or an `httpfs`-readable URI). The Phase 1 Protocol surface
(``query`` / ``intersect`` / ``union`` / ``iter_slices``) is preserved —
loaders, samplers, and the `geotoolz.patch` bridge work against either
backend without branching.

Why DuckDB:

- **Predicate pushdown** via GeoParquet 1.1 bbox covering struct — a
  small-AOI query on a 10⁶-row catalog reads ~10⁵ rows of bbox metadata,
  not 10⁶ WKB geometries.
- **Parallel spatial joins** via the `spatial` extension's R-tree probe,
  beating `gpd.overlay` by 2-3 orders of magnitude at scale.
- **Portable artifact** — the catalog *is* the GeoParquet file; share,
  version, hash, sign.
- **httpfs** transparently reads S3 / GCS / Azure / HuggingFace.

See ``research_journal_v2/notes/geotoolz/plans/geodatabase/geoduckdb.md``
for the design.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
import shapely.geometry


if TYPE_CHECKING:
    import duckdb as duckdb_mod

from geotoolz.catalog._src.base import CatalogRow
from geotoolz.catalog._src.memory import InMemoryGeoCatalog, _coerce_interval
from geotoolz.types import GeoSlice


# DuckDB is the optional dep for this backend. The module loader inside
# `geotoolz.catalog.__getattr__` raises a friendly ImportError if the
# `[duckdb]` extra is missing, so we don't need to repeat that message in
# every public function.
try:
    import duckdb
except ImportError:  # pragma: no cover - exercised via the [duckdb] extra
    duckdb = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


_BACKEND_T = Literal["raster", "xarray", "vector"]


def _require_duckdb() -> Any:
    """Return the `duckdb` module or raise a friendly ImportError."""
    if duckdb is None:
        raise ImportError(
            "DuckDBGeoCatalog requires the [duckdb] extra; install via "
            "`pip install 'geotoolz[duckdb]'`."
        )
    return duckdb


def _ensure_spatial(con: duckdb_mod.DuckDBPyConnection) -> None:
    """Install + load the `spatial` extension on a connection.

    Idempotent — DuckDB no-ops on a second LOAD. Pulled into a helper so
    every code path that constructs a `DuckDBGeoCatalog` goes through the
    same setup.
    """
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")


class DuckDBGeoCatalog:
    """Lazy, SQL-backed catalog over a GeoParquet artifact.

    Holds a DuckDB *relation* — a query plan, not materialised rows.
    `query` / `intersect` / `union` return new relations; nothing
    executes until ``iter_rows`` / ``materialize`` / ``gdf`` is touched.

    The class implements the same `GeoCatalog` Protocol as
    `InMemoryGeoCatalog`. Loaders that take ``catalog.gdf`` work but
    materialise the relation on access — explicit callers should prefer
    ``iter_rows`` for streaming.

    Args:
        relation: A DuckDB relation whose columns include ``filepath``,
            ``geometry`` (WKB BLOB or GEOMETRY), ``start_time``,
            ``end_time``, optionally a ``bbox`` covering struct (GeoParquet
            1.1), plus any per-backend extras.
        con: The connection that owns ``relation``. Must outlive the
            catalog.
        crs: CRS of the ``geometry`` column. The catalog stores it once
            (canonical-CRS convention) — per-row CRS is not supported by
            this backend.
        backend: ``"raster"`` / ``"xarray"`` / ``"vector"``; drives
            loader dispatch.

    Attributes:
        relation: The underlying DuckDB relation; escape hatch for SQL
            power users.
        con: The owning connection.
        crs: ``pyproj.CRS`` for ``geometry``.
        backend: Loader dispatch tag.
    """

    def __init__(
        self,
        relation: duckdb_mod.DuckDBPyRelation,
        *,
        con: duckdb_mod.DuckDBPyConnection,
        crs: Any,
        backend: _BACKEND_T,
    ) -> None:
        _require_duckdb()
        self.relation = relation
        self.con = con
        self.crs = pyproj.CRS.from_user_input(crs)
        self.backend = backend

    # ── factories ────────────────────────────────────────────────────────

    @classmethod
    def open(
        cls,
        source: str | Path,
        *,
        backend: _BACKEND_T | None = None,
        crs: Any | None = None,
    ) -> DuckDBGeoCatalog:
        """Open a GeoParquet file (or directory of shards) lazily.

        Reads the source via DuckDB's `read_parquet`; the relation
        carries the schema but no rows are materialised until queried.
        Local paths, ``s3://``, ``gs://``, ``https://`` are all
        supported provided the `httpfs` extension is loaded (call
        ``con.execute("LOAD httpfs")`` on the returned ``.con``).

        CRS is recovered from the GeoParquet column metadata (PROJJSON).
        The backend tag is recovered from the reserved ``_backend``
        column written by `to_geoparquet`; ad-hoc parquet files lacking
        it default to ``"raster"`` unless overridden.

        Args:
            source: Path or URI. A directory or glob (``shards/*.parquet``)
                is read as one virtual table.
            backend: Loader dispatch tag override. ``None`` reads the
                ``_backend`` column, falling back to ``"raster"``.
            crs: CRS override. ``None`` reads the GeoParquet PROJJSON
                metadata; falls back to ``EPSG:4326`` if neither is
                present (noisy default rather than silent coercion).

        Returns:
            A `DuckDBGeoCatalog` over the relation.
        """
        dd = _require_duckdb()
        con = dd.connect()
        _ensure_spatial(con)
        source_str = str(source)
        if crs is None:
            crs = _read_geoparquet_crs(source, default="EPSG:4326")
        if backend is None:
            backend = _read_backend_tag(con, source_str, default="raster")
        # Parameter binding (rather than f-string interpolation) keeps
        # paths containing apostrophes — `s3://bucket/o'malley/cat.parquet`
        # or tmpdirs under a username with one — from breaking the
        # query, and avoids opening a SQL-injection surface if `source`
        # ever flows from untrusted input.
        relation = con.sql(
            "SELECT * FROM read_parquet($src)", params={"src": source_str}
        )
        return cls(relation, con=con, crs=crs, backend=backend)

    @classmethod
    def from_memory(
        cls,
        catalog: InMemoryGeoCatalog,
        *,
        con: duckdb_mod.DuckDBPyConnection | None = None,
    ) -> DuckDBGeoCatalog:
        """Register an `InMemoryGeoCatalog` as a DuckDB relation.

        Useful for upgrading a small in-RAM catalog into the SQL surface
        (e.g. to spatial-join two in-memory catalogs in parallel) without
        going through a Parquet round-trip.

        Args:
            catalog: The in-memory catalog to wrap.
            con: Optional DuckDB connection to register into. A fresh
                in-memory connection is created if omitted.

        Returns:
            A `DuckDBGeoCatalog` over a view of the same rows. The view
            holds a reference to the original gdf; mutating the gdf in
            place after this call leads to undefined behaviour.
        """
        dd = _require_duckdb()
        if con is None:
            con = dd.connect()
        _ensure_spatial(con)
        df = _gdf_to_arrow_df(catalog.gdf)
        view_name = f"_geotoolz_mem_{id(catalog):x}"
        con.register(view_name, df)
        relation = con.sql(
            f"SELECT * EXCLUDE (geometry), "
            f"  ST_GeomFromWKB(geometry) AS geometry "
            f"FROM {view_name}"
        )
        return cls(relation, con=con, crs=catalog.gdf.crs, backend=catalog.backend)

    # ── lazy ↔ eager bridges ─────────────────────────────────────────────

    @cached_property
    def gdf(self) -> gpd.GeoDataFrame:
        """Materialise the relation as a `gpd.GeoDataFrame`.

        Cached — the first access executes the query; subsequent
        accesses are O(1). This is the eager bridge that lets loaders
        written against the in-memory backend work unchanged. For
        streaming/scale callers, prefer `iter_rows`.
        """
        return self.materialize().gdf

    def materialize(self) -> InMemoryGeoCatalog:
        """Execute the relation and wrap the result as an `InMemoryGeoCatalog`.

        Decodes WKB to shapely geometries and rebuilds the
        ``IntervalIndex`` from ``start_time`` / ``end_time``. Useful for
        moving the result into operations that need the full pandas /
        geopandas surface (custom merges, plotting, debugging) but
        defeats the lazy story — call after a selective ``query``.

        Returns:
            An `InMemoryGeoCatalog` over the materialised rows, same
            CRS, same backend tag.
        """
        df = self.relation.df()
        return _df_to_inmemory(df, crs=self.crs, backend=self.backend)

    # ── Protocol surface ─────────────────────────────────────────────────

    def query(
        self,
        slice_: GeoSlice | None = None,
        *,
        bounds: tuple[float, float, float, float] | None = None,
        crs: Any | None = None,
        time: tuple[Any, Any] | pd.Interval | None = None,
    ) -> DuckDBGeoCatalog:
        """SQL spatial + temporal filter — lazy, returns a new relation.

        Builds a ``WHERE`` clause that DuckDB can push down to the
        Parquet reader (bbox-column predicates skip whole row-groups).
        Mirrors `InMemoryGeoCatalog.query`; AOI bounds in a different
        CRS are reprojected internally before being interpolated into
        the SQL.

        Args:
            slice_: A `GeoSlice` whose bbox + interval drive the filter.
                Mutually exclusive with the keyword args.
            bounds: ``(xmin, ymin, xmax, ymax)`` in ``crs`` units.
            crs: CRS of ``bounds``; defaults to the catalog CRS.
            time: Either a ``(start, end)`` pair (`pd.Timestamp`-like)
                or a `pd.Interval`. ``None`` skips the temporal filter.

        Returns:
            A new `DuckDBGeoCatalog` over the filtered relation. May be
            empty; ``len()`` triggers a count query.

        Raises:
            TypeError: If both ``slice_`` and any of (``bounds``,
                ``time``) are passed.
        """
        if slice_ is not None and (bounds is not None or time is not None):
            raise TypeError("query: pass either slice_ or (bounds + time), not both")
        if slice_ is not None:
            q_bounds = slice_.bounds
            q_crs = slice_.crs
            q_interval = slice_.interval
        else:
            if bounds is None and time is None:
                return self
            q_bounds = bounds
            q_crs = crs
            q_interval = _coerce_interval(time) if time is not None else None

        where: list[str] = []
        if q_bounds is not None:
            xmin, ymin, xmax, ymax = _reproject_bounds(q_bounds, q_crs, self.crs)
            where.append(
                f"ST_Intersects(geometry, "
                f"ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}))"
            )
        if q_interval is not None:
            t_lo = pd.Timestamp(q_interval.left).isoformat()
            t_hi = pd.Timestamp(q_interval.right).isoformat()
            where.append(f"end_time >= TIMESTAMP '{t_lo}'")
            where.append(f"start_time <= TIMESTAMP '{t_hi}'")

        if not where:
            return self
        clause = " AND ".join(where)
        filtered = self.relation.filter(clause)
        return DuckDBGeoCatalog(
            filtered, con=self.con, crs=self.crs, backend=self.backend
        )

    def intersect(
        self,
        other: DuckDBGeoCatalog | InMemoryGeoCatalog,
        *,
        spatial_only: bool = False,
    ) -> DuckDBGeoCatalog:
        """Cross-catalog AND via SQL spatial join.

        Translates to ``SELECT ... FROM self JOIN other ON
        ST_Intersects(self.geometry, other.geometry) [AND interval-overlap]``.
        The DuckDB planner builds an R-tree on the smaller side and
        probes with the larger; cost scales as ``O(n log n + m log n)``
        versus `gpd.overlay`'s near-quadratic blow-up at scale.

        Args:
            other: Another catalog. An `InMemoryGeoCatalog` is registered
                via `from_memory` before joining. CRS is reprojected
                into ``self.crs`` if the two differ — done at registration
                time, not in SQL, so the join itself runs in one CRS.
            spatial_only: If True, drop the temporal predicate.

        Returns:
            A new `DuckDBGeoCatalog` over the joined relation. Footprints
            are clipped to ``ST_Intersection``; intervals to the
            tightest overlap of left and right (or the left interval
            when ``spatial_only=True``).
        """
        other_duck = _coerce_to_duckdb(other, con=self.con, target_crs=self.crs)

        left_name = f"_geotoolz_left_{id(self):x}"
        right_name = f"_geotoolz_right_{id(other_duck):x}"
        # `relation.create_view` registers via SQL so the planner sees
        # GEOMETRY-typed columns directly (Arrow round-trips would
        # drop the GEOMETRY type back to BLOB and force a re-decode).
        self.relation.create_view(left_name, replace=True)
        other_duck.relation.create_view(right_name, replace=True)

        temporal = (
            ""
            if spatial_only
            else (" AND L.end_time >= R.start_time AND L.start_time <= R.end_time")
        )
        time_select = (
            "L.start_time AS start_time, L.end_time AS end_time"
            if spatial_only
            else (
                "GREATEST(L.start_time, R.start_time) AS start_time, "
                "LEAST   (L.end_time,   R.end_time)   AS end_time"
            )
        )
        sql = f"""
            SELECT
                L.filepath AS filepath,
                ST_Intersection(L.geometry, R.geometry) AS geometry,
                {time_select}
            FROM {left_name} AS L
            JOIN {right_name} AS R
              ON ST_Intersects(L.geometry, R.geometry)
                 {temporal}
        """
        joined = self.con.sql(sql)
        return DuckDBGeoCatalog(
            joined, con=self.con, crs=self.crs, backend=self.backend
        )

    def union(self, other: DuckDBGeoCatalog | InMemoryGeoCatalog) -> DuckDBGeoCatalog:
        """Cross-catalog OR via SQL ``UNION ALL``.

        ``self``'s CRS and backend tag win. ``other`` is reprojected
        into ``self.crs`` if needed before the union. Schemas must be
        compatible for the columns both sides share; extra columns on
        one side become NULL on the other.

        Args:
            other: Catalog to concatenate. Reprojected if needed.

        Returns:
            A new `DuckDBGeoCatalog` over the union relation.
        """
        other_duck = _coerce_to_duckdb(other, con=self.con, target_crs=self.crs)
        left_name = f"_geotoolz_unionL_{id(self):x}"
        right_name = f"_geotoolz_unionR_{id(other_duck):x}"
        self.relation.create_view(left_name, replace=True)
        other_duck.relation.create_view(right_name, replace=True)
        # `UNION ALL BY NAME` matches columns by name and fills missing
        # ones with NULL on the other side — that's what preserves
        # backend-specific columns (`time_var`, `data_vars`, `layer`)
        # rather than dropping them like a positional `UNION ALL`
        # would. Without this, downstream xarray/vector loaders break
        # after a union because the metadata columns vanish.
        sql = f"""
            SELECT * FROM {left_name}
            UNION ALL BY NAME
            SELECT * FROM {right_name}
        """
        unioned = self.con.sql(sql)
        return DuckDBGeoCatalog(
            unioned, con=self.con, crs=self.crs, backend=self.backend
        )

    def iter_rows(self, *, batch_size: int = 1024) -> Iterator[CatalogRow]:
        """Stream rows as `CatalogRow` instances.

        Materialises the relation once via `.df()`, then yields rows
        one at a time. ``batch_size`` is currently advisory — DuckDB's
        Python API materialises in one chunk; we may switch to
        `.fetchmany()` if streaming benchmarks demand it.

        Args:
            batch_size: Advisory batch size. Currently unused.

        Yields:
            `CatalogRow` with ``geometry`` decoded from WKB.
        """
        del batch_size
        df = self.relation.df()
        if len(df) == 0:
            return
        geoms = _decode_geometry_column(df["geometry"])
        starts = pd.to_datetime(df["start_time"])
        ends = pd.to_datetime(df["end_time"])
        reserved = {"geometry", "filepath", "start_time", "end_time", "bbox"}
        # `_backend`, `_schema_version` and any other underscore-prefixed
        # column belong to the on-disk schema, not the user-visible row
        # metadata. Filtering them keeps `extras` clean for downstream
        # loaders that introspect it.
        extra_cols = [
            c for c in df.columns if c not in reserved and not c.startswith("_")
        ]
        for i in range(len(df)):
            extras = {c: df[c].iloc[i] for c in extra_cols}
            yield CatalogRow(
                filepath=str(df["filepath"].iloc[i]),
                geometry=geoms[i],
                interval=pd.Interval(starts.iloc[i], ends.iloc[i], closed="both"),
                crs=self.crs,
                extras=extras,
            )

    def iter_slices(self, *, resolution: tuple[float, float]) -> Iterator[GeoSlice]:
        """Yield one `GeoSlice` per row at the given target resolution.

        Lazy in shape but materialising in practice (delegates to
        `iter_rows`). For very large catalogs, prefer to ``query`` down
        first.

        Args:
            resolution: ``(x_res, y_res)`` in CRS units, baked into
                every emitted slice.

        Yields:
            `GeoSlice` instances in catalog row order.
        """
        for row in self.iter_rows():
            yield GeoSlice(
                bounds=tuple(row.geometry.bounds),  # type: ignore[arg-type]
                interval=row.interval,
                resolution=resolution,
                crs=row.crs,
            )

    # ── properties + persistence ─────────────────────────────────────────

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        """Union bbox over the relation — one SQL aggregate, not a scan.

        Returns:
            ``(xmin, ymin, xmax, ymax)`` in catalog-CRS units. Four
            NaNs for an empty catalog.
        """
        if len(self) == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        df = self.relation.aggregate(
            "MIN(ST_XMin(geometry)) AS xmin, "
            "MIN(ST_YMin(geometry)) AS ymin, "
            "MAX(ST_XMax(geometry)) AS xmax, "
            "MAX(ST_YMax(geometry)) AS ymax"
        ).df()
        return (
            float(df["xmin"].iloc[0]),
            float(df["ymin"].iloc[0]),
            float(df["xmax"].iloc[0]),
            float(df["ymax"].iloc[0]),
        )

    @property
    def temporal_extent(self) -> pd.Interval:
        """Tightest interval over the relation — one SQL aggregate.

        Returns:
            ``pd.Interval(min(start_time), max(end_time), closed='both')``.
            Both endpoints are ``pd.NaT`` for an empty catalog.
        """
        if len(self) == 0:
            return pd.Interval(pd.NaT, pd.NaT, closed="both")
        df = self.relation.aggregate(
            "MIN(start_time) AS tmin, MAX(end_time) AS tmax"
        ).df()
        return pd.Interval(
            pd.Timestamp(df["tmin"].iloc[0]),
            pd.Timestamp(df["tmax"].iloc[0]),
            closed="both",
        )

    def to_geoparquet(
        self,
        path: str | Path,
        *,
        write_covering_bbox: bool = True,
    ) -> None:
        """Persist the relation as a GeoParquet file.

        Materialises the relation, then routes through the existing
        `geopandas` writer so the resulting Parquet carries proper
        GeoParquet metadata (column CRS, bbox covering struct in 1.1).
        DuckDB's native GeoParquet writer is newer and less battle-tested
        — going through geopandas avoids that adoption caveat (§10.2 of
        the design plan).

        Args:
            path: Destination path.
            write_covering_bbox: Emit the per-row bbox covering struct
                that GeoParquet 1.1 readers (DuckDB, geopandas) use for
                predicate pushdown. Default True.
        """
        from geotoolz.catalog._src.parquet import to_geoparquet as _write

        _write(
            self.materialize(),
            path,
            write_covering_bbox=write_covering_bbox,
        )

    def sql(self, where: str) -> DuckDBGeoCatalog:
        """Apply an arbitrary SQL ``WHERE`` clause — escape hatch.

        Lets users write predicates the structured `query` API doesn't
        cover (``cloud_pct < 5 AND sensor = 'S2A'``). The result is
        still a `DuckDBGeoCatalog`, so further `query` / `intersect` /
        `union` calls chain off it.

        Args:
            where: The body of a SQL ``WHERE`` clause, *without* the
                ``WHERE`` keyword. Column names available are whatever
                the underlying relation has.

        Returns:
            A filtered `DuckDBGeoCatalog`.
        """
        return DuckDBGeoCatalog(
            self.relation.filter(where),
            con=self.con,
            crs=self.crs,
            backend=self.backend,
        )

    def __len__(self) -> int:
        """Number of rows — runs one COUNT(*) query."""
        df = self.relation.aggregate("COUNT(*) AS n").df()
        return int(df["n"].iloc[0])

    def __repr__(self) -> str:
        return (
            f"DuckDBGeoCatalog(backend={self.backend!r}, crs={self.crs.to_string()!r})"
        )

    def get_config(self) -> dict[str, Any]:
        """JSON-serialisable summary — backend tag, row count, CRS.

        Returns:
            ``{"backend": str, "len": int, "crs": str, "engine": "duckdb"}``.
            ``len`` triggers a count query — comparable shape to
            `InMemoryGeoCatalog.get_config` but with a SQL hop.
        """
        return {
            "backend": self.backend,
            "len": len(self),
            "crs": self.crs.to_string(),
            "engine": "duckdb",
        }


# ── helpers ──────────────────────────────────────────────────────────────


def _coerce_to_duckdb(
    other: DuckDBGeoCatalog | InMemoryGeoCatalog,
    *,
    con: duckdb_mod.DuckDBPyConnection,
    target_crs: pyproj.CRS,
) -> DuckDBGeoCatalog:
    """Pull ``other`` into a DuckDB relation on ``con`` in ``target_crs``.

    DuckDB views are connection-scoped — a relation on one connection
    can't be referenced from another. Always re-register onto ``con``
    when ``other`` carries a different connection (independently
    `open()`-ed catalogs hit this) by materialising and re-importing
    through `from_memory`.
    """
    if isinstance(other, DuckDBGeoCatalog):
        if other.con is con and other.crs == target_crs:
            return other
        # Either different connection or different CRS — materialise and
        # re-register onto our connection. Reprojection happens via
        # geopandas (PROJ-bound `ST_Transform` is slow per row).
        mem = other.materialize()
        if mem.gdf.crs != target_crs:
            mem = InMemoryGeoCatalog(mem.gdf.to_crs(target_crs), backend=mem.backend)
        return DuckDBGeoCatalog.from_memory(mem, con=con)
    if other.gdf.crs != target_crs:
        other = InMemoryGeoCatalog(other.gdf.to_crs(target_crs), backend=other.backend)
    return DuckDBGeoCatalog.from_memory(other, con=con)


def _gdf_to_arrow_df(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Project a GeoDataFrame to a vanilla DataFrame with WKB geometry.

    DuckDB's `register` consumes a `pd.DataFrame` or a `pyarrow.Table`;
    feeding it a `GeoDataFrame` works but the geometry column becomes a
    Python object column. Encoding to WKB makes the round-trip
    explicit and avoids the slow per-row WKT serialise that DuckDB
    falls back to otherwise.
    """
    df = pd.DataFrame(gdf.drop(columns=["geometry"]))
    df["geometry"] = gpd.GeoSeries(gdf.geometry).to_wkb()
    if isinstance(gdf.index, pd.IntervalIndex):
        df["start_time"] = gdf.index.left
        df["end_time"] = gdf.index.right
    if "filepath" not in df.columns:
        df["filepath"] = [str(i) for i in range(len(df))]
    return df


def _read_geoparquet_crs(source: str | Path, *, default: str) -> Any:
    """Pull the catalog CRS out of a GeoParquet file's column metadata.

    GeoParquet stores CRS as PROJJSON inside the ``geo`` key of the
    Parquet file metadata (`pyarrow` exposes it as raw bytes). This
    helper decodes it via `pyproj.CRS.from_user_input` so the DuckDB
    backend can carry the catalog CRS even though DuckDB itself ignores
    the GeoParquet column metadata. Directories / globs fall back to
    the default — we'd need to inspect one shard to know better.
    """
    import json

    import pyarrow.parquet as pq

    path = Path(source)
    if not path.is_file():
        return default
    try:
        md = pq.read_metadata(path).metadata or {}
    except Exception:
        return default
    geo = md.get(b"geo")
    if geo is None:
        return default
    try:
        geo_meta = json.loads(geo.decode())
        primary = geo_meta.get("primary_column", "geometry")
        crs_val = geo_meta.get("columns", {}).get(primary, {}).get("crs")
    except (ValueError, KeyError):
        return default
    if crs_val is None:
        return default
    return pyproj.CRS.from_user_input(crs_val)


def _read_backend_tag(
    con: duckdb_mod.DuckDBPyConnection, source: str, *, default: _BACKEND_T
) -> _BACKEND_T:
    """Recover the ``_backend`` column written by `to_geoparquet`.

    Returns the default for ad-hoc parquet files lacking the column;
    that's the case when an externally produced GeoParquet (e.g. one
    built by DuckDB or GDAL directly) is opened with this factory.

    Uses parameter binding for ``source`` so paths with apostrophes
    don't break the lookup, and narrowed exception handling so genuine
    SQL parse errors aren't silently swallowed as a missing column.
    """
    dd = _require_duckdb()
    try:
        df = con.sql(
            "SELECT _backend FROM read_parquet($src) LIMIT 1",
            params={"src": source},
        ).df()
    except dd.BinderException:
        # Missing `_backend` column — externally produced parquet.
        return default
    except dd.IOException:
        # Unreadable parquet path; caller will hit a clearer error
        # on the next read.
        return default
    if len(df) == 0 or pd.isna(df["_backend"].iloc[0]):
        return default
    tag = str(df["_backend"].iloc[0])
    if tag in ("raster", "xarray", "vector"):
        return tag  # type: ignore[return-value]
    return default


def _df_to_inmemory(
    df: pd.DataFrame,
    *,
    crs: pyproj.CRS,
    backend: _BACKEND_T,
) -> InMemoryGeoCatalog:
    """Build an `InMemoryGeoCatalog` from a DuckDB-materialised DataFrame."""
    if len(df) == 0:
        gdf = gpd.GeoDataFrame(
            {"geometry": []},
            geometry="geometry",
            crs=crs,
            index=pd.IntervalIndex.from_arrays(
                np.array([], dtype="datetime64[ns]"),
                np.array([], dtype="datetime64[ns]"),
                closed="both",
                name="datetime",
            ),
        )
        return InMemoryGeoCatalog(gdf, backend=backend)
    geom = _decode_geometry_column(df["geometry"])
    starts = pd.to_datetime(df["start_time"])
    ends = pd.to_datetime(df["end_time"])
    out = df.drop(columns=["geometry", "start_time", "end_time"]).copy()
    out["geometry"] = geom
    gdf = gpd.GeoDataFrame(out, geometry="geometry", crs=crs)
    idx = pd.IntervalIndex.from_arrays(starts, ends, closed="both", name="datetime")
    gdf = gdf.set_index(idx)
    return InMemoryGeoCatalog(gdf, backend=backend)


def _decode_geometry_column(col: pd.Series) -> list[Any]:
    """Decode a DuckDB-returned geometry column into shapely geometries.

    DuckDB's spatial extension hands `GEOMETRY` columns back to Python
    as WKB bytes (`bytes` / `bytearray`), `memoryview`, or — when the
    column was loaded from a vanilla Parquet BLOB column — `bytes`
    again. Anything already a shapely geometry passes through (the
    materialise path from `from_memory` already decoded via
    `ST_GeomFromWKB`, then DuckDB hands the result back as a WKB blob
    too — we re-decode).
    """
    out: list[Any] = []
    for val in col:
        if val is None:
            out.append(None)
            continue
        if hasattr(val, "is_empty") and hasattr(val, "geom_type"):
            out.append(val)
            continue
        if isinstance(val, (bytes, bytearray, memoryview)):
            out.append(shapely.from_wkb(bytes(val)))
            continue
        if isinstance(val, str):
            # WKT fallback for older DuckDB versions.
            out.append(shapely.from_wkt(val))
            continue
        # Last-ditch: try shapely's parser.
        out.append(shapely.from_wkb(val))
    return out


def _reproject_bounds(
    bounds: tuple[float, float, float, float],
    src_crs: Any | None,
    dst_crs: pyproj.CRS,
) -> tuple[float, float, float, float]:
    """Reproject AOI bounds into ``dst_crs``, no-op when CRSs match.

    Mirrors `geotoolz.catalog._src.memory._reproject_bounds` — kept
    local so the DuckDB module doesn't reach into the InMemory module's
    private surface. ``src_crs=None`` is treated as "already in
    catalog CRS"; the bbox passes through unchanged.
    """
    if src_crs is None:
        return bounds
    src = pyproj.CRS.from_user_input(src_crs)
    if src == dst_crs:
        return bounds
    transformer = pyproj.Transformer.from_crs(src, dst_crs, always_xy=True)
    return transformer.transform_bounds(*bounds)  # type: ignore[return-value]
