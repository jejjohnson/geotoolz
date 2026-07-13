"""`DuckDBGeoCatalog` — Phase 2 SQL-backed catalog over GeoParquet.

The DuckDB backend swaps Phase 1's in-RAM `GeoDataFrame` for a lazy SQL
relation on top of a GeoParquet artifact (a single file, a directory of
shards, or a DuckDB-readable URI). The Phase 1 Protocol surface
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
- **Remote extensions** transparently read S3 / GCS / Azure / HuggingFace.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
import shapely.geometry
from loguru import logger as log


if TYPE_CHECKING:
    import duckdb as duckdb_mod

from geocatalog._src.base import RESERVED_COLUMNS, CatalogMetadataError, CatalogRow
from geocatalog._src.geoslice import GeoSlice
from geocatalog._src.memory import (
    InMemoryGeoCatalog,
    _coerce_interval,
    _reproject_bounds,
)
from geocatalog._src.retry import retry_transient_io


# DuckDB is the optional dep for this backend. The module loader inside
# `geocatalog.__getattr__` raises a friendly ImportError if the
# `[duckdb]` extra is missing, so we don't need to repeat that message in
# every public function.
try:
    import duckdb
except ImportError:  # pragma: no cover - exercised via the [duckdb] extra
    duckdb = None  # type: ignore[assignment]


_BACKEND_T = Literal["raster", "xarray", "vector"]
_BACKEND_TAG_CACHE: WeakKeyDictionary[
    duckdb_mod.DuckDBPyConnection, dict[str, _BACKEND_T]
] = WeakKeyDictionary()


def _require_duckdb() -> Any:
    """Return the `duckdb` module or raise a friendly ImportError."""
    if duckdb is None:
        raise ImportError(
            "DuckDBGeoCatalog requires the [duckdb] extra; install via "
            "`pip install 'geocatalog[duckdb]'`."
        )
    return duckdb


def _ensure_spatial(con: duckdb_mod.DuckDBPyConnection) -> None:
    """Install + load the `spatial` extension on a connection.

    Idempotent — DuckDB no-ops on a second LOAD. If the extension cannot
    be installed (for example in an offline environment), opening a
    catalog still succeeds for non-spatial operations such as ``len()``
    and partition-column filters; later spatial SQL calls will raise the
    DuckDB error that names the missing extension.
    """
    dd = _require_duckdb()
    try:
        con.execute("LOAD spatial")
    except dd.Error:
        try:
            con.execute("INSTALL spatial")
            con.execute("LOAD spatial")
        except dd.Error as exc:
            log.warning(
                "DuckDB spatial extension is unavailable; spatial SQL operations "
                "will fail until the extension can be installed and loaded "
                "(check network access and extension-directory permissions): {}",
                exc,
            )
            return


def _scheme(source: str | Path) -> str | None:
    """Return the lowercase URI scheme for ``source``, or ``None`` for paths.

    Only strings containing ``://`` are treated as URIs. Shell-style
    ``name:foo.parquet`` (e.g. ``s3:catalog.parquet``) is a *local path*
    in POSIX semantics, not an S3 URI — `urlsplit` would still parse a
    ``scheme`` out of it, so we gate on the ``://`` separator first to
    avoid triggering remote-extension installs for local files.

    Examples:
        >>> _scheme("s3://bucket/cat.parquet")
        's3'
        >>> _scheme("s3:catalog.parquet")
        None
        >>> _scheme(Path("cat.parquet"))
        None
        >>> _scheme("C:/data/cat.parquet")
        None
    """
    if isinstance(source, Path):
        return None
    if "://" not in source:
        return None
    if (
        len(source) >= 3
        and source[0].isalpha()
        and source[1] == ":"
        and source[2] in ("/", "\\")
    ):
        return None
    scheme = urlsplit(source).scheme
    return scheme.lower() if scheme else None


class DuckDBGeoCatalog:
    """Lazy, SQL-backed catalog over a GeoParquet artifact.

    Holds a DuckDB *relation* — a query plan, not materialised rows.
    `query` / `intersect` / `union` return new relations; nothing
    executes until ``iter_rows`` / ``materialize`` / ``gdf`` is touched.

    The class implements the same `GeoCatalog` Protocol as
    `InMemoryGeoCatalog`. Loaders that take ``catalog.gdf`` work but
    materialise the relation on access — explicit callers should prefer
    ``iter_rows`` for streaming.

    Connection ownership is explicit: catalogs returned by `open` own
    their DuckDB connection and close it from `close` or context-manager
    exit. Catalogs derived from `query` / `intersect` / `union` share
    their parent's connection and ``close()`` is a no-op for them. Even
    a no-filter `query` returns a non-owning wrapper so calling ``close``
    on the result cannot close the parent catalog.

    Derived catalogs additionally hold a strong reference back to the
    originating *owning* catalog via ``_owner``. This keeps the fluent
    chain ``DuckDBGeoCatalog.open(path).query(...)`` safe: the
    transitively-referenced owner stays alive (and its connection
    open) as long as any derived catalog in the chain is reachable.
    When a derived catalog is used as a context manager
    (``with cat.query(...) as out:``), its ``__exit__`` closes the
    owning catalog so the underlying connection is released on block
    exit. Calling ``close()`` directly on a derived catalog remains a
    no-op — only ``__exit__`` (or closing the owner directly) tears
    down the shared connection.

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
        _owns_con: Whether this catalog is responsible for closing
            ``con``.

    Attributes:
        relation: The underlying DuckDB relation; escape hatch for SQL
            power users.
        con: The associated connection. Set to ``None`` after an owning
            catalog is closed.
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
        _owns_con: bool = False,
    ) -> None:
        _require_duckdb()
        self.relation = relation
        self.con: duckdb_mod.DuckDBPyConnection | None = con
        self.crs = pyproj.CRS.from_user_input(crs)
        self.backend = backend
        self._owns_con = _owns_con
        # Strong ref back to the *owning* catalog when this instance is a
        # derivation (`query` / `intersect` / `union` / `sql`). Keeps the
        # owner — and therefore the underlying DuckDB connection — alive
        # for the lifetime of any derived catalog in the chain, so the
        # fluent pattern `DuckDBGeoCatalog.open(p).query(...)` doesn't
        # drop the only reference to the owner mid-expression and leak
        # the connection in long-lived processes.
        self._owner: DuckDBGeoCatalog | None = None

    def _require_open_con(self) -> duckdb_mod.DuckDBPyConnection:
        # Derived catalogs share the owner's connection — if the owner
        # was closed, our `self.con` still points at the (now-closed)
        # DuckDBPyConnection object, so check the owner first.
        if self._owner is not None and self._owner.con is None:
            dd = _require_duckdb()
            raise dd.ConnectionException(
                "DuckDBGeoCatalog connection has already been closed. Open a new "
                "catalog, or keep the parent catalog open when using derived catalogs."
            )
        con = self.con
        if con is None:
            dd = _require_duckdb()
            raise dd.ConnectionException(
                "DuckDBGeoCatalog connection has already been closed. Open a new "
                "catalog, or keep the parent catalog open when using derived catalogs."
            )
        return con

    def _derive(self, relation: duckdb_mod.DuckDBPyRelation) -> DuckDBGeoCatalog:
        derived = DuckDBGeoCatalog(
            relation,
            con=self._require_open_con(),
            crs=self.crs,
            backend=self.backend,
        )
        # Anchor the derivation chain at the originating owning catalog.
        # If `self` owns its connection, `self` is the owner; otherwise
        # `self` itself is a derivation and we inherit its owner. This
        # gives every derived instance a direct strong ref to the root
        # owner — `A.query().intersect()` keeps `A` reachable until the
        # outermost derived catalog is dropped.
        derived._owner = self if self._owns_con else self._owner
        return derived

    def close(self) -> None:
        """Close this catalog's DuckDB connection if it owns it.

        No-op for derived catalogs (from `query` / `intersect` / `union`
        / `sql`) — they share the owner's connection and closing them
        would invalidate sibling derivations. Use the *owning* catalog's
        ``close``, or a context manager on the owning instance, to tear
        down the connection. A derived catalog used as a context
        manager closes its owner on ``__exit__``.
        """
        if self._owns_con and self.con is not None:
            self.con.close()
            self.con = None

    def __enter__(self) -> DuckDBGeoCatalog:
        self._require_open_con()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        # On exit, close the owner if we have one — that's the fluent
        # `with DuckDBGeoCatalog.open(p).query(...) as cat:` pattern,
        # where the only handle the caller has is `cat` (a derivation),
        # and the underlying connection would otherwise leak. For an
        # owning catalog `_owner is None`, so we fall back to closing
        # ourselves. Direct `close()` calls on a derivation remain a
        # no-op so sibling derivations on the same owner stay valid.
        if self._owner is not None:
            self._owner.close()
        else:
            self.close()

    # ── factories ────────────────────────────────────────────────────────

    @classmethod
    def open(
        cls,
        source: str | Path,
        *,
        backend: _BACKEND_T | None = None,
        crs: Any | None = None,
        retries: int = 3,
        storage_options: dict[str, Any] | None = None,
        strict: bool = False,
    ) -> DuckDBGeoCatalog:
        """Open a GeoParquet file (or directory of shards) lazily.

        Reads the source via DuckDB's `read_parquet`; the relation
        carries the schema but no rows are materialised until queried.
        Local paths are supported directly. URI sources with ``s3://``,
        ``gs://``, ``gcs://``, ``http://``, ``https://``, ``r2://``, or
        ``hf://`` auto-load DuckDB's `httpfs` extension; ``az://`` and
        ``azure://`` auto-load DuckDB's `azure` extension.

        CRS is recovered from the GeoParquet column metadata (PROJJSON)
        for **local files only** — the metadata reader uses
        `pyarrow.parquet.read_metadata` over a local `Path`. For URI
        sources (``s3://``, ``https://``, ``hf://``, …) the auto-detect
        cannot reach the remote object and silently falls back to the
        ``EPSG:4326`` default; a `UserWarning` is emitted in that case.
        Pass ``crs=`` explicitly for remote sources to avoid the
        fallback. (Remote GeoParquet CRS introspection would need an
        fsspec/pyarrow filesystem hookup and is tracked separately.)

        The backend tag is recovered from the reserved ``_backend``
        column written by `to_geoparquet`; ad-hoc parquet files lacking
        it default to ``"raster"`` unless overridden.

        Args:
            source: Path or URI. A directory or glob (``shards/*.parquet``)
                is read as one virtual table.
            backend: Loader dispatch tag override. ``None`` reads the
                ``_backend`` column, falling back to ``"raster"``.
            crs: CRS override. ``None`` reads the GeoParquet PROJJSON
                metadata for local files; falls back to ``EPSG:4326`` if
                neither is present (noisy default rather than silent
                coercion). For URI sources, ``None`` always falls back
                to the default — pass ``crs=`` explicitly.
            retries: Number of retries for transient remote I/O failures.
                ``0`` disables retry/backoff.
            storage_options: Not supported by this backend — DuckDB reads
                URIs natively via its `httpfs` / `azure` extensions and
                does not accept an fsspec-style credential mapping. Pass
                ``None``; configure DuckDB secrets (or pre-set env vars)
                before calling `open`. Use ``engine='memory'`` if you need
                fsspec-backed reads.
            strict: If ``True``, raise `CatalogMetadataError` instead of
                falling back when the ``_backend`` column is missing /
                unreadable (and ``backend=`` was not passed) or the
                GeoParquet ``geo`` metadata is unreadable (and ``crs=``
                was not passed). Remote URIs cannot be introspected at
                all, so ``strict=True`` requires an explicit ``crs=``
                for them. Explicit ``backend=`` / ``crs=`` overrides
                bypass the corresponding check.

        Returns:
            A `DuckDBGeoCatalog` over the relation.
        """
        if storage_options is not None:
            raise ValueError(
                "DuckDBGeoCatalog does not support storage_options. Use "
                "open_catalog(source, engine='memory', storage_options=...) "
                "for fsspec-backed reads (loads the full catalog into memory), "
                "or configure DuckDB credentials directly."
            )
        dd = _require_duckdb()
        con = dd.connect()
        try:
            _ensure_spatial(con)
            source_str = _read_parquet_source(source)
            partitioned = _is_partitioned_source(source)
            scheme = _scheme(source)
            if scheme in ("s3", "gs", "gcs", "https", "http", "r2", "hf"):
                con.execute("INSTALL httpfs")
                con.execute("LOAD httpfs")
            elif scheme in ("az", "azure"):
                con.execute("INSTALL azure")
                con.execute("LOAD azure")
            if crs is None:
                if scheme is not None and strict:
                    raise CatalogMetadataError(
                        f"cannot auto-detect CRS from remote URI {source_str}: "
                        "GeoParquet metadata introspection only works for "
                        "local files. Pass crs= explicitly when strict=True."
                    )
                if scheme is not None:
                    warnings.warn(
                        f"DuckDBGeoCatalog.open({source_str!r}): cannot auto-detect "
                        "CRS from a remote URI; falling back to EPSG:4326. Pass "
                        "`crs=` explicitly for remote sources to avoid silent "
                        "default coercion.",
                        UserWarning,
                        stacklevel=2,
                    )
                crs = retry_transient_io(
                    _read_geoparquet_crs,
                    source,
                    default="EPSG:4326",
                    strict=strict,
                    retries=retries,
                )
            if backend is None:
                backend = retry_transient_io(
                    _read_backend_tag,
                    con,
                    source_str,
                    default="raster",
                    partitioned=partitioned,
                    strict=strict,
                    retries=retries,
                )
            retry_transient_io(
                _check_schema_version,
                con,
                source_str,
                partitioned=partitioned,
                retries=retries,
            )
            # Parameter binding (rather than f-string interpolation) keeps
            # paths containing apostrophes — `s3://bucket/o'malley/cat.parquet`
            # or tmpdirs under a username with one — from breaking the
            # query, and avoids opening a SQL-injection surface if `source`
            # ever flows from untrusted input.
            # `hive_partitioning` is conditional: enabling it on a single
            # file under a `key=value` directory would inject a synthetic
            # partition column into the schema.
            relation = retry_transient_io(
                con.sql,
                "SELECT * FROM read_parquet($src, hive_partitioning = $hive)",
                params={"src": source_str, "hive": partitioned},
                retries=retries,
            )
        except BaseException:
            # Setup failed (bad extension load, schema mismatch, IO error);
            # don't leak the freshly opened connection.
            con.close()
            raise
        return cls(relation, con=con, crs=crs, backend=backend, _owns_con=True)

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
                in-memory connection is created if omitted; ownership
                of an externally supplied connection stays with the
                caller (close is a no-op for it).

        Returns:
            A `DuckDBGeoCatalog` over a view of the same rows. The view
            holds a reference to the original gdf; mutating the gdf in
            place after this call leads to undefined behaviour.
        """
        dd = _require_duckdb()
        owns_con = con is None
        if con is None:
            con = dd.connect()
        try:
            _ensure_spatial(con)
            df = _gdf_to_arrow_df(catalog.gdf)
            view_name = f"_geocatalog_mem_{id(catalog):x}"
            con.register(view_name, df)
            relation = con.sql(
                f"SELECT * EXCLUDE (geometry), "
                f"  ST_GeomFromWKB(geometry) AS geometry "
                f"FROM {view_name}"
            )
        except BaseException:
            if owns_con:
                con.close()
            raise
        return cls(
            relation,
            con=con,
            crs=catalog.gdf.crs,
            backend=catalog.backend,
            _owns_con=owns_con,
        )

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
        self._require_open_con()
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
        self._require_open_con()
        if slice_ is not None and (bounds is not None or time is not None):
            raise TypeError("query: pass either slice_ or (bounds + time), not both")
        if slice_ is not None:
            q_bounds = slice_.bounds
            q_crs = slice_.crs
            q_interval = slice_.interval
        else:
            if bounds is None and time is None:
                return self._derive(self.relation)
            q_bounds = bounds
            q_crs = crs
            q_interval = _coerce_interval(time) if time is not None else None

        # The relation API's `.filter()` takes a SQL string, so these
        # predicates are interpolated rather than bound as parameters.
        # Every interpolated value is force-coerced to a safe literal
        # first (float / `pd.Timestamp.isoformat`) — never interpolate
        # a raw user string here.
        where: list[str] = []
        if q_bounds is not None:
            xmin, ymin, xmax, ymax = (
                float(v) for v in _reproject_bounds(q_bounds, q_crs, self.crs)
            )
            where.append(
                f"ST_Intersects(geometry, "
                f"ST_MakeEnvelope({xmin!r}, {ymin!r}, {xmax!r}, {ymax!r}))"
            )
        if q_interval is not None:
            t_lo = pd.Timestamp(q_interval.left).isoformat()
            t_hi = pd.Timestamp(q_interval.right).isoformat()
            where.append(f"end_time >= TIMESTAMP '{t_lo}'")
            where.append(f"start_time <= TIMESTAMP '{t_hi}'")

        if not where:
            return self._derive(self.relation)
        clause = " AND ".join(where)
        filtered = self.relation.filter(clause)
        return self._derive(filtered)

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
        con = self._require_open_con()
        other_duck = _coerce_to_duckdb(other, con=con, target_crs=self.crs)

        left_name = f"_geocatalog_left_{id(self):x}"
        right_name = f"_geocatalog_right_{id(other_duck):x}"
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
        # GEOS intersection is not bit-symmetric under operand order for
        # near-degenerate sliver overlaps (gh #40) — canonicalise each
        # pair's operand order by WKB bytes, mirroring the in-memory
        # engine's `_symmetric_intersection`, so `a.intersect(b)` and
        # `b.intersect(a)` compute identical geometry per row pair.
        sql = f"""
            SELECT
                L.filepath AS filepath,
                CASE
                    WHEN ST_AsWKB(L.geometry) > ST_AsWKB(R.geometry)
                        THEN ST_Intersection(R.geometry, L.geometry)
                    ELSE ST_Intersection(L.geometry, R.geometry)
                END AS geometry,
                {time_select}
            FROM {left_name} AS L
            JOIN {right_name} AS R
              ON ST_Intersects(L.geometry, R.geometry)
                 {temporal}
        """
        joined = con.sql(sql)
        return self._derive(joined)

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
        con = self._require_open_con()
        other_duck = _coerce_to_duckdb(other, con=con, target_crs=self.crs)
        left_name = f"_geocatalog_unionL_{id(self):x}"
        right_name = f"_geocatalog_unionR_{id(other_duck):x}"
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
        unioned = con.sql(sql)
        return self._derive(unioned)

    def iter_rows(self, *, batch_size: int = 1024) -> Iterator[CatalogRow]:
        """Stream rows as `CatalogRow` instances in Arrow batches.

        Streams the relation through DuckDB's Arrow record-batch reader,
        so time-to-first-row and peak memory are ``O(batch_size)`` —
        not ``O(len(catalog))`` as the previous ``.df()``-materialising
        implementation was. Geometry WKB is decoded one vectorised
        `shapely.from_wkb` call per batch.

        Args:
            batch_size: Rows per Arrow record batch. Bounds both the
                first-row latency and the peak memory of a full
                iteration.

        Yields:
            `CatalogRow` with ``geometry`` decoded from WKB.
        """
        self._require_open_con()
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}")
        for batch in self._arrow_reader(batch_size):
            df = batch.to_pandas()
            if len(df) == 0:
                continue
            geoms = _decode_geometry_column(df["geometry"])
            starts = pd.to_datetime(df["start_time"])
            ends = pd.to_datetime(df["end_time"])
            # `_backend`, `_schema_version` and any other underscore-prefixed
            # column belong to the on-disk schema, not the user-visible row
            # metadata. Filtering them keeps `extras` clean for downstream
            # loaders that introspect it.
            extra_cols = [
                c
                for c in df.columns
                if c not in RESERVED_COLUMNS and not c.startswith("_")
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

    def _arrow_reader(self, batch_size: int) -> Any:
        """Record-batch reader over the relation, across DuckDB versions.

        ``to_arrow_reader`` is the current API; ``fetch_arrow_reader``
        is its deprecated pre-1.x-series spelling.
        """
        to_reader = getattr(self.relation, "to_arrow_reader", None)
        if to_reader is not None:
            return to_reader(batch_size)
        return self.relation.fetch_arrow_reader(batch_size)

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
        # `align="off"` because footprints are arbitrary shapes; their
        # bbox extents are almost never integer multiples of an
        # arbitrary target resolution, so a stricter default would
        # warn on every row. Callers wanting validation call
        # `aligned_shape()` on the emitted slice.
        for row in self.iter_rows():
            yield GeoSlice(
                bounds=tuple(row.geometry.bounds),  # type: ignore[arg-type]
                interval=row.interval,
                resolution=resolution,
                crs=row.crs,
                align="off",
            )

    # ── properties + persistence ─────────────────────────────────────────

    @cached_property
    def total_bounds(self) -> tuple[float, float, float, float]:
        """Union bbox over the relation — one SQL aggregate, not a scan.

        Returns:
            ``(xmin, ymin, xmax, ymax)`` in catalog-CRS units. Four
            NaNs for an empty catalog.
        """
        self._require_open_con()
        df = self.relation.aggregate(
            "MIN(ST_XMin(geometry)) AS xmin, "
            "MIN(ST_YMin(geometry)) AS ymin, "
            "MAX(ST_XMax(geometry)) AS xmax, "
            "MAX(ST_YMax(geometry)) AS ymax"
        ).df()
        if pd.isna(df["xmin"].iloc[0]):
            return (np.nan, np.nan, np.nan, np.nan)
        return (
            float(df["xmin"].iloc[0]),
            float(df["ymin"].iloc[0]),
            float(df["xmax"].iloc[0]),
            float(df["ymax"].iloc[0]),
        )

    @cached_property
    def temporal_extent(self) -> pd.Interval:
        """Tightest interval over the relation — one SQL aggregate.

        Returns:
            ``pd.Interval(min(start_time), max(end_time), closed='both')``.
            Both endpoints are ``pd.NaT`` for an empty catalog.
        """
        self._require_open_con()
        df = self.relation.aggregate(
            "MIN(start_time) AS tmin, MAX(end_time) AS tmax"
        ).df()
        if pd.isna(df["tmin"].iloc[0]):
            return pd.Interval(pd.NaT, pd.NaT, closed="both")
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
        from geocatalog._src.parquet import to_geoparquet as _write

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
        self._require_open_con()
        return self._derive(self.relation.filter(where))

    @cached_property
    def _row_count(self) -> int:
        """Cached row count from one COUNT(*) query."""
        self._require_open_con()
        df = self.relation.aggregate("COUNT(*) AS n").df()
        return int(df["n"].iloc[0])

    def __len__(self) -> int:
        """Number of rows — cached after one COUNT(*) query."""
        self._require_open_con()
        return self._row_count

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


def _read_geoparquet_crs(
    source: str | Path, *, default: str, strict: bool = False
) -> Any:
    """Pull the catalog CRS out of a GeoParquet file's column metadata.

    GeoParquet stores CRS as PROJJSON inside the ``geo`` key of the
    Parquet file metadata (`pyarrow` exposes it as raw bytes). This
    helper decodes it via `pyproj.CRS.from_user_input` so the DuckDB
    backend can carry the catalog CRS even though DuckDB itself ignores
    the GeoParquet column metadata.

    Directory sources are special-cased: we walk ``rglob("*.parquet")``
    and inspect the first shard found. All shards written by
    `StreamingParquetWriter` carry the same CRS, so one is
    representative. A directory with no ``.parquet`` files raises
    `FileNotFoundError` rather than silently falling back to the
    default (mistyped path == data-loss bug otherwise).

    Glob *strings* (``"shards/*.parquet"`` etc.) still fall back to the
    default — there's no unambiguous "first shard" of an arbitrary
    glob pattern without re-implementing the glob expansion, and
    DuckDB's own glob support pulls from heterogeneous sources where
    picking one shard could be misleading.
    """
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(source)
    if path.is_dir():
        first_shard = next(path.rglob("*.parquet"), None)
        if first_shard is None:
            raise FileNotFoundError(f"No .parquet files found in directory: {path}")
        path = first_shard
    if not path.is_file():
        return default
    try:
        md = pq.read_metadata(path).metadata or {}
    except (OSError, pa.ArrowInvalid) as exc:
        # A corrupt or unreadable shard shouldn't silently masquerade as
        # EPSG:4326 without a trace — surface the reason at WARNING so
        # operators can tell "no geo metadata" apart from "broken file".
        if strict:
            raise CatalogMetadataError(
                f"could not read Parquet metadata from {path}: {exc}. "
                "Pass crs=... explicitly, or fix the source."
            ) from exc
        log.warning(
            "duckdb backend: could not read Parquet metadata from {!r} "
            "({}); falling back to {}",
            str(path),
            exc,
            default,
        )
        return default
    geo = md.get(b"geo")
    if geo is None:
        return default
    try:
        geo_meta = json.loads(geo.decode())
        primary = geo_meta.get("primary_column", "geometry")
        crs_val = geo_meta.get("columns", {}).get(primary, {}).get("crs")
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        # ValueError covers json.JSONDecodeError and UnicodeDecodeError;
        # AttributeError/TypeError cover structurally valid JSON whose
        # `columns` / primary-column entries aren't mappings.
        if strict:
            raise CatalogMetadataError(
                f"malformed GeoParquet 'geo' metadata in {path}: {exc}. "
                "Pass crs=... explicitly, or fix the source."
            ) from exc
        log.warning(
            "duckdb backend: malformed GeoParquet 'geo' metadata in {!r} "
            "({}); falling back to {}",
            str(path),
            exc,
            default,
        )
        return default
    if crs_val is None:
        return default
    try:
        return pyproj.CRS.from_user_input(crs_val)
    except pyproj.exceptions.CRSError as exc:
        if strict:
            raise CatalogMetadataError(
                f"unparseable CRS in GeoParquet 'geo' metadata of {path}: "
                f"{exc}. Pass crs=... explicitly, or fix the source."
            ) from exc
        log.warning(
            "duckdb backend: unparseable CRS in GeoParquet 'geo' metadata "
            "of {!r} ({}); falling back to {}",
            str(path),
            exc,
            default,
        )
        return default


def _check_schema_version(
    con: duckdb_mod.DuckDBPyConnection, source: str, *, partitioned: bool = False
) -> None:
    """Raise `CatalogSchemaError` if the artifact's `_schema_version` is unsupported.

    Aggregates the reserved ``_schema_version`` column across *every*
    shard in the source — `DuckDBGeoCatalog.open` supports directories
    / globs, so a multi-file source could legitimately mix versions
    if the user concatenated shards from different library releases.
    `LIMIT 1` would sample one row and miss the conflict (codex P1).

    Three rejection cases:

    1. ``max(version) > SCHEMA_VERSION_CURRENT`` — the artifact is
       newer than the reader. User should upgrade `geocatalog`.
    2. ``min(version) < SCHEMA_VERSION_CURRENT`` — the artifact is
       older than the reader. The DuckDB backend doesn't materialise
       the relation through pandas just to migrate; the user should
       run ``geocatalog migrate`` on the affected file(s).
    3. ``min(version) != max(version)`` — mixed-version shards. We
       can't pick which is canonical; the user has to migrate each
       shard separately or rewrite into one file.

    Ad-hoc parquet files without the column are treated as
    ``SCHEMA_VERSION_CURRENT`` (no migration needed).
    """
    from geocatalog._src.base import CatalogSchemaError
    from geocatalog._src.parquet import SCHEMA_VERSION_CURRENT

    dd = _require_duckdb()
    try:
        df = con.sql(
            "SELECT MIN(_schema_version) AS lo, MAX(_schema_version) AS hi "
            "FROM read_parquet($src, hive_partitioning = $hive)",
            params={"src": source, "hive": partitioned},
        ).df()
    except dd.BinderException:
        # Missing `_schema_version` column — externally produced parquet.
        return
    except dd.IOException:
        # Unreadable parquet path; caller will hit a clearer error
        # on the next read.
        return
    if len(df) == 0 or pd.isna(df["lo"].iloc[0]) or pd.isna(df["hi"].iloc[0]):
        return
    lo = int(df["lo"].iloc[0])
    hi = int(df["hi"].iloc[0])
    if lo != hi:
        raise CatalogSchemaError(
            f"artifact {source} has mixed `_schema_version` values "
            f"(min={lo}, max={hi}); the DuckDB backend can't open a "
            "multi-version source. Migrate each shard separately or "
            "rewrite into one file at a single version."
        )
    v_artifact = lo
    if v_artifact > SCHEMA_VERSION_CURRENT:
        raise CatalogSchemaError(
            f"artifact {source} has _schema_version={v_artifact}, "
            f"exceeds reader v{SCHEMA_VERSION_CURRENT}. "
            "Upgrade `geocatalog` to read this artifact."
        )
    if v_artifact < SCHEMA_VERSION_CURRENT:
        raise CatalogSchemaError(
            f"artifact {source} has _schema_version={v_artifact} < reader "
            f"v{SCHEMA_VERSION_CURRENT}. The DuckDB backend does not run "
            "forward migrations in-place; bring the artifact up to date with "
            f"`geocatalog migrate {source} --to-version {SCHEMA_VERSION_CURRENT}`."
        )


def _read_backend_tag(
    con: duckdb_mod.DuckDBPyConnection,
    source: str,
    *,
    default: _BACKEND_T,
    partitioned: bool = False,
    strict: bool = False,
) -> _BACKEND_T:
    """Recover the ``_backend`` column written by `to_geoparquet`.

    Returns the default for ad-hoc parquet files lacking the column;
    that's the case when an externally produced GeoParquet (e.g. one
    built by DuckDB or GDAL directly) is opened with this factory.

    Uses parameter binding for ``source`` so paths with apostrophes
    don't break the lookup, and narrowed exception handling so genuine
    SQL parse errors aren't silently swallowed as a missing column.
    """
    source_cache = _lookup_backend_cache(con)
    if source_cache is not None:
        cached = source_cache.get(source)
        if cached is not None:
            return cached

    dd = _require_duckdb()
    try:
        df = con.sql(
            "SELECT _backend FROM read_parquet($src, hive_partitioning = $hive) "
            "LIMIT 1",
            params={"src": source, "hive": partitioned},
        ).df()
    except dd.BinderException:
        # Missing `_backend` column — externally produced parquet.
        if strict:
            raise CatalogMetadataError(
                f"{source} is missing the reserved '_backend' column. "
                "Pass backend=... explicitly, or write the catalog via "
                "geocatalog's to_geoparquet first."
            ) from None
        log.warning(
            "opened {!r}: no _backend column found; defaulting to "
            "backend={!r}. Pass backend=... explicitly to silence.",
            source,
            default,
        )
        _cache_backend_tag(con, source, default)
        return default
    except dd.IOException as exc:
        # Unreadable parquet path; caller will hit a clearer error
        # on the next read.
        if strict:
            raise CatalogMetadataError(
                f"could not read '_backend' column from {source}: {exc}. "
                "Pass backend=... explicitly, or fix the source."
            ) from exc
        log.warning(
            "opened {!r}: could not read _backend column ({}); defaulting "
            "to backend={!r}. Pass backend=... explicitly to silence.",
            source,
            exc,
            default,
        )
        _cache_backend_tag(con, source, default)
        return default
    if len(df) == 0 or pd.isna(df["_backend"].iloc[0]):
        if strict:
            raise CatalogMetadataError(
                f"{source} has a '_backend' column but no readable value "
                "(empty artifact or null tag). Pass backend=... explicitly, "
                "or fix the source."
            )
        log.warning(
            "opened {!r}: _backend column present but empty/null; defaulting "
            "to backend={!r}. Pass backend=... explicitly to silence.",
            source,
            default,
        )
        _cache_backend_tag(con, source, default)
        return default
    tag = str(df["_backend"].iloc[0])
    if tag in ("raster", "xarray", "vector"):
        _cache_backend_tag(con, source, cast(_BACKEND_T, tag))
        return cast(_BACKEND_T, tag)
    if strict:
        raise CatalogMetadataError(
            f"{source} carries an unrecognised _backend tag {tag!r}; expected "
            "'raster', 'xarray', or 'vector'. Pass backend=... explicitly, "
            "or fix the source."
        )
    log.warning(
        "opened {!r}: unrecognised _backend tag {!r}; defaulting to "
        "backend={!r}. Pass backend=... explicitly to silence.",
        source,
        tag,
        default,
    )
    _cache_backend_tag(con, source, default)
    return default


def _lookup_backend_cache(
    con: duckdb_mod.DuckDBPyConnection,
) -> dict[str, _BACKEND_T] | None:
    """`WeakKeyDictionary` lookup with a non-weakref-able-key fallback.

    The repo dep is `duckdb>=1.1`, and `DuckDBPyConnection` only gained
    weakref support in newer releases — on older DuckDBs both `.get(con)`
    and `cache[con] = ...` raise `TypeError`. Treat that as a cache miss
    so behaviour stays correct; only the per-source memoisation is lost.
    """
    try:
        return _BACKEND_TAG_CACHE.get(con)
    except TypeError:
        return None


def _cache_backend_tag(
    con: duckdb_mod.DuckDBPyConnection, source: str, tag: _BACKEND_T
) -> None:
    source_cache = _lookup_backend_cache(con)
    if source_cache is None:
        source_cache = {}
        # See `_lookup_backend_cache` — older DuckDB versions reject
        # weakref keys; skip caching for those connections.
        try:
            _BACKEND_TAG_CACHE[con] = source_cache
        except TypeError:
            return
    source_cache[source] = tag


def _read_parquet_source(source: str | Path) -> str:
    """Return a DuckDB `read_parquet` source for files or partitioned dirs."""
    path = Path(source)
    if path.is_dir():
        return str(path / "**" / "*.parquet")
    return str(source)


def _is_partitioned_source(source: str | Path) -> bool:
    """True for directories/globs, False for single-file paths.

    Hive partitioning must only be enabled when DuckDB is scanning a
    tree of shards (or a glob) — turning it on for a single-file
    source that happens to sit under a ``key=value`` directory injects
    synthetic partition columns (e.g. ``year``) that the catalog
    schema shouldn't carry.
    """
    if isinstance(source, str) and ("*" in source or "?" in source):
        return True
    try:
        return Path(source).is_dir()
    except (TypeError, ValueError):
        return False


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
    values = col.to_numpy() if hasattr(col, "to_numpy") else list(col)
    # Fast path: a homogeneous WKB column decodes in one vectorised
    # shapely call (~50x faster than per-row `shapely.from_wkb`).
    if len(values) > 0 and all(
        isinstance(v, (bytes, bytearray, memoryview)) for v in values
    ):
        wkb = np.empty(len(values), dtype=object)
        wkb[:] = [bytes(v) for v in values]
        return list(shapely.from_wkb(wkb))
    out: list[Any] = []
    for val in values:
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
