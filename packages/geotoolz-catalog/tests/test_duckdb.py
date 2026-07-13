"""Tests for the DuckDB-backed catalog (Phase 2). Skipped without [duckdb]."""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import shapely.geometry


duckdb = pytest.importorskip("duckdb")

import geocatalog._src.duckdb_backend as duckdb_backend
from geocatalog import (
    DuckDBGeoCatalog,
    GeoSlice,
    InMemoryGeoCatalog,
    open_catalog,
    to_geoparquet,
)


CLOSED_CONNECTION_MESSAGE = (
    "DuckDBGeoCatalog connection has already been closed. "
    "Open a new catalog, or keep the parent catalog open when using derived catalogs."
)
CLOSED_CONNECTION_MATCH = re.escape(CLOSED_CONNECTION_MESSAGE)
# Substring rather than DuckDB's full error phrase — message wording has
# shifted across releases (e.g. "Connection has already been closed",
# "Connection already closed!", "Connection Error: ... already been
# closed"), so anchor on the stable "already" + "closed" fragment to
# keep the test version-tolerant.
DUCKDB_CLOSED_CONNECTION_MATCH = r"[Aa]lready.*closed"


def _mem_two_tiles(crs: str = "EPSG:32629") -> InMemoryGeoCatalog:
    """Two non-overlapping tiles, slightly offset in time."""
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                shapely.geometry.box(0, 0, 100, 100),
                shapely.geometry.box(200, 0, 300, 100),
            ],
            "start_time": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
            ],
            "end_time": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            ],
            "filepath": ["A.tif", "B.tif"],
        },
        geometry="geometry",
        crs=crs,
    )
    return InMemoryGeoCatalog(gdf, backend="raster")


class _AggregateResult:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def df(self) -> pd.DataFrame:
        return self._df


class _CountingRelation:
    def __init__(
        self,
        *,
        n: int = 2,
        bounds: tuple[float, float, float, float] = (0.0, 0.0, 300.0, 100.0),
        temporal: tuple[pd.Timestamp, pd.Timestamp] = (
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-03"),
        ),
        child_relation: _CountingRelation | None = None,
    ) -> None:
        self.n = n
        self.bounds = bounds
        self.temporal = temporal
        self.child_relation = child_relation
        self.aggregate_calls: list[str] = []
        self.filter_calls: list[str] = []

    def aggregate(self, expression: str) -> _AggregateResult:
        self.aggregate_calls.append(expression)
        if "COUNT(*)" in expression:
            return _AggregateResult(pd.DataFrame({"n": [self.n]}))
        if "ST_XMin" in expression:
            xmin, ymin, xmax, ymax = self.bounds
            return _AggregateResult(
                pd.DataFrame(
                    {
                        "xmin": [xmin],
                        "ymin": [ymin],
                        "xmax": [xmax],
                        "ymax": [ymax],
                    }
                )
            )
        if "start_time" in expression:
            tmin, tmax = self.temporal
            return _AggregateResult(pd.DataFrame({"tmin": [tmin], "tmax": [tmax]}))
        raise AssertionError(f"unexpected aggregate: {expression}")

    def filter(self, where: str) -> _CountingRelation:
        self.filter_calls.append(where)
        if self.child_relation is None:
            self.child_relation = _CountingRelation(
                n=self.n,
                bounds=self.bounds,
                temporal=self.temporal,
            )
        return self.child_relation


class _CountingConnection:
    def __init__(self, backend: str = "vector") -> None:
        self.backend = backend
        self.calls = 0

    def sql(self, _query: str, **_kwargs: object) -> _AggregateResult:
        self.calls += 1
        return _AggregateResult(pd.DataFrame({"_backend": [self.backend]}))


@pytest.fixture
def parquet_two_tiles(tmp_path: Path) -> Path:
    """A GeoParquet artifact written by `to_geoparquet`."""
    mem = _mem_two_tiles()
    path = tmp_path / "cat.parquet"
    to_geoparquet(mem, path)
    return path


class TestFromMemory:
    def test_wraps_in_memory(self) -> None:
        mem = _mem_two_tiles()
        duck = DuckDBGeoCatalog.from_memory(mem)
        assert isinstance(duck, DuckDBGeoCatalog)
        assert len(duck) == 2
        assert duck.crs == mem.gdf.crs
        assert duck.backend == "raster"

    def test_materialize_round_trip(self) -> None:
        mem = _mem_two_tiles()
        duck = DuckDBGeoCatalog.from_memory(mem)
        out = duck.materialize()
        assert isinstance(out, InMemoryGeoCatalog)
        assert len(out) == 2
        assert set(out.gdf["filepath"]) == {"A.tif", "B.tif"}


class TestOpen:
    def test_reads_crs_from_geoparquet_metadata(self, parquet_two_tiles: Path) -> None:
        """Regression for the early-cut bug where DuckDB backend
        fell back to EPSG:4326 because it never inspected the
        Parquet `geo` column metadata, breaking every subsequent
        spatial query in non-4326 catalog space."""
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        assert duck.crs.to_epsg() == 32629
        # And consequently the query in the catalog's CRS works:
        out = duck.query(bounds=(0, 0, 50, 50), crs="EPSG:32629")
        assert len(out) == 1

    def test_reads_backend_tag(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        assert duck.backend == "raster"

    def test_factory_auto_picks_duckdb(self, parquet_two_tiles: Path) -> None:
        cat = open_catalog(parquet_two_tiles)
        # The auto path prefers DuckDB when the extra is installed.
        assert isinstance(cat, DuckDBGeoCatalog)

    def test_factory_memory_falls_back_to_inmemory(
        self, parquet_two_tiles: Path
    ) -> None:
        cat = open_catalog(parquet_two_tiles, engine="memory")
        assert isinstance(cat, InMemoryGeoCatalog)


class TestLifecycle:
    def test_context_manager_closes_owned_connection(
        self, parquet_two_tiles: Path
    ) -> None:
        with DuckDBGeoCatalog.open(parquet_two_tiles) as duck:
            assert len(duck) == 2

        assert duck.con is None
        with pytest.raises(
            duckdb.ConnectionException,
            match=CLOSED_CONNECTION_MATCH,
        ):
            len(duck)
        with pytest.raises(
            duckdb.ConnectionException,
            match=CLOSED_CONNECTION_MATCH,
        ):
            list(duck.iter_rows())

    def test_derived_catalog_close_preserves_parent_connection(
        self, parquet_two_tiles: Path
    ) -> None:
        duck = DuckDBGeoCatalog.open(parquet_two_tiles)
        try:
            filtered = duck.query(bounds=(0, 0, 50, 50), crs="EPSG:32629")

            filtered.close()

            assert duck.con is not None
            assert len(duck) == 2
            assert len(filtered) == 1
        finally:
            duck.close()

    def test_closing_parent_invalidates_derived_catalog(
        self, parquet_two_tiles: Path
    ) -> None:
        duck = DuckDBGeoCatalog.open(parquet_two_tiles)
        filtered = duck.query(bounds=(0, 0, 50, 50), crs="EPSG:32629")

        duck.close()

        assert duck.con is None
        with pytest.raises(
            duckdb.ConnectionException,
            match=CLOSED_CONNECTION_MATCH,
        ):
            len(duck)
        with pytest.raises(
            duckdb.ConnectionException,
            match=DUCKDB_CLOSED_CONNECTION_MATCH,
        ):
            len(filtered)

    def test_close_is_idempotent(self, parquet_two_tiles: Path) -> None:
        duck = DuckDBGeoCatalog.open(parquet_two_tiles)
        duck.close()
        # Second close on an owned-but-already-closed catalog must not
        # raise — common in `try/finally` cleanup paths.
        duck.close()
        assert duck.con is None

    def test_from_memory_owns_fresh_connection(self) -> None:
        mem = _mem_two_tiles()
        duck = DuckDBGeoCatalog.from_memory(mem)
        try:
            assert duck._owns_con is True
            assert len(duck) == 2
        finally:
            duck.close()
        assert duck.con is None

    def test_from_memory_does_not_own_external_connection(self) -> None:
        dd = duckdb.connect()
        try:
            mem = _mem_two_tiles()
            duck = DuckDBGeoCatalog.from_memory(mem, con=dd)
            assert duck._owns_con is False
            duck.close()
            # External connection must still be usable after derived
            # catalog's close (which is a no-op).
            assert dd.execute("SELECT 1").fetchone() == (1,)
        finally:
            dd.close()

    def test_open_closes_connection_when_setup_raises(
        self, parquet_two_tiles: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: if `_ensure_spatial` (or any other setup step in
        `open()`) raises, the freshly opened connection must be closed
        before the exception propagates — otherwise long-lived processes
        leak a DuckDB handle per failed open.
        """
        from geocatalog._src import duckdb_backend as backend

        opened: list[duckdb.DuckDBPyConnection] = []
        real_connect = backend.duckdb.connect

        def recording_connect(*args: object, **kwargs: object):
            con = real_connect(*args, **kwargs)
            opened.append(con)
            return con

        def boom(_con: duckdb.DuckDBPyConnection) -> None:
            raise RuntimeError("simulated extension load failure")

        monkeypatch.setattr(backend.duckdb, "connect", recording_connect)
        monkeypatch.setattr(backend, "_ensure_spatial", boom)

        with pytest.raises(RuntimeError, match="simulated extension load failure"):
            DuckDBGeoCatalog.open(parquet_two_tiles)

        assert len(opened) == 1
        # SELECT 1 on the recorded connection should fail because it was
        # closed during failure-path cleanup.
        with pytest.raises(
            duckdb.ConnectionException,
            match=DUCKDB_CLOSED_CONNECTION_MATCH,
        ):
            opened[0].execute("SELECT 1")

    def test_from_memory_closes_owned_connection_when_setup_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mirror of the `open()` failure-path test for `from_memory()`
        when it allocates a fresh connection: a failure inside
        `_ensure_spatial` must close that connection before re-raising.
        Externally-supplied connections stay untouched (covered by
        `test_from_memory_does_not_own_external_connection`).
        """
        from geocatalog._src import duckdb_backend as backend

        opened: list[duckdb.DuckDBPyConnection] = []
        real_connect = backend.duckdb.connect

        def recording_connect(*args: object, **kwargs: object):
            con = real_connect(*args, **kwargs)
            opened.append(con)
            return con

        def boom(_con: duckdb.DuckDBPyConnection) -> None:
            raise RuntimeError("simulated extension load failure")

        monkeypatch.setattr(backend.duckdb, "connect", recording_connect)
        monkeypatch.setattr(backend, "_ensure_spatial", boom)

        mem = _mem_two_tiles()
        with pytest.raises(RuntimeError, match="simulated extension load failure"):
            DuckDBGeoCatalog.from_memory(mem)

        assert len(opened) == 1
        with pytest.raises(
            duckdb.ConnectionException,
            match=DUCKDB_CLOSED_CONNECTION_MATCH,
        ):
            opened[0].execute("SELECT 1")

    def test_fluent_chain_records_owner_via_from_memory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression for the codex P1 fluent-chain leak: a derived
        catalog from `DuckDBGeoCatalog.from_memory(mem).query(...)` must
        hold a strong ref back to the originating owning catalog via
        `_owner`, and using the derivation as a context manager must
        close that owner on `__exit__`. Without this, the only
        reference to the owning catalog is dropped mid-expression and
        the underlying DuckDB connection leaks for the lifetime of the
        process.

        We bypass `_ensure_spatial` because no spatial operations run
        in this test — the chain uses a `ST_Intersects` predicate
        through `query()` but the AOI filter is the only spatial bit
        and the test only inspects row count, geometry decode, and
        connection state. Keeping the test independent of the
        network-fetched `spatial` extension also keeps it green in
        sandboxed CI.
        """
        from geocatalog._src import duckdb_backend as backend

        monkeypatch.setattr(backend, "_ensure_spatial", lambda _con: None)

        mem = _mem_two_tiles()
        # `.query()` without bounds returns a no-filter derivation —
        # avoids needing `ST_Intersects` (which lives in the spatial
        # extension we just stubbed out).
        with DuckDBGeoCatalog.from_memory(mem).query() as cat:
            assert len(cat) == 2
            owner = cat._owner
            assert owner is not None
            assert owner._owns_con is True
            assert owner.con is not None

        # After exit the owner must be closed and the shared connection
        # torn down — `cat` points at the same connection so native
        # DuckDB operations on it now fail.
        assert owner.con is None
        with pytest.raises(
            duckdb.ConnectionException,
            match=DUCKDB_CLOSED_CONNECTION_MATCH,
        ):
            len(cat)

    def test_fluent_chain_of_derivations_holds_owner_alive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chained derivations (`A.query().sql(...)`) must transitively
        anchor the owner: every derived catalog along the chain points
        at the same root owner so a multi-step fluent expression doesn't
        leak the connection either.
        """
        from geocatalog._src import duckdb_backend as backend

        monkeypatch.setattr(backend, "_ensure_spatial", lambda _con: None)

        mem = _mem_two_tiles()
        with DuckDBGeoCatalog.from_memory(mem).query().sql("filepath = 'A.tif'") as cat:
            # Inner derivation inherits the same root owner as the
            # intermediate one.
            assert cat._owner is not None
            assert cat._owner._owns_con is True
            assert len(cat) == 1
            owner = cat._owner

        assert owner.con is None

    def test_derived_close_remains_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bare `derived.close()` (without context manager) must stay
        a no-op so sibling derivations on the same owner keep working —
        this guards against accidentally repurposing `close()` as the
        chain-tear-down hook now that `__exit__` does that job.
        """
        from geocatalog._src import duckdb_backend as backend

        monkeypatch.setattr(backend, "_ensure_spatial", lambda _con: None)

        mem = _mem_two_tiles()
        owner = DuckDBGeoCatalog.from_memory(mem)
        try:
            derived = owner.query()
            assert derived._owner is owner
            derived.close()  # no-op
            assert owner.con is not None
            assert len(owner) == 2
            assert len(derived) == 2
        finally:
            owner.close()


class TestQuery:
    def test_spatial_filter(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        out = duck.query(bounds=(0, 0, 50, 50), crs="EPSG:32629")
        assert len(out) == 1
        assert out.materialize().gdf["filepath"].iloc[0] == "A.tif"

    def test_temporal_filter_via_slice(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        sl = GeoSlice(
            bounds=(0, 0, 300, 100),
            interval=pd.Interval(
                pd.Timestamp("2024-01-02 06:00"),
                pd.Timestamp("2024-01-03 12:00"),
                closed="both",
            ),
            resolution=(1.0, 1.0),
            crs="EPSG:32629",
        )
        out = duck.query(sl)
        # Both tiles overlap the time window — A ends 01-02 00:00 which is
        # before the query start; B (01-02 → 01-03) overlaps.
        files = set(out.materialize().gdf["filepath"])
        assert "B.tif" in files

    def test_cross_crs_query_reprojects(self, parquet_two_tiles: Path) -> None:
        """Regression for the §10.1-style footgun: a 4326 AOI must not
        silently return zero rows from a UTM-zone-29N catalog."""
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        # UTM 29N (50, 50) ≈ (-13.488, 0.00045) in 4326.
        out = duck.query(bounds=(-13.4885, 0.0001, -13.4880, 0.0008), crs="EPSG:4326")
        assert len(out) == 1
        assert out.materialize().gdf["filepath"].iloc[0] == "A.tif"

    def test_rejects_both_slice_and_parts(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        sl = GeoSlice(
            bounds=(0, 0, 50, 50),
            interval=pd.Interval(0, 1, closed="both"),
            resolution=(1.0, 1.0),
            crs="EPSG:32629",
        )
        with pytest.raises(TypeError, match="either"):
            duck.query(sl, bounds=(0, 0, 50, 50))


class TestSetAlgebra:
    def test_intersect_spatial_join_clips(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        # Pair each row with a vector catalog covering tile A only.
        labels = InMemoryGeoCatalog(
            gpd.GeoDataFrame(
                {
                    "geometry": [shapely.geometry.box(50, 50, 250, 150)],
                    "start_time": [pd.Timestamp("2024-01-01")],
                    "end_time": [pd.Timestamp("2024-01-04")],
                    "filepath": ["labels.gpkg"],
                },
                geometry="geometry",
                crs="EPSG:32629",
            ),
            backend="vector",
        )
        joint = duck.intersect(labels)
        mat = joint.materialize()
        assert len(mat) == 2  # Both tiles spatially overlap the label tile.
        bounds_set = {tuple(g.bounds) for g in mat.gdf.geometry}
        assert (50.0, 50.0, 100.0, 100.0) in bounds_set  # A ∩ labels
        assert (200.0, 50.0, 250.0, 100.0) in bounds_set  # B ∩ labels

    def test_intersect_spatial_only(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        # Far-future label so the temporal axis would otherwise drop both.
        labels = InMemoryGeoCatalog(
            gpd.GeoDataFrame(
                {
                    "geometry": [shapely.geometry.box(50, 50, 250, 150)],
                    "start_time": [pd.Timestamp("2030-01-01")],
                    "end_time": [pd.Timestamp("2030-01-04")],
                    "filepath": ["future.gpkg"],
                },
                geometry="geometry",
                crs="EPSG:32629",
            ),
            backend="vector",
        )
        joint = duck.intersect(labels, spatial_only=True)
        assert len(joint) == 2

    def test_intersect_temporal_filter_drops_mismatch(
        self, parquet_two_tiles: Path
    ) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        labels = InMemoryGeoCatalog(
            gpd.GeoDataFrame(
                {
                    "geometry": [shapely.geometry.box(50, 50, 250, 150)],
                    "start_time": [pd.Timestamp("2030-01-01")],
                    "end_time": [pd.Timestamp("2030-01-04")],
                    "filepath": ["future.gpkg"],
                },
                geometry="geometry",
                crs="EPSG:32629",
            ),
            backend="vector",
        )
        joint = duck.intersect(labels)  # spatial_only=False
        assert len(joint) == 0

    def test_union_concatenates(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        other = _mem_two_tiles()
        merged = duck.union(other)
        assert len(merged) == 4

    def test_union_reprojects(self, parquet_two_tiles: Path) -> None:
        """Union with a non-matching-CRS catalog reprojects under the hood."""
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        other = InMemoryGeoCatalog(
            gpd.GeoDataFrame(
                {
                    "geometry": [
                        shapely.geometry.box(400_000, 4_000_000, 500_000, 4_100_000)
                    ],
                    "start_time": [pd.Timestamp("2024-02-01")],
                    "end_time": [pd.Timestamp("2024-02-02")],
                    "filepath": ["tile_C.tif"],
                },
                geometry="geometry",
                crs="EPSG:32630",
            ),
            backend="raster",
        )
        merged = duck.union(other)
        assert len(merged) == 3


class TestIterators:
    def test_iter_rows_yields_catalog_rows(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        rows = list(duck.iter_rows())
        assert len(rows) == 2
        # Filepaths preserved; geometry decoded to shapely.
        assert {r.filepath for r in rows} == {"A.tif", "B.tif"}
        for r in rows:
            assert hasattr(r.geometry, "bounds")
            assert r.crs.to_epsg() == 32629
            assert r.interval.closed == "both"

    def test_iter_rows_on_in_memory_catalog(self) -> None:
        """`iter_rows` is on the Protocol — both backends honour it."""
        mem = _mem_two_tiles()
        rows = list(mem.iter_rows())
        assert len(rows) == 2
        assert {r.filepath for r in rows} == {"A.tif", "B.tif"}

    def test_iter_slices_at_resolution(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        slices = list(duck.iter_slices(resolution=(10.0, 10.0)))
        assert len(slices) == 2
        for sl in slices:
            assert isinstance(sl, GeoSlice)
            assert sl.resolution == (10.0, 10.0)
            assert sl.crs.to_epsg() == 32629


class TestRoundTrip:
    def test_write_then_open(self, tmp_path: Path) -> None:
        mem = _mem_two_tiles()
        duck = DuckDBGeoCatalog.from_memory(mem)
        out = tmp_path / "rt.parquet"
        duck.to_geoparquet(out)
        reopened = open_catalog(out, engine="duckdb")
        assert len(reopened) == 2
        assert reopened.crs.to_epsg() == 32629

    def test_empty_catalog_round_trip(self, tmp_path: Path) -> None:
        """A filtered-to-zero catalog should still materialise cleanly."""
        mem = _mem_two_tiles()
        duck = DuckDBGeoCatalog.from_memory(mem)
        empty = duck.query(bounds=(1e6, 1e6, 2e6, 2e6), crs="EPSG:32629")
        assert len(empty) == 0
        mat = empty.materialize()
        assert isinstance(mat, InMemoryGeoCatalog)
        assert len(mat) == 0


class TestProperties:
    def test_total_bounds(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        assert duck.total_bounds == (0.0, 0.0, 300.0, 100.0)

    def test_temporal_extent(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        ext = duck.temporal_extent
        assert ext.left == pd.Timestamp("2024-01-01")
        assert ext.right == pd.Timestamp("2024-01-03")

    def test_get_config(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        cfg = duck.get_config()
        assert cfg["engine"] == "duckdb"
        assert cfg["backend"] == "raster"
        assert cfg["len"] == 2


class TestCaching:
    def test_len_runs_one_query(self) -> None:
        relation = _CountingRelation()
        duck = DuckDBGeoCatalog(
            relation, con=duckdb.connect(), crs="EPSG:32629", backend="raster"
        )

        for _ in range(10):
            assert len(duck) == 2

        assert len(relation.aggregate_calls) == 1
        assert "COUNT(*)" in relation.aggregate_calls[0]

    def test_total_bounds_runs_one_query(self) -> None:
        relation = _CountingRelation()
        duck = DuckDBGeoCatalog(
            relation, con=duckdb.connect(), crs="EPSG:32629", backend="raster"
        )

        for _ in range(10):
            assert duck.total_bounds == (0.0, 0.0, 300.0, 100.0)

        assert len(relation.aggregate_calls) == 1
        assert "ST_XMin" in relation.aggregate_calls[0]

    def test_temporal_extent_runs_one_query(self) -> None:
        relation = _CountingRelation()
        duck = DuckDBGeoCatalog(
            relation, con=duckdb.connect(), crs="EPSG:32629", backend="raster"
        )

        for _ in range(10):
            ext = duck.temporal_extent
            assert ext.left == pd.Timestamp("2024-01-01")
            assert ext.right == pd.Timestamp("2024-01-03")

        assert len(relation.aggregate_calls) == 1
        assert "start_time" in relation.aggregate_calls[0]

    def test_derived_catalog_gets_own_cache(self) -> None:
        child_relation = _CountingRelation(bounds=(0.0, 0.0, 100.0, 100.0))
        parent_relation = _CountingRelation(child_relation=child_relation)
        duck = DuckDBGeoCatalog(
            parent_relation,
            con=duckdb.connect(),
            crs="EPSG:32629",
            backend="raster",
        )

        assert duck.total_bounds == (0.0, 0.0, 300.0, 100.0)
        assert duck.total_bounds == (0.0, 0.0, 300.0, 100.0)
        derived = duck.query(bounds=(0, 0, 50, 50), crs="EPSG:32629")
        assert derived.total_bounds == (0.0, 0.0, 100.0, 100.0)
        assert derived.total_bounds == (0.0, 0.0, 100.0, 100.0)

        assert len(parent_relation.aggregate_calls) == 1
        assert len(child_relation.aggregate_calls) == 1

    def test_backend_tag_read_is_cached_per_connection_and_source(self) -> None:
        con = _CountingConnection()
        other_con = _CountingConnection("xarray")
        duckdb_backend._BACKEND_TAG_CACHE.clear()

        first = duckdb_backend._read_backend_tag(
            con, "catalog.parquet", default="raster"
        )
        second = duckdb_backend._read_backend_tag(
            con, "catalog.parquet", default="raster"
        )
        other_source = duckdb_backend._read_backend_tag(
            con, "other.parquet", default="raster"
        )
        other_connection = duckdb_backend._read_backend_tag(
            other_con, "catalog.parquet", default="raster"
        )

        assert first == "vector"
        assert second == "vector"
        assert other_source == "vector"
        assert other_connection == "xarray"
        assert con.calls == 2
        assert other_con.calls == 1

    def test_backend_tag_read_falls_back_when_connection_not_weakrefable(
        self,
    ) -> None:
        """Older DuckDB releases (>=1.1, <1.5) ship a `DuckDBPyConnection`
        without weakref support. Adding such a connection as a
        `WeakKeyDictionary` key raises `TypeError`; the helper must
        degrade to no-cache rather than propagate the error."""

        class _NoWeakRefConnection:
            __slots__ = ("backend", "calls")

            def __init__(self) -> None:
                self.backend = "vector"
                self.calls = 0

            def sql(self, _query: str, **_kwargs: object) -> _AggregateResult:
                self.calls += 1
                return _AggregateResult(pd.DataFrame({"_backend": [self.backend]}))

        con = _NoWeakRefConnection()
        duckdb_backend._BACKEND_TAG_CACHE.clear()

        # Sanity: this connection genuinely can't be weakref'd.
        import weakref

        with pytest.raises(TypeError):
            weakref.ref(con)

        first = duckdb_backend._read_backend_tag(
            con, "catalog.parquet", default="raster"
        )
        second = duckdb_backend._read_backend_tag(
            con, "catalog.parquet", default="raster"
        )

        # Correct value on both calls…
        assert first == "vector"
        assert second == "vector"
        # …but cache was skipped, so every call re-queried.
        assert con.calls == 2


class TestSqlEscape:
    def test_sql_filter(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        out = duck.sql("filepath = 'A.tif'")
        assert len(out) == 1
        assert out.materialize().gdf["filepath"].iloc[0] == "A.tif"


class TestRegression:
    def test_open_catalog_honours_stored_backend_tag(self, tmp_path: Path) -> None:
        """Regression for the P1 bug where `open_catalog`'s default
        `backend='raster'` always overrode the `_backend` column DuckDB
        was about to read from the file — silently miscategorising
        xarray / vector catalogs and breaking loader dispatch.
        """
        # Write a vector-tagged catalog…
        mem = _mem_two_tiles()
        mem.backend = "vector"
        path = tmp_path / "labels.parquet"
        to_geoparquet(mem, path)
        # …and re-open it. The backend tag should round-trip.
        reopened = open_catalog(path, engine="duckdb")
        assert reopened.backend == "vector"

    def test_intersect_across_independent_connections(self, tmp_path: Path) -> None:
        """Regression for the P1 bug where `_coerce_to_duckdb`
        returned a same-CRS DuckDBGeoCatalog unchanged even when it
        lived on a different DuckDB connection — making the subsequent
        view-on-self.con join fail because views are connection-scoped.
        """
        a_path = tmp_path / "a.parquet"
        b_path = tmp_path / "b.parquet"
        to_geoparquet(_mem_two_tiles(), a_path)
        # B covers tile A in space and time but on a fresh connection.
        b_mem = InMemoryGeoCatalog(
            gpd.GeoDataFrame(
                {
                    "geometry": [shapely.geometry.box(50, 50, 250, 150)],
                    "start_time": [pd.Timestamp("2024-01-01")],
                    "end_time": [pd.Timestamp("2024-01-04")],
                    "filepath": ["labels.gpkg"],
                },
                geometry="geometry",
                crs="EPSG:32629",
            ),
            backend="vector",
        )
        to_geoparquet(b_mem, b_path)

        a = open_catalog(a_path, engine="duckdb")
        b = open_catalog(b_path, engine="duckdb")
        # Independent connections — without the fix this raised.
        assert a.con is not b.con
        joint = a.intersect(b)
        assert len(joint) == 2

    def test_union_preserves_backend_specific_columns(self, tmp_path: Path) -> None:
        """Regression for the P1 bug where the DuckDB `union` projected
        only `filepath/geometry/start_time/end_time`, dropping per-row
        backend metadata like `layer` or `data_vars` that downstream
        loaders need.
        """
        # Two vector catalogs, each with a `layer` column.
        a = InMemoryGeoCatalog(
            gpd.GeoDataFrame(
                {
                    "geometry": [shapely.geometry.box(0, 0, 100, 100)],
                    "start_time": [pd.Timestamp("2024-01-01")],
                    "end_time": [pd.Timestamp("2024-01-02")],
                    "filepath": ["A.gpkg"],
                    "layer": ["roads"],
                },
                geometry="geometry",
                crs="EPSG:32629",
            ),
            backend="vector",
        )
        b = InMemoryGeoCatalog(
            gpd.GeoDataFrame(
                {
                    "geometry": [shapely.geometry.box(200, 0, 300, 100)],
                    "start_time": [pd.Timestamp("2024-01-02")],
                    "end_time": [pd.Timestamp("2024-01-03")],
                    "filepath": ["B.gpkg"],
                    "layer": ["buildings"],
                },
                geometry="geometry",
                crs="EPSG:32629",
            ),
            backend="vector",
        )
        a_path = tmp_path / "a.parquet"
        b_path = tmp_path / "b.parquet"
        to_geoparquet(a, a_path)
        to_geoparquet(b, b_path)
        a_duck = open_catalog(a_path, engine="duckdb")
        b_duck = open_catalog(b_path, engine="duckdb")
        merged = a_duck.union(b_duck)
        gdf = merged.materialize().gdf
        # The `layer` column survives the union — without the fix it
        # was projected away.
        assert "layer" in gdf.columns
        assert set(gdf["layer"]) == {"roads", "buildings"}

    def test_iter_rows_does_not_leak_schema_metadata_into_extras(
        self, parquet_two_tiles: Path
    ) -> None:
        """Regression for the P2 bug where `_backend` /
        `_schema_version` (and any other underscore-prefixed
        on-disk schema column) leaked into `CatalogRow.extras`.
        """
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        for row in duck.iter_rows():
            assert "_backend" not in row.extras
            assert "_schema_version" not in row.extras
            assert not any(k.startswith("_") for k in row.extras)

    def test_open_handles_paths_with_apostrophe(self, tmp_path: Path) -> None:
        """Regression for the P2 SQL-injection / quote-in-path bug
        where `open()` interpolated the path directly into the SQL,
        breaking on paths containing a single quote and opening a
        SQL-injection surface.
        """
        weird_dir = tmp_path / "o'malley"
        weird_dir.mkdir()
        path = weird_dir / "cat.parquet"
        to_geoparquet(_mem_two_tiles(), path)
        duck = open_catalog(path, engine="duckdb")
        assert len(duck) == 2


class TestIterRowsStreaming:
    """`iter_rows` streams in Arrow batches (gh #4) — no full `.df()`."""

    def test_parity_with_previous_behaviour(self, parquet_two_tiles: Path) -> None:
        cat = DuckDBGeoCatalog.open(parquet_two_tiles)
        rows = list(cat.iter_rows())
        assert [r.filepath for r in rows] == ["A.tif", "B.tif"]
        assert [r.geometry.bounds for r in rows] == [
            (0.0, 0.0, 100.0, 100.0),
            (200.0, 0.0, 300.0, 100.0),
        ]
        assert all(r.interval.closed == "both" for r in rows)
        assert all("_backend" not in r.extras for r in rows)

    def test_batches_are_consumed_incrementally(
        self, parquet_two_tiles: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cat = DuckDBGeoCatalog.open(parquet_two_tiles)
        batches: list[int] = []
        original = DuckDBGeoCatalog._arrow_reader

        def counting(self: DuckDBGeoCatalog, batch_size: int):
            for batch in original(self, batch_size):
                batches.append(batch.num_rows)
                yield batch

        monkeypatch.setattr(DuckDBGeoCatalog, "_arrow_reader", counting)
        it = cat.iter_rows(batch_size=1)
        first = next(it)
        # Only the first batch has been pulled from the reader so far.
        assert batches == [1]
        assert first.filepath == "A.tif"
        rest = list(it)
        assert len(rest) == 1
        assert batches == [1, 1]

    def test_relation_df_not_called(
        self, parquet_two_tiles: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cat = DuckDBGeoCatalog.open(parquet_two_tiles)
        relation = cat.relation

        def boom() -> None:  # pragma: no cover - failure path
            raise AssertionError("iter_rows must not materialise via .df()")

        monkeypatch.setattr(type(relation), "df", lambda self: boom(), raising=False)
        assert len(list(cat.iter_rows())) == 2

    def test_empty_relation(self, parquet_two_tiles: Path) -> None:
        cat = DuckDBGeoCatalog.open(parquet_two_tiles)
        empty = cat.query(
            GeoSlice(
                bounds=(1_000.0, 1_000.0, 1_100.0, 1_100.0),
                interval=pd.Interval(
                    pd.Timestamp("2030-01-01"),
                    pd.Timestamp("2030-01-02"),
                    closed="both",
                ),
                resolution=(10.0, 10.0),
                crs="EPSG:32629",
            )
        )
        assert list(empty.iter_rows()) == []

    def test_rejects_nonpositive_batch_size(self, parquet_two_tiles: Path) -> None:
        cat = DuckDBGeoCatalog.open(parquet_two_tiles)
        with pytest.raises(ValueError, match="batch_size"):
            next(cat.iter_rows(batch_size=0))


class TestIntersectSymmetryDuckDB:
    """The gh #40 canonical-order fix applies to the SQL engine too."""

    @staticmethod
    def _sliver_pair() -> tuple[DuckDBGeoCatalog, DuckDBGeoCatalog]:
        def one(box: shapely.geometry.base.BaseGeometry, fp: str) -> DuckDBGeoCatalog:
            gdf = gpd.GeoDataFrame(
                {
                    "geometry": [box],
                    "start_time": [pd.Timestamp("2000-01-01")],
                    "end_time": [pd.Timestamp("2000-01-01")],
                    "filepath": [fp],
                },
                geometry="geometry",
                crs="EPSG:4326",
            )
            return DuckDBGeoCatalog.from_memory(
                InMemoryGeoCatalog(gdf, backend="raster")
            )

        left = one(shapely.geometry.box(-1, -6.5, 0, 0), "left.tif")
        right = one(
            shapely.geometry.box(
                -3.8005323668172852e-165, -3, 1.875, 1.8113965363604467e-218
            ),
            "right.tif",
        )
        return left, right

    def test_sliver_overlap_cardinality_symmetric(self) -> None:
        left, right = self._sliver_pair()
        assert len(left.intersect(right)) == len(right.intersect(left))
