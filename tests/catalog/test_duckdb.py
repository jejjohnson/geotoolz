"""Tests for the DuckDB-backed catalog (Phase 2). Skipped without [duckdb]."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import shapely.geometry


duckdb = pytest.importorskip("duckdb")

from geotoolz.catalog import (
    DuckDBGeoCatalog,
    InMemoryGeoCatalog,
    open_catalog,
    to_geoparquet,
)
from geotoolz.types import GeoSlice


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


class TestSqlEscape:
    def test_sql_filter(self, parquet_two_tiles: Path) -> None:
        duck = open_catalog(parquet_two_tiles, engine="duckdb")
        out = duck.sql("filepath = 'A.tif'")
        assert len(out) == 1
        assert out.materialize().gdf["filepath"].iloc[0] == "A.tif"
