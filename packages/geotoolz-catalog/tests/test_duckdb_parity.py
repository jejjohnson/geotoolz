"""DuckDB and InMemory backends agree row for row (#225)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import shapely


pytest.importorskip("duckdb")

from geocatalog import (
    CatalogSchemaError,
    DuckDBGeoCatalog,
    InMemoryGeoCatalog,
    open_catalog,
    to_geoparquet,
)
from geocatalog._src import duckdb_backend


def _cat(rows: list[dict], crs: str = "EPSG:32629") -> InMemoryGeoCatalog:
    base = {
        "start_time": pd.Timestamp("2024-01-01"),
        "end_time": pd.Timestamp("2024-01-03"),
    }
    gdf = gpd.GeoDataFrame([{**base, **r} for r in rows], geometry="geometry", crs=crs)
    return InMemoryGeoCatalog(gdf, backend="raster")


def _summary(cat: object) -> list[tuple[str, str, str | None]]:
    gdf = cat.gdf  # type: ignore[attr-defined]
    return sorted(
        (str(fp), shapely.normalize(geom).wkt, cloud)
        for fp, geom, cloud in zip(
            gdf["filepath"], gdf.geometry, gdf["cloud"], strict=True
        )
    )


LEFT = [
    {"geometry": shapely.box(0, 0, 2, 2), "filepath": "l1", "cloud": "low"},
    {"geometry": shapely.box(10, 0, 12, 2), "filepath": "l2", "cloud": "high"},
    {"geometry": shapely.box(20, 0, 22, 2), "filepath": "l3", "cloud": "mid"},
]
RIGHT = [
    # Overlaps l1 in an area and touches its left edge: GeometryCollection clip.
    {
        "geometry": shapely.MultiPolygon(
            [shapely.box(1, 0, 3, 2), shapely.box(-1, 0, 0, 2)]
        ),
        "filepath": "r1",
        "label": "a",
    },
    # Only touches l2 along an edge: a LineString clip, dropped by InMemory.
    {"geometry": shapely.box(12, 0, 14, 2), "filepath": "r2", "label": "b"},
    # Plain overlap with l3.
    {"geometry": shapely.box(21, 1, 23, 3), "filepath": "r3", "label": "c"},
]


class TestIntersectParity:
    def test_rows_geometry_and_extras_match(self) -> None:
        left, right = _cat(LEFT), _cat(RIGHT)
        mem = left.intersect(right)
        duck = DuckDBGeoCatalog.from_memory(left).intersect(right)
        assert _summary(duck.materialize()) == _summary(mem)
        assert {"l1", "l3"} == {r.filepath for r in duck.iter_rows()}

    def test_right_columns_are_prefixed(self) -> None:
        out = DuckDBGeoCatalog.from_memory(_cat(LEFT)).intersect(_cat(RIGHT))
        gdf = out.materialize().gdf
        assert {"cloud", "_right_filepath", "_right_label"} <= set(gdf.columns)
        assert "_right_start_time" not in gdf.columns

    def test_single_part_clip_is_not_multi(self) -> None:
        out = DuckDBGeoCatalog.from_memory(_cat(LEFT[:1])).intersect(_cat(RIGHT[:1]))
        (row,) = list(out.iter_rows())
        assert row.geometry.geom_type == "Polygon"

    def test_intersect_without_filepath_column(self) -> None:
        left = _cat([{"geometry": shapely.box(0, 0, 2, 2)}])
        right = _cat([{"geometry": shapely.box(1, 1, 3, 3)}])
        out = DuckDBGeoCatalog.from_memory(left).intersect(right)
        assert len(out) == 1


def test_iter_rows_filepath_fallback_matches() -> None:
    mem = _cat([{"geometry": shapely.box(0, 0, 1, 1)}])
    duck = DuckDBGeoCatalog.from_memory(mem)
    assert [r.filepath for r in duck.iter_rows()] == [
        r.filepath for r in mem.iter_rows()
    ]


def test_get_config_crs_matches(tmp_path: Path) -> None:
    path = tmp_path / "c.parquet"
    to_geoparquet(_cat(LEFT, crs="epsg:32629"), path)
    mem = open_catalog(path, engine="memory").get_config()
    duck = open_catalog(path, engine="duckdb").get_config()
    assert mem["crs"] == duck["crs"] == "EPSG:32629"
    assert (mem["engine"], duck["engine"]) == ("memory", "duckdb")


class TestGlobCrs:
    def test_glob_reads_crs_from_first_shard(self, tmp_path: Path) -> None:
        for i in range(2):
            to_geoparquet(_cat(LEFT), tmp_path / f"shard_{i}.parquet")
        cat = DuckDBGeoCatalog.open(str(tmp_path / "shard_*.parquet"))
        assert cat.crs.to_epsg() == 32629

    def test_glob_without_matches_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No files match"):
            duckdb_backend._read_geoparquet_crs(
                str(tmp_path / "missing_*.parquet"), default="EPSG:4326"
            )


def test_unversioned_artifact_needs_migration_on_both_engines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `_schema_version` column means legacy v0 for both engines."""
    path = tmp_path / "legacy.parquet"
    _cat(LEFT).gdf.reset_index(drop=True).to_parquet(path)
    from geocatalog._src import parquet

    monkeypatch.setattr(parquet, "SCHEMA_VERSION_CURRENT", 1)
    with pytest.raises(CatalogSchemaError, match="geocatalog migrate"):
        DuckDBGeoCatalog.open(path, backend="raster")
    with pytest.raises(CatalogSchemaError, match="missing migration v0"):
        open_catalog(path, engine="memory", backend="raster")


class TestOpenCatalogFactory:
    @pytest.fixture
    def path(self, tmp_path: Path) -> Path:
        p = tmp_path / "c.parquet"
        to_geoparquet(_cat(LEFT), p)
        return p

    def test_unknown_engine_rejected(self, path: Path) -> None:
        with pytest.raises(ValueError, match="engine must be one of"):
            open_catalog(path, engine="duckbd")  # type: ignore[arg-type]

    def test_empty_storage_options_keep_auto_engine(self, path: Path) -> None:
        assert isinstance(open_catalog(path, storage_options={}), DuckDBGeoCatalog)

    def test_memory_engine_applies_crs_override(self, path: Path) -> None:
        cat = open_catalog(path, engine="memory", crs="EPSG:32630")
        assert cat.crs.to_epsg() == 32630

    def test_auto_falls_back_without_spatial_extension(
        self, path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(duckdb_backend, "_ensure_spatial", lambda con: False)
        with pytest.warns(UserWarning, match="spatial extension is unavailable"):
            cat = open_catalog(path)
        assert isinstance(cat, InMemoryGeoCatalog)
