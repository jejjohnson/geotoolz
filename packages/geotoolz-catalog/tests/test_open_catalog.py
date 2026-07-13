"""Tests for the `open_catalog` factory."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import shapely.geometry

import geocatalog as gc


def _toy_catalog() -> gc.InMemoryGeoCatalog:
    gdf = gpd.GeoDataFrame(
        {
            "filepath": ["a.tif", "b.tif"],
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
        },
        geometry="geometry",
        crs="EPSG:32629",
    )
    return gc.InMemoryGeoCatalog(gdf, backend="raster")


@pytest.fixture
def parquet_path(tmp_path: Path) -> Path:
    cat = _toy_catalog()
    path = tmp_path / "cat.parquet"
    gc.to_geoparquet(cat, path)
    return path


class TestOpenCatalogMemoryEngine:
    def test_returns_in_memory_catalog(self, parquet_path: Path) -> None:
        cat = gc.open_catalog(parquet_path, engine="memory")
        assert isinstance(cat, gc.InMemoryGeoCatalog)
        assert len(cat) == 2

    def test_backend_override(self, parquet_path: Path) -> None:
        cat = gc.open_catalog(parquet_path, engine="memory", backend="vector")
        assert cat.backend == "vector"

    def test_backend_default_preserves_tag(self, parquet_path: Path) -> None:
        cat = gc.open_catalog(parquet_path, engine="memory")
        assert cat.backend == "raster"


class TestOpenCatalogAutoEngine:
    def test_returns_duckdb_when_extra_installed(self, parquet_path: Path) -> None:
        pytest.importorskip("duckdb")
        cat = gc.open_catalog(parquet_path)
        assert type(cat).__name__ == "DuckDBGeoCatalog"
        assert len(list(cat.iter_rows())) == 2


class TestOpenCatalogDuckDBEngine:
    def test_explicit_duckdb_engine(self, parquet_path: Path) -> None:
        pytest.importorskip("duckdb")
        cat = gc.open_catalog(parquet_path, engine="duckdb")
        assert type(cat).__name__ == "DuckDBGeoCatalog"
        assert len(list(cat.iter_rows())) == 2
