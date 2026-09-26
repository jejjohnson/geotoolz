"""`DuckDBGeoCatalog.open` is lazy and `query` prunes with the bbox column (#221)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely


pytest.importorskip("duckdb")

from geocatalog import (
    DuckDBGeoCatalog,
    InMemoryGeoCatalog,
    open_catalog,
    to_geoparquet,
)


def _mem(n: int = 400, seed: int = 0) -> InMemoryGeoCatalog:
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 10_000, n), rng.uniform(0, 10_000, n)
    gdf = gpd.GeoDataFrame(
        {
            "geometry": shapely.box(x, y, x + 50, y + 50),
            "start_time": [pd.Timestamp("2024-01-01")] * n,
            "end_time": [pd.Timestamp("2024-01-02")] * n,
            "filepath": [f"f{i}" for i in range(n)],
        },
        geometry="geometry",
        crs="EPSG:32629",
    )
    return InMemoryGeoCatalog(gdf, backend="raster")


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    path = tmp_path / "cat.parquet"
    to_geoparquet(_mem(), path)
    return path


def _scan_section(plan: str) -> str:
    return plan.split("PARQUET_SCAN", 1)[-1]


def test_open_returns_unmaterialised_relation(artifact: Path) -> None:
    cat = DuckDBGeoCatalog.open(artifact)
    assert cat.relation.type != "MATERIALIZED_RELATION"


def test_query_pushes_bbox_predicate_into_parquet_scan(artifact: Path) -> None:
    cat = DuckDBGeoCatalog.open(artifact)
    out = cat.query(bounds=(1000.0, 1000.0, 3000.0, 3000.0), crs="EPSG:32629")
    scan = _scan_section(out.relation.explain())
    assert "Filters:" in scan
    assert "bbox" in scan


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_pushdown_results_match_inmemory(artifact: Path, seed: int) -> None:
    duck = open_catalog(artifact, engine="duckdb")
    mem = open_catalog(artifact, engine="memory")
    rng = np.random.default_rng(seed)
    for _ in range(10):
        x0, y0 = rng.uniform(0, 9000, 2)
        w, h = rng.uniform(10, 2000, 2)
        box = (x0, y0, x0 + w, y0 + h)
        got = {r.filepath for r in duck.query(bounds=box, crs="EPSG:32629").iter_rows()}
        want = {r.filepath for r in mem.query(bounds=box, crs="EPSG:32629").iter_rows()}
        assert got == want


def test_union_rows_without_bbox_are_not_filtered_out(artifact: Path) -> None:
    """Rows from an in-memory side have a NULL bbox after a union."""
    cat = DuckDBGeoCatalog.open(artifact)
    extra = InMemoryGeoCatalog(
        gpd.GeoDataFrame(
            {
                "geometry": [shapely.box(20_000, 20_000, 20_010, 20_010)],
                "start_time": [pd.Timestamp("2024-01-01")],
                "end_time": [pd.Timestamp("2024-01-02")],
                "filepath": ["extra"],
            },
            geometry="geometry",
            crs="EPSG:32629",
        ),
        backend="raster",
    )
    out = cat.union(extra).query(
        bounds=(19_000.0, 19_000.0, 21_000.0, 21_000.0), crs="EPSG:32629"
    )
    assert [r.filepath for r in out.iter_rows()] == ["extra"]


def test_partitioned_directory_still_opens(tmp_path: Path) -> None:
    mem = _mem(60)
    mem.gdf["year"] = 2024
    out_dir = tmp_path / "parts"
    to_geoparquet(mem, out_dir, partition_by=["year"])
    cat = DuckDBGeoCatalog.open(out_dir)
    assert len(cat) == 60
    assert len(cat.query(bounds=(0.0, 0.0, 10_000.0, 10_000.0), crs="EPSG:32629")) == 60
