"""A DuckDB catalog can be shared across threads (#224).

A DuckDBPyConnection is not thread-safe, and any query started on it
closes a result that is still streaming. Before the fix, concurrent
`query` / `iter_rows` / `materialize` / `len` from worker threads (the
path `aload_raster` takes via `asyncio.to_thread`) failed with "Query
Stream is closed" or "closed pending query result".
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
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


N_ROWS = 500


def _mem(offset: float = 0.0) -> InMemoryGeoCatalog:
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                shapely.box(i + offset, 0, i + offset + 1, 1) for i in range(N_ROWS)
            ],
            "start_time": [pd.Timestamp("2024-01-01")] * N_ROWS,
            "end_time": [pd.Timestamp("2024-01-02")] * N_ROWS,
            "filepath": [f"f{i}" for i in range(N_ROWS)],
        },
        geometry="geometry",
        crs="EPSG:32629",
    )
    return InMemoryGeoCatalog(gdf, backend="raster")


@pytest.fixture
def opened(tmp_path: Path) -> DuckDBGeoCatalog:
    path = tmp_path / "cat.parquet"
    to_geoparquet(_mem(), path)
    cat = open_catalog(path, engine="duckdb")
    assert isinstance(cat, DuckDBGeoCatalog)
    return cat


def test_concurrent_query_iterate_materialize(opened: DuckDBGeoCatalog) -> None:
    def work(i: int) -> tuple[int, int, int]:
        lo = (i * 7) % (N_ROWS - 40)
        q = opened.query(bounds=(lo + 0.5, 0.2, lo + 20.5, 0.8), crs="EPSG:32629")
        streamed = sum(1 for _ in q.iter_rows(batch_size=3))
        return streamed, len(q.materialize()), len(q)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(64), timeout=120))
    assert all(r == (21, 21, 21) for r in results), results


def test_concurrent_aggregates_on_derived_catalogs(opened: DuckDBGeoCatalog) -> None:
    derived = [
        opened.query(bounds=(i * 10.0, 0, i * 10.0 + 5, 1), crs="EPSG:32629")
        for i in range(16)
    ]

    def work(cat: DuckDBGeoCatalog) -> tuple[float, ...]:
        return (*cat.total_bounds, float(cat.temporal_extent is not None))

    with ThreadPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(work, derived, timeout=120))
    assert all(r[-1] == 1.0 for r in out)


def test_opposite_direction_joins_do_not_deadlock() -> None:
    """a.intersect(b) and b.intersect(a) on different connections, together."""
    a = DuckDBGeoCatalog.from_memory(_mem())
    b = DuckDBGeoCatalog.from_memory(_mem(offset=0.5))

    def ab(_: int) -> int:
        return len(a.intersect(b, spatial_only=True))

    def ba(_: int) -> int:
        return len(b.intersect(a, spatial_only=True))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(ab if i % 2 else ba, i) for i in range(12)]
        counts = [f.result(timeout=120) for f in futures]
    assert len(set(counts)) == 1
    assert counts[0] > 0
