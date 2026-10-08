"""Benchmarks for `DuckDBGeoCatalog` (#21).

Tracks predicate-pushdown query latency on a 10⁵-row GeoParquet
artifact. The whole module is gated on the `[duckdb]` extra — when
the extra isn't installed we skip rather than crash, so the
default bench job still runs cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from geocatalog.storage import to_geoparquet

from .conftest import make_inmemory_catalog


if TYPE_CHECKING:
    from geocatalog.backends import DuckDBGeoCatalog as _DuckDBGeoCatalog


# Probe the extra eagerly so we can `skipif` cleanly; resolving the
# public-surface attribute triggers the same friendly ImportError path
# the library exposes to library consumers (`geocatalog.__getattr__`).
try:
    from geocatalog.backends import DuckDBGeoCatalog

    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False
    DuckDBGeoCatalog = None  # type: ignore[assignment, misc]


pytestmark = pytest.mark.skipif(
    not _HAS_DUCKDB,
    reason="DuckDB extra not installed: pip install 'geotoolz-catalog[duckdb]'.",
)


_N_MEDIUM = 100_000
_N_LARGE = 300_000


def _archive(tmp_path_factory: pytest.TempPathFactory, n_rows: int) -> Path:
    path = tmp_path_factory.mktemp(f"bench_duckdb_{n_rows}") / "catalog.parquet"
    to_geoparquet(make_inmemory_catalog(n_rows, seed=0), path)
    return path


@pytest.fixture(scope="module")
def duckdb_catalog(
    tmp_path_factory: pytest.TempPathFactory,
) -> _DuckDBGeoCatalog:
    """Persist a 10⁵-row in-memory catalog as GeoParquet and reopen via DuckDB."""
    return DuckDBGeoCatalog.open(_archive(tmp_path_factory, _N_MEDIUM))


@pytest.fixture(scope="module")
def large_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 3x10⁵-row synthetic GeoParquet archive."""
    return _archive(tmp_path_factory, _N_LARGE)


def _fetch(catalog: _DuckDBGeoCatalog, **query: Any) -> list[tuple[Any, ...]]:
    """Run a query *and execute it*.

    `query` only builds a lazy relation; timing that alone measures plan
    construction, not the pushdown scan this suite tracks (#235).
    """
    return catalog.query(**query).relation.fetchall()


def test_duckdb_query_small_aoi(benchmark, duckdb_catalog) -> None:
    """Small-AOI query — exercises the GeoParquet 1.1 bbox-pushdown path."""
    benchmark(_fetch, duckdb_catalog, bounds=(0.0, 0.0, 1.0, 1.0), crs="EPSG:4326")


def test_duckdb_query_small_aoi_large_archive(benchmark, large_archive) -> None:
    """Small-AOI query on 3x10⁵ rows, executed end to end."""
    catalog = DuckDBGeoCatalog.open(large_archive)
    benchmark(_fetch, catalog, bounds=(0.0, 0.0, 1.0, 1.0), crs="EPSG:4326")


def test_duckdb_query_space_time_large_archive(benchmark, large_archive) -> None:
    """Spatial + temporal filter on 3x10⁵ rows, executed end to end."""
    catalog = DuckDBGeoCatalog.open(large_archive)
    benchmark(
        _fetch,
        catalog,
        bounds=(-5.0, -5.0, 5.0, 5.0),
        crs="EPSG:4326",
        time=("2022-01-01", "2022-03-31"),
    )


def test_duckdb_open_and_count_large_archive(benchmark, large_archive) -> None:
    """Open the archive lazily and count it — the first query a user runs."""
    benchmark(lambda: len(DuckDBGeoCatalog.open(large_archive)))
