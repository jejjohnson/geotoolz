"""Benchmarks for the in-memory catalog backend (#21).

Tracks query, intersect, and union latency at 10⁴ rows. Larger
sizes (10⁵+) belong on the DuckDB backend; the in-memory backend's
design ceiling is `O(n)` for many ops, so a 10⁴-row baseline catches
regressions in the asymptote we care about.
"""

from __future__ import annotations

import pytest

from geocatalog import InMemoryGeoCatalog, intersect, query, union

from .conftest import make_inmemory_catalog


_N_QUERY = 10_000
# Intersect uses GeoPandas' spatial index, but stays at 10³x10³ so the
# bench-quick CI job remains under the ~30 s budget the issue specifies.
_N_INTERSECT = 1_000


@pytest.fixture(scope="module")
def small_catalog() -> InMemoryGeoCatalog:
    """A 10⁴-row in-memory catalog reused across query/union benches."""
    return make_inmemory_catalog(_N_QUERY, seed=0)


@pytest.fixture(scope="module")
def small_catalog_b() -> InMemoryGeoCatalog:
    """A second 10⁴-row catalog with a different seed for union."""
    return make_inmemory_catalog(_N_QUERY, seed=1)


@pytest.fixture(scope="module")
def tiny_catalog() -> InMemoryGeoCatalog:
    """A 10³-row catalog used by the heavier intersect/overlay bench."""
    return make_inmemory_catalog(_N_INTERSECT, seed=0)


@pytest.fixture(scope="module")
def tiny_catalog_b() -> InMemoryGeoCatalog:
    """A second 10³-row catalog with a different seed for intersect."""
    return make_inmemory_catalog(_N_INTERSECT, seed=1)


def test_query_small_aoi(benchmark, small_catalog) -> None:
    """Point-AOI query — exercises the GeoPandas R-tree + IntervalIndex path."""
    benchmark(
        query,
        small_catalog,
        bounds=(0.0, 0.0, 1.0, 1.0),
        crs="EPSG:4326",
    )


def test_intersect_two_catalogs(benchmark, tiny_catalog, tiny_catalog_b) -> None:
    """Cross-catalog AND between two 10³-row catalogs."""
    benchmark(intersect, tiny_catalog, tiny_catalog_b)


def test_union_two_catalogs(benchmark, small_catalog, small_catalog_b) -> None:
    """Concatenate two 10⁴-row catalogs into one virtual dataset."""
    benchmark(union, small_catalog, small_catalog_b)
