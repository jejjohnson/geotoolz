"""One time-zone policy across constructors, queries and backends (#231).

Catalogs hold times naive in UTC. Naive inputs are UTC by contract;
tz-aware inputs in any zone are converted. Every combination of catalog
zone, query zone and backend must select the same rows and report the
same naive-UTC intervals — including a TIMESTAMPTZ GeoParquet artifact
read by DuckDB under a non-UTC session time zone.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely

from geocatalog import GeoCatalog, GeoSlice, InMemoryGeoCatalog, to_geoparquet


# Row: 12:00-14:00 in the catalog's zone. Madrid is UTC+1 in January,
# so the Madrid row is 11:00-13:00 UTC; a query of 11:00-12:30 UTC hits
# every variant.
EXPECTED_UTC = {
    None: (pd.Timestamp("2024-01-01T12:00"), pd.Timestamp("2024-01-01T14:00")),
    "UTC": (pd.Timestamp("2024-01-01T12:00"), pd.Timestamp("2024-01-01T14:00")),
    "Europe/Madrid": (
        pd.Timestamp("2024-01-01T11:00"),
        pd.Timestamp("2024-01-01T13:00"),
    ),
}
QUERIES = {
    "naive": ("2024-01-01T11:00", "2024-01-01T12:30", None),
    "utc": ("2024-01-01T11:00", "2024-01-01T12:30", "UTC"),
    # Same instants as "utc", expressed in UTC-5.
    "new_york": ("2024-01-01T06:00", "2024-01-01T07:30", "America/New_York"),
}


def _gdf(tz: str | None) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "geometry": [shapely.box(0, 0, 1, 1)],
            "start_time": [pd.Timestamp("2024-01-01T12:00", tz=tz)],
            "end_time": [pd.Timestamp("2024-01-01T14:00", tz=tz)],
            "filepath": ["a.tif"],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _query_interval(name: str) -> pd.Interval:
    lo, hi, tz = QUERIES[name]
    return pd.Interval(pd.Timestamp(lo, tz=tz), pd.Timestamp(hi, tz=tz), closed="both")


def _catalog(backend: str, tz: str | None, tmp_path: Path) -> GeoCatalog:
    mem = InMemoryGeoCatalog(_gdf(tz), backend="raster")
    if backend == "memory":
        return mem
    pytest.importorskip("duckdb")
    from geocatalog import open_catalog

    path = tmp_path / "cat.parquet"
    to_geoparquet(mem, path)
    return open_catalog(path, engine="duckdb")


@pytest.mark.parametrize("backend", ["memory", "duckdb"])
@pytest.mark.parametrize("cat_tz", list(EXPECTED_UTC))
@pytest.mark.parametrize("query_name", list(QUERIES))
def test_query_matches_for_every_zone_combination(
    backend: str, cat_tz: str | None, query_name: str, tmp_path: Path
) -> None:
    cat = _catalog(backend, cat_tz, tmp_path)
    out = cat.query(time=_query_interval(query_name))
    assert len(out) == 1
    expected = pd.Interval(*EXPECTED_UTC[cat_tz], closed="both")
    assert out.temporal_extent == expected
    (row,) = list(out.iter_rows())
    assert row.interval == expected
    assert row.interval.left.tzinfo is None


def test_inmemory_constructor_stores_naive_utc() -> None:
    cat = InMemoryGeoCatalog(_gdf("Europe/Madrid"), backend="raster")
    assert cat.gdf.index.left.tz is None
    assert cat.gdf["start_time"].dt.tz is None
    assert cat.gdf["start_time"].iloc[0] == pd.Timestamp("2024-01-01T11:00")


def test_inmemory_constructor_accepts_mixed_zone_column() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.box(0, 0, 1, 1), shapely.box(1, 1, 2, 2)],
            "start_time": [
                pd.Timestamp("2024-01-01T12:00"),
                pd.Timestamp("2024-01-01T13:00", tz="Europe/Madrid"),
            ],
            "end_time": [
                pd.Timestamp("2024-01-01T14:00"),
                pd.Timestamp("2024-01-01T15:00", tz="Europe/Madrid"),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    cat = InMemoryGeoCatalog(gdf, backend="raster")
    assert list(cat.gdf.index.left) == [
        pd.Timestamp("2024-01-01T12:00"),
        pd.Timestamp("2024-01-01T12:00"),
    ]


def test_union_of_naive_and_aware_catalogs() -> None:
    naive = InMemoryGeoCatalog(_gdf(None), backend="raster")
    aware = InMemoryGeoCatalog(_gdf("Europe/Madrid"), backend="raster")
    merged = naive.union(aware)
    assert len(merged) == 2
    assert merged.temporal_extent == pd.Interval(
        pd.Timestamp("2024-01-01T11:00"),
        pd.Timestamp("2024-01-01T14:00"),
        closed="both",
    )


def test_geoslice_aware_equals_naive_utc() -> None:
    aware = GeoSlice(
        (0.0, 0.0, 1.0, 1.0), _query_interval("new_york"), (0.1, 0.1), "EPSG:4326"
    )
    naive = GeoSlice(
        (0.0, 0.0, 1.0, 1.0), _query_interval("naive"), (0.1, 0.1), "EPSG:4326"
    )
    assert aware.interval.left.tzinfo is None
    assert aware == naive
    assert hash(aware) == hash(naive)


def test_duckdb_timestamptz_artifact_ignores_session_zone(tmp_path: Path) -> None:
    """A TIMESTAMPTZ artifact must not be read through the host's zone."""
    duckdb = pytest.importorskip("duckdb")
    del duckdb
    from geocatalog import open_catalog

    path = tmp_path / "cat.parquet"
    to_geoparquet(InMemoryGeoCatalog(_gdf(None), backend="raster"), path)
    # Rewrite the time columns as timestamp[us, UTC], as an external tool
    # or an older geocatalog writer would have.
    table = pq.read_table(path)
    for name in ("start_time", "end_time"):
        i = table.schema.get_field_index(name)
        table = table.set_column(
            i, name, table.column(name).cast(pa.timestamp("us", tz="UTC"))
        )
    pq.write_table(table, path)

    cat = open_catalog(path, engine="duckdb")
    assert cat.con is not None
    cat.con.execute("SET TimeZone='America/New_York'")
    for name in QUERIES:
        out = cat.query(time=_query_interval(name))
        assert len(out) == 1, name
    (row,) = list(cat.iter_rows())
    assert row.interval == pd.Interval(*EXPECTED_UTC[None], closed="both")
    assert cat.temporal_extent == pd.Interval(*EXPECTED_UTC[None], closed="both")


def test_bundle_ingest_aware_source_rows_into_naive_catalog() -> None:
    from geocatalog._src.sources._base import SourceRow
    from geocatalog.bundle import CatalogBundle

    class _Src:
        name = "fake"

        def query(
            self, bounds: object, interval: object = None, **_: object
        ) -> Iterator[SourceRow]:
            yield SourceRow(
                id="b",
                source="fake",
                collection="c",
                geometry=shapely.box(0, 0, 1, 1),
                interval=pd.Interval(
                    pd.Timestamp(datetime(2024, 1, 1, 12, tzinfo=UTC)),
                    pd.Timestamp(datetime(2024, 1, 1, 13, tzinfo=UTC)),
                    closed="both",
                ),
                assets={"data": "b.tif"},
                properties={},
            )

    naive = InMemoryGeoCatalog(
        _gdf(None).assign(id=["a"], source=["local"], collection=["c"]),
        backend="raster",
    )
    bundle = CatalogBundle.from_catalog(naive)
    bundle.ingest(_Src(), bounds=(-1, -1, 2, 2))  # type: ignore[arg-type]
    assert bundle.n_items == 2
    assert bundle.catalog.gdf.index.left.tz is None
