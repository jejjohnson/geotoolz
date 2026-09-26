"""Derived DuckDB catalogs never change behind the caller's back (#222).

`from_memory` / `intersect` / `union` used to register named views keyed
on ``id()`` with ``replace=True``; derived relations resolved those
views by name at execution, so any later registration that reused an
id silently swapped their data, and views piled up on the connection.
"""

from __future__ import annotations

import gc

import geopandas as gpd
import pandas as pd
import pytest
import shapely


duckdb = pytest.importorskip("duckdb")

from geocatalog import DuckDBGeoCatalog, InMemoryGeoCatalog


def _mem(name: str, geom: shapely.Geometry, **extra: object) -> InMemoryGeoCatalog:
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [geom],
            "start_time": [pd.Timestamp("2024-01-01")],
            "end_time": [pd.Timestamp("2024-01-02")],
            "filepath": [name],
            **{k: [v] for k, v in extra.items()},
        },
        geometry="geometry",
        crs="EPSG:32629",
    )
    return InMemoryGeoCatalog(gdf, backend="raster")


def _user_views(con: duckdb.DuckDBPyConnection) -> int:
    return con.sql("SELECT count(*) FROM duckdb_views() WHERE NOT internal").fetchone()[
        0
    ]


def _rows(cat: DuckDBGeoCatalog) -> list[tuple[str, tuple[float, ...]]]:
    return sorted((r.filepath, r.geometry.bounds) for r in cat.iter_rows())


def test_set_algebra_creates_no_named_views() -> None:
    a = DuckDBGeoCatalog.from_memory(_mem("a", shapely.box(0, 0, 10, 10)))
    assert a.con is not None
    for i in range(25):
        a.intersect(_mem(f"i{i}", shapely.box(1, 1, 2, 2)))
        a.union(_mem(f"u{i}", shapely.box(20, 20, 21, 21)))
        DuckDBGeoCatalog.from_memory(_mem(f"m{i}", shapely.box(0, 0, 1, 1)), con=a.con)
    assert _user_views(a.con) == 0


def test_derived_catalogs_stable_across_later_operations() -> None:
    a = DuckDBGeoCatalog.from_memory(_mem("a", shapely.box(0, 0, 10, 10)))
    x = a.intersect(_mem("b", shapely.box(5, 5, 6, 6)))
    u = a.union(_mem("d", shapely.box(20, 20, 21, 21)))
    x_before, u_before = _rows(x), _rows(u)
    for i in range(30):
        a.intersect(_mem(f"c{i}", shapely.box(1, 1, 2, 2)))
        a.union(_mem(f"e{i}", shapely.box(30, 30, 31, 31)))
        gc.collect()
    assert _rows(x) == x_before
    assert _rows(u) == u_before


def test_derived_catalog_ignores_legacy_view_names() -> None:
    """Simulate the id collision directly: shadow every legacy view name."""
    a = DuckDBGeoCatalog.from_memory(_mem("a", shapely.box(0, 0, 10, 10)))
    assert a.con is not None
    b_mem = _mem("b", shapely.box(5, 5, 6, 6))
    b = DuckDBGeoCatalog.from_memory(b_mem, con=a.con)
    x = a.intersect(b)
    u = a.union(b)
    x_before, u_before, b_before = _rows(x), _rows(u), _rows(b)

    decoy = DuckDBGeoCatalog.from_memory(
        _mem("decoy", shapely.box(50, 50, 60, 60)), con=a.con
    )
    for name in (
        f"_geocatalog_left_{id(a):x}",
        f"_geocatalog_right_{id(b):x}",
        f"_geocatalog_unionL_{id(a):x}",
        f"_geocatalog_unionR_{id(b):x}",
    ):
        decoy.relation.create_view(name, replace=True)
    a.con.register(f"_geocatalog_mem_{id(b_mem):x}", decoy.relation.df())

    assert _rows(x) == x_before
    assert _rows(u) == u_before
    assert _rows(b) == b_before


def test_union_by_name_fills_missing_columns() -> None:
    left = DuckDBGeoCatalog.from_memory(_mem("l", shapely.box(0, 0, 1, 1), cloud=5.0))
    right = _mem("r", shapely.box(2, 2, 3, 3), layer="roads")
    out = left.union(right)
    rows = {r.filepath: r.extras for r in out.iter_rows()}
    assert rows["l"]["cloud"] == 5.0
    assert pd.isna(rows["l"]["layer"])
    assert rows["r"]["layer"] == "roads"
    assert pd.isna(rows["r"]["cloud"])
