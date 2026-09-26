"""Antimeridian-crossing AOIs in `query()` on both backends (#229).

Rows are placed so the single "complement" envelope that a naive
``box(xmin, ymin, xmax, ymax)`` with ``xmin > xmax`` produces selects
exactly the wrong row: A and B sit just either side of 180°, C sits in
the same latitude band on the prime meridian.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pyproj
import pytest
import shapely

from geocatalog import GeoCatalog, InMemoryGeoCatalog


A_EAST = shapely.box(179.2, -20.0, 179.8, -15.0)
B_WEST = shapely.box(-179.8, -20.0, -179.2, -15.0)
C_FAR = shapely.box(0.0, -20.0, 10.0, -15.0)


def _mem() -> InMemoryGeoCatalog:
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [A_EAST, B_WEST, C_FAR],
            "start_time": [pd.Timestamp("2024-01-01")] * 3,
            "end_time": [pd.Timestamp("2024-01-02")] * 3,
            "filepath": ["a.tif", "b.tif", "c.tif"],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    return InMemoryGeoCatalog(gdf, backend="raster")


@pytest.fixture(params=["memory", "duckdb"])
def catalog(request: pytest.FixtureRequest) -> GeoCatalog:
    mem = _mem()
    if request.param == "memory":
        return mem
    pytest.importorskip("duckdb")
    from geocatalog import DuckDBGeoCatalog

    return DuckDBGeoCatalog.from_memory(mem)


def _filepaths(cat: GeoCatalog) -> set[str]:
    return {row.filepath for row in cat.iter_rows()}


def test_projected_aoi_crossing_180_matches_both_sides(catalog: GeoCatalog) -> None:
    # EPSG:3832 (PDC Mercator, centred at 150°E) is contiguous across 180°.
    to_3832 = pyproj.Transformer.from_crs(4326, 3832, always_xy=True)
    bounds_3832 = to_3832.transform_bounds(179.0, -20.0, 181.0, -15.0)
    out = catalog.query(bounds=bounds_3832, crs="EPSG:3832")
    assert _filepaths(out) == {"a.tif", "b.tif"}


def test_geographic_bounds_with_xmin_gt_xmax(catalog: GeoCatalog) -> None:
    out = catalog.query(bounds=(179.0, -20.0, -179.0, -15.0), crs="EPSG:4326")
    assert _filepaths(out) == {"a.tif", "b.tif"}


def test_ordinary_bounds_unchanged(catalog: GeoCatalog) -> None:
    out = catalog.query(bounds=(-1.0, -21.0, 11.0, -14.0), crs="EPSG:4326")
    assert _filepaths(out) == {"c.tif"}


def test_inverted_y_rejected(catalog: GeoCatalog) -> None:
    with pytest.raises(ValueError, match="ymin > ymax"):
        catalog.query(bounds=(0.0, -15.0, 10.0, -20.0), crs="EPSG:4326")


def test_inverted_x_in_projected_catalog_rejected() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.box(0, 0, 10, 10)],
            "start_time": [pd.Timestamp("2024-01-01")],
            "end_time": [pd.Timestamp("2024-01-02")],
        },
        geometry="geometry",
        crs="EPSG:32629",
    )
    cat = InMemoryGeoCatalog(gdf, backend="raster")
    with pytest.raises(ValueError, match="xmin > xmax"):
        cat.query(bounds=(10.0, 0.0, 0.0, 10.0), crs="EPSG:32629")
