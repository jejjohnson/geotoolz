"""Tests for the `CatalogDomain` ↔ `geotoolz.patch` bridge."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import shapely.geometry

from geotoolz.catalog import CatalogDomain, InMemoryGeoCatalog
from geotoolz.types import GeoSlice


def _toy_catalog() -> InMemoryGeoCatalog:
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
    return InMemoryGeoCatalog(gdf, backend="raster")


class TestCatalogDomain:
    def test_bounds_match_catalog(self) -> None:
        cat = _toy_catalog()
        dom = CatalogDomain(catalog=cat, resolution=(10.0, 10.0))
        assert dom.bounds == cat.total_bounds

    def test_len_match_catalog(self) -> None:
        cat = _toy_catalog()
        dom = CatalogDomain(catalog=cat, resolution=(10.0, 10.0))
        assert len(dom) == len(cat)

    def test_slices_match_rows(self) -> None:
        cat = _toy_catalog()
        dom = CatalogDomain(catalog=cat, resolution=(10.0, 10.0))
        slices = dom.slices()
        assert len(slices) == 2
        assert all(isinstance(s, GeoSlice) for s in slices)
        assert all(s.resolution == (10.0, 10.0) for s in slices)

    def test_get_config(self) -> None:
        cat = _toy_catalog()
        dom = CatalogDomain(catalog=cat, resolution=(10.0, 10.0))
        config = dom.get_config()
        assert config["resolution"] == (10.0, 10.0)
        assert config["catalog"]["backend"] == "raster"
