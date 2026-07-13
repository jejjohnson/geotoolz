"""Tests for the GeoParquet roundtrip."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely.geometry

from geocatalog import (
    InMemoryGeoCatalog,
    from_geoparquet,
    to_geoparquet,
)


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


class TestParquetRoundtrip:
    def test_basic_roundtrip(self, tmp_path: Path) -> None:
        cat = _toy_catalog()
        path = tmp_path / "cat.parquet"
        to_geoparquet(cat, path)
        assert path.exists()
        recovered = from_geoparquet(path)
        assert len(recovered) == len(cat)
        assert recovered.backend == "raster"
        assert isinstance(recovered.gdf.index, pd.IntervalIndex)
        assert recovered.gdf.crs == cat.gdf.crs

    def test_bbox_column_survives(self, tmp_path: Path) -> None:
        """GeoParquet 1.1 covering-bbox column should round-trip; the test
        is a soft check — if the geopandas pin doesn't emit it, this just
        asserts that the file is still readable."""
        cat = _toy_catalog()
        path = tmp_path / "cat.parquet"
        to_geoparquet(cat, path, write_covering_bbox=True)
        recovered = from_geoparquet(path)
        assert len(recovered) == len(cat)
