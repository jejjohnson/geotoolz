"""Tests for the GeoParquet roundtrip."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
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
        """The GeoParquet 1.1 covering bbox is written, declared and correct."""
        cat = _toy_catalog()
        path = tmp_path / "cat.parquet"
        to_geoparquet(cat, path, write_covering_bbox=True)

        table = pq.read_table(path)
        bbox = table.column("bbox").to_pylist()
        assert bbox == [
            {"xmin": 0.0, "ymin": 0.0, "xmax": 100.0, "ymax": 100.0},
            {"xmin": 200.0, "ymin": 0.0, "xmax": 300.0, "ymax": 100.0},
        ]
        geo = json.loads(table.schema.metadata[b"geo"])
        assert geo["columns"]["geometry"]["covering"]["bbox"]["xmin"] == [
            "bbox",
            "xmin",
        ]

        recovered = from_geoparquet(path)
        assert len(recovered) == len(cat)
        # The covering column is storage, not a catalog extra.
        assert all("bbox" not in row.extras for row in recovered.iter_rows())
