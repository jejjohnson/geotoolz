"""Tests for `build_vector_catalog` + `load_vector`."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry

from geotoolz.catalog import build_vector_catalog, load_vector
from geotoolz.types import GeoSlice


@pytest.fixture
def vector_file(tmp_path: Path) -> Path:
    """Two labelled polygons in EPSG:32629."""
    gdf = gpd.GeoDataFrame(
        {
            "class_id": [1, 2],
            "geometry": [
                shapely.geometry.box(500_000, 4_000_000, 500_160, 4_000_160),
                shapely.geometry.box(500_160, 4_000_160, 500_320, 4_000_320),
            ],
        },
        crs="EPSG:32629",
    )
    path = tmp_path / "labels_20240115.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


REGEX = r"labels_(?P<date>\d{8})\.gpkg"


class TestBuildVectorCatalog:
    def test_one_file(self, vector_file: Path) -> None:
        catalog = build_vector_catalog([vector_file], filename_regex=REGEX)
        assert len(catalog) == 1
        assert catalog.backend == "vector"

    def test_no_regex(self, vector_file: Path) -> None:
        catalog = build_vector_catalog([vector_file], filename_regex=None)
        assert len(catalog) == 1


class TestLoadVector:
    def test_semantic_segmentation(self, vector_file: Path) -> None:
        catalog = build_vector_catalog([vector_file], filename_regex=REGEX)
        sl = GeoSlice(
            bounds=(500_000, 4_000_000, 500_320, 4_000_320),
            interval=pd.Interval(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-31"),
                closed="both",
            ),
            resolution=(10.0, 10.0),
            crs="EPSG:32629",
        )
        tensor = load_vector(
            catalog, sl, task="semantic_segmentation", label_field="class_id"
        )
        assert tensor.values.shape == (1, 32, 32)
        # We rasterised two classes; their IDs should appear in the output.
        unique = set(np.unique(tensor.values).tolist())
        # Background is 0; classes 1 and 2 land in the raster.
        assert {1, 2}.issubset(unique)

    def test_object_detection_not_implemented(self, vector_file: Path) -> None:
        catalog = build_vector_catalog([vector_file], filename_regex=REGEX)
        sl = GeoSlice(
            bounds=(500_000, 4_000_000, 500_320, 4_000_320),
            interval=pd.Interval(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-31"),
                closed="both",
            ),
            resolution=(10.0, 10.0),
            crs="EPSG:32629",
        )
        with pytest.raises(NotImplementedError):
            load_vector(catalog, sl, task="object_detection")
