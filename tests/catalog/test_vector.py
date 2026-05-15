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

    def test_mixed_crs_inputs_reproject_to_uniform_crs(self, tmp_path: Path) -> None:
        """Regression for the §10.1-style footgun where mixed-CRS vector
        inputs (without an explicit `target_crs`) produced footprints in
        different CRSs but all stored under the first file's CRS tag.
        After the fix the first observed CRS becomes the catalog CRS and
        every later file gets reprojected to it, so footprints are
        coherent and queries against the catalog work.
        """
        # File A in UTM 29N (Iberia).
        a_gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    shapely.geometry.box(500_000, 4_000_000, 500_320, 4_000_320),
                ],
            },
            crs="EPSG:32629",
        )
        a_path = tmp_path / "a_20240115.gpkg"
        a_gdf.to_file(a_path, driver="GPKG")
        # File B in EPSG:4326, slightly north of File A in real-world
        # terms — solidly inside the UTM 29N zone (~37°N, 9°W).
        b_gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    shapely.geometry.box(-9.0, 37.05, -8.997, 37.052),
                ],
            },
            crs="EPSG:4326",
        )
        b_path = tmp_path / "b_20240116.gpkg"
        b_gdf.to_file(b_path, driver="GPKG")

        catalog = build_vector_catalog(
            [a_path, b_path],
            filename_regex=r"[ab]_(?P<date>\d{8})\.gpkg",
        )
        # First file's CRS wins; second file's footprint was reprojected.
        assert catalog.gdf.crs == "EPSG:32629"
        assert len(catalog) == 2
        # Both footprints should live in UTM 29N coordinate space — i.e.
        # the second file's polygon is *not* still in degrees.
        for geom in catalog.gdf.geometry:
            xmin, ymin, _xmax, _ymax = geom.bounds
            # UTM 29N over Iberia: easting ~400k-700k, northing ~4M.
            assert 400_000 < xmin < 700_000, f"xmin {xmin} not in UTM range"
            assert 3_900_000 < ymin < 4_200_000, f"ymin {ymin} not in UTM range"


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

    def test_works_when_layer_column_missing(self, vector_file: Path) -> None:
        """Regression for the bug where `filtered.gdf.get('layer')`
        returned `None` for externally constructed catalogs lacking the
        column, then crashed `zip(filepaths, None)` with TypeError.
        """
        catalog = build_vector_catalog([vector_file], filename_regex=REGEX)
        # Simulate an externally produced catalog by dropping the column.
        catalog.gdf.drop(columns=["layer"], inplace=True)
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
        # Should not raise TypeError despite the missing column.
        tensor = load_vector(
            catalog, sl, task="semantic_segmentation", label_field="class_id"
        )
        assert tensor.values.shape == (1, 32, 32)
