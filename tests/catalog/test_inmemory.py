"""Tests for `InMemoryGeoCatalog` — query, intersect, union, iter_slices."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry

from geotoolz.catalog import InMemoryGeoCatalog, intersect, query, union
from geotoolz.types import GeoSlice


def _build(rows: list[dict], crs: str = "EPSG:32629") -> InMemoryGeoCatalog:
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return InMemoryGeoCatalog(gdf, backend="raster")


@pytest.fixture
def two_tile_catalog() -> InMemoryGeoCatalog:
    """Two non-overlapping tiles, same week."""
    return _build(
        [
            {
                "geometry": shapely.geometry.box(0, 0, 100, 100),
                "start_time": pd.Timestamp("2024-01-01"),
                "end_time": pd.Timestamp("2024-01-02"),
                "filepath": "tile_A.tif",
            },
            {
                "geometry": shapely.geometry.box(200, 0, 300, 100),
                "start_time": pd.Timestamp("2024-01-02"),
                "end_time": pd.Timestamp("2024-01-03"),
                "filepath": "tile_B.tif",
            },
        ]
    )


class TestConstruction:
    def test_promote_columns_to_interval_index(
        self, two_tile_catalog: InMemoryGeoCatalog
    ) -> None:
        assert isinstance(two_tile_catalog.gdf.index, pd.IntervalIndex)
        assert two_tile_catalog.gdf.index.closed == "both"

    def test_rejects_unset_crs(self) -> None:
        gdf = gpd.GeoDataFrame(
            {"geometry": [shapely.geometry.box(0, 0, 1, 1)]},
            geometry="geometry",
        )
        with pytest.raises(ValueError, match=r"gdf\.crs"):
            InMemoryGeoCatalog(gdf, backend="raster")

    def test_rejects_missing_time(self) -> None:
        gdf = gpd.GeoDataFrame(
            {"geometry": [shapely.geometry.box(0, 0, 1, 1)]},
            geometry="geometry",
            crs="EPSG:32629",
        )
        with pytest.raises(ValueError, match="IntervalIndex"):
            InMemoryGeoCatalog(gdf, backend="raster")


class TestProperties:
    def test_total_bounds(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        assert two_tile_catalog.total_bounds == (0.0, 0.0, 300.0, 100.0)

    def test_temporal_extent(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        ext = two_tile_catalog.temporal_extent
        assert ext.left == pd.Timestamp("2024-01-01")
        assert ext.right == pd.Timestamp("2024-01-03")

    def test_len(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        assert len(two_tile_catalog) == 2


class TestQuery:
    def test_query_filters_by_bbox(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        out = two_tile_catalog.query(bounds=(0, 0, 50, 50), crs="EPSG:32629")
        assert len(out) == 1
        assert out.gdf["filepath"].iloc[0] == "tile_A.tif"

    def test_query_filters_by_time(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        out = two_tile_catalog.query(time=("2024-01-02 12:00", "2024-01-03 12:00"))
        assert len(out) == 1
        assert out.gdf["filepath"].iloc[0] == "tile_B.tif"

    def test_query_by_slice(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        sl = GeoSlice(
            bounds=(0, 0, 50, 50),
            interval=pd.Interval(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                closed="both",
            ),
            resolution=(1.0, 1.0),
            crs="EPSG:32629",
        )
        out = two_tile_catalog.query(sl)
        assert len(out) == 1

    def test_query_in_wrong_crs_reprojects_internally(
        self, two_tile_catalog: InMemoryGeoCatalog
    ) -> None:
        """Regression test for §10.1 footgun: an AOI in EPSG:4326 must
        not silently return empty against a catalog in EPSG:32629."""
        # UTM 29N coords (50, 50) reproject to ≈ (-13.488, 0.00045) in 4326.
        # A small 4326 bbox around that point should match tile_A after
        # the catalog reprojects it back to UTM internally.
        out = two_tile_catalog.query(
            bounds=(-13.4885, 0.0001, -13.4880, 0.0008), crs="EPSG:4326"
        )
        assert len(out) == 1
        assert out.gdf["filepath"].iloc[0] == "tile_A.tif"

    def test_query_rejects_both_slice_and_parts(
        self, two_tile_catalog: InMemoryGeoCatalog
    ) -> None:
        sl = GeoSlice(
            bounds=(0, 0, 50, 50),
            interval=pd.Interval(0, 1, closed="both"),
            resolution=(1.0, 1.0),
            crs="EPSG:32629",
        )
        with pytest.raises(TypeError, match="either"):
            two_tile_catalog.query(sl, bounds=(0, 0, 50, 50))


class TestSetAlgebra:
    def test_intersect_spatiotemporal(
        self, two_tile_catalog: InMemoryGeoCatalog
    ) -> None:
        # Build a second catalog overlapping tile_A in both space and time.
        other = _build(
            [
                {
                    "geometry": shapely.geometry.box(50, 50, 250, 150),
                    "start_time": pd.Timestamp("2024-01-01"),
                    "end_time": pd.Timestamp("2024-01-04"),
                    "filepath": "labels.gpkg",
                },
            ]
        )
        joint = intersect(two_tile_catalog, other)
        # Both A and B share space with the labels row; both intersect in time.
        assert len(joint) == 2
        # Footprints are clipped:
        bounds_set = {tuple(g.bounds) for g in joint.gdf.geometry}
        assert (50.0, 50.0, 100.0, 100.0) in bounds_set  # A ∩ labels
        assert (200.0, 50.0, 250.0, 100.0) in bounds_set  # B ∩ labels

    def test_intersect_drops_temporal_mismatch(
        self, two_tile_catalog: InMemoryGeoCatalog
    ) -> None:
        other = _build(
            [
                {
                    "geometry": shapely.geometry.box(50, 50, 250, 150),
                    "start_time": pd.Timestamp("2030-01-01"),
                    "end_time": pd.Timestamp("2030-01-04"),
                    "filepath": "future.gpkg",
                },
            ]
        )
        joint = intersect(two_tile_catalog, other)
        assert len(joint) == 0

    def test_intersect_spatial_only(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        other = _build(
            [
                {
                    "geometry": shapely.geometry.box(50, 50, 250, 150),
                    "start_time": pd.Timestamp("2030-01-01"),
                    "end_time": pd.Timestamp("2030-01-04"),
                    "filepath": "static.gpkg",
                },
            ]
        )
        joint = intersect(two_tile_catalog, other, spatial_only=True)
        # Time mismatch ignored; both A and B clip against the labels footprint.
        assert len(joint) == 2

    def test_union(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        other = _build(
            [
                {
                    "geometry": shapely.geometry.box(400, 0, 500, 100),
                    "start_time": pd.Timestamp("2024-02-01"),
                    "end_time": pd.Timestamp("2024-02-02"),
                    "filepath": "tile_C.tif",
                },
            ]
        )
        all_three = union(two_tile_catalog, other)
        assert len(all_three) == 3

    def test_union_reprojects(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        # other is in EPSG:32630, which doesn't match the UTM 29N catalog —
        # union should silently reproject before concat.
        other = _build(
            [
                {
                    "geometry": shapely.geometry.box(
                        400_000, 4_000_000, 500_000, 4_100_000
                    ),
                    "start_time": pd.Timestamp("2024-02-01"),
                    "end_time": pd.Timestamp("2024-02-02"),
                    "filepath": "tile_C.tif",
                },
            ],
            crs="EPSG:32630",
        )
        merged = union(two_tile_catalog, other)
        assert len(merged) == 3
        assert merged.gdf.crs == two_tile_catalog.gdf.crs


class TestIterSlices:
    def test_yields_one_per_row(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        slices = list(two_tile_catalog.iter_slices(resolution=(10.0, 10.0)))
        assert len(slices) == 2
        for s in slices:
            assert isinstance(s, GeoSlice)
            assert s.resolution == (10.0, 10.0)

    def test_slice_bounds_match_footprints(
        self, two_tile_catalog: InMemoryGeoCatalog
    ) -> None:
        slices = list(two_tile_catalog.iter_slices(resolution=(10.0, 10.0)))
        np.testing.assert_allclose(slices[0].bounds, (0, 0, 100, 100))
        np.testing.assert_allclose(slices[1].bounds, (200, 0, 300, 100))


class TestWhere:
    def test_filter_by_column(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        out = two_tile_catalog.where("filepath == 'tile_A.tif'")
        assert len(out) == 1


class TestQueryFreeFunction:
    def test_delegates(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        out = query(two_tile_catalog, bounds=(0, 0, 50, 50), crs="EPSG:32629")
        assert len(out) == 1
