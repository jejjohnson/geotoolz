"""Tests for `InMemoryGeoCatalog` — query, intersect, union, iter_slices."""

from __future__ import annotations

from collections import Counter

import geopandas as gpd
import pandas as pd
import pytest
import shapely
import shapely.geometry

from geocatalog import InMemoryGeoCatalog


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


class TestMemoryOnly:
    def test_intersect_overlay_engine_matches_sjoin(
        self, two_tile_catalog: InMemoryGeoCatalog
    ) -> None:
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
        default = two_tile_catalog.intersect(other)
        legacy = two_tile_catalog.intersect(other, engine="overlay")

        assert len(default) == len(legacy)
        # ``set`` would mask duplicate-row multiplicity; geometries aren't
        # hashable so key by normalised WKB hex inside a ``Counter``.
        assert Counter(
            shapely.normalize(g).wkb_hex for g in default.gdf.geometry
        ) == Counter(shapely.normalize(g).wkb_hex for g in legacy.gdf.geometry)
        assert Counter(default.gdf.index) == Counter(legacy.gdf.index)

    def test_intersect_rejects_unknown_engine(
        self, two_tile_catalog: InMemoryGeoCatalog
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported intersect engine"):
            two_tile_catalog.intersect(two_tile_catalog, engine="missing")  # type: ignore[arg-type]

    def test_does_not_leak_bbox_or_schema_metadata_into_extras(self) -> None:
        """Regression for the P2 bug where the GeoParquet 1.1 ``bbox``
        covering struct and underscore-prefixed schema columns
        (e.g. ``_internal``) leaked into ``CatalogRow.extras``, where
        they'd flow into downstream consumers like STAC export.
        Mirrors `DuckDBGeoCatalog.iter_rows`'s filter list.
        """
        cat = _build(
            [
                {
                    "geometry": shapely.geometry.box(0, 0, 100, 100),
                    "start_time": pd.Timestamp("2024-01-01"),
                    "end_time": pd.Timestamp("2024-01-02"),
                    "filepath": "tile.tif",
                    # GeoParquet 1.1 bbox covering column.
                    "bbox": {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
                    # Underscore-prefixed schema column.
                    "_internal": "secret",
                    # Real user column — must survive.
                    "eo:cloud_cover": 7.5,
                },
            ]
        )

        rows = list(cat.iter_rows())
        assert len(rows) == 1
        extras = rows[0].extras
        assert "bbox" not in extras
        assert "_internal" not in extras
        assert not any(k.startswith("_") for k in extras)
        assert extras["eo:cloud_cover"] == 7.5

    @pytest.mark.parametrize("engine", ["sjoin", "overlay"])
    def test_geometry_collection_clip_keeps_polygon_part(self, engine: str) -> None:
        left = _build([_row(shapely.box(0, 0, 2, 2), "2024-01-01", "2024-01-03", "l")])
        # Overlaps [1, 2] x [0, 2] and touches the left edge at x = 0, so the
        # clip is GeometryCollection(Polygon, LineString).
        right_geom = shapely.MultiPolygon(
            [shapely.box(1, 0, 3, 2), shapely.box(-1, 0, 0, 2)]
        )
        right = _build([_row(right_geom, "2024-01-01", "2024-01-03", "r")])
        out = left.intersect(right, engine=engine)  # type: ignore[arg-type]
        assert len(out) == 1
        geom = out.gdf.geometry.iloc[0]
        assert geom.geom_type == "Polygon"
        assert geom.equals(shapely.box(1, 0, 2, 2))


class TestWhere:
    def test_filter_by_column(self, two_tile_catalog: InMemoryGeoCatalog) -> None:
        out = two_tile_catalog.where("filepath == 'tile_A.tif'")
        assert len(out) == 1


def _row(geom: shapely.Geometry, start: str, end: str, path: str) -> dict:
    return {
        "geometry": geom,
        "start_time": pd.Timestamp(start),
        "end_time": pd.Timestamp(end),
        "filepath": path,
    }


class TestConstructorValidation:
    def test_rejects_half_open_interval_index(self) -> None:
        idx = pd.IntervalIndex.from_arrays(
            [pd.Timestamp("2024-01-01")], [pd.Timestamp("2024-01-02")], closed="left"
        )
        gdf = gpd.GeoDataFrame(
            {"geometry": [shapely.box(0, 0, 1, 1)]},
            geometry="geometry",
            crs="EPSG:32629",
            index=idx,
        )
        with pytest.raises(ValueError, match="closed='both'"):
            InMemoryGeoCatalog(gdf, backend="raster")

    def test_rejects_unknown_backend_tag(
        self, two_tile_catalog: InMemoryGeoCatalog
    ) -> None:
        with pytest.raises(ValueError, match="backend must be one of"):
            InMemoryGeoCatalog(two_tile_catalog.gdf, backend="rastr")  # type: ignore[arg-type]
