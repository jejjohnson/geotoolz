"""Tests for `InMemoryGeoCatalog` — query, intersect, union, iter_slices."""

from __future__ import annotations

from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import shapely.geometry

from geocatalog import GeoSlice, InMemoryGeoCatalog, intersect, query, union


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

    def test_intersect_sjoin_handles_invalid_geometry(self) -> None:
        # Bowtie self-intersecting polygon — would crash GEOS without the
        # ``make_valid`` repair mirrored from ``gpd.overlay``.
        bowtie = shapely.geometry.Polygon([(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)])
        assert not bowtie.is_valid
        left = _build(
            [
                {
                    "geometry": bowtie,
                    "start_time": pd.Timestamp("2024-01-01"),
                    "end_time": pd.Timestamp("2024-01-02"),
                    "filepath": "invalid.tif",
                },
            ]
        )
        right = _build(
            [
                {
                    "geometry": shapely.geometry.box(0, 0, 10, 10),
                    "start_time": pd.Timestamp("2024-01-01"),
                    "end_time": pd.Timestamp("2024-01-02"),
                    "filepath": "labels.gpkg",
                },
            ]
        )
        joined = left.intersect(right)
        assert len(joined) >= 1
        assert not joined.gdf.geometry.is_empty.any()
        assert joined.gdf.geometry.area.sum() > 0

    def test_intersect_drops_boundary_only_matches(self) -> None:
        left = _build(
            [
                {
                    "geometry": shapely.geometry.box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-01-01"),
                    "end_time": pd.Timestamp("2024-01-02"),
                    "filepath": "left.tif",
                },
            ]
        )
        right = _build(
            [
                {
                    "geometry": shapely.geometry.box(1, 0, 2, 1),
                    "start_time": pd.Timestamp("2024-01-01"),
                    "end_time": pd.Timestamp("2024-01-02"),
                    "filepath": "right.tif",
                },
            ]
        )

        assert len(left.intersect(right)) == 0

    def test_intersect_cardinality_symmetric_on_sliver_overlap(self) -> None:
        # Regression for gh #40: GEOS intersection is order-sensitive on
        # near-degenerate sliver overlaps (Polygon one way, empty the
        # other). The Hypothesis falsifying example reduced to this pair:
        # the overlap is a sliver ~4e-165 degrees wide.
        left = _build(
            [
                {
                    "geometry": shapely.geometry.box(-1, -6.5, 0, 0),
                    "start_time": pd.Timestamp("2000-01-01"),
                    "end_time": pd.Timestamp("2000-01-01"),
                    "filepath": "left.tif",
                },
            ]
        )
        right = _build(
            [
                {
                    "geometry": shapely.geometry.box(
                        -3.8005323668172852e-165, -3, 1.875, 1.8113965363604467e-218
                    ),
                    "start_time": pd.Timestamp("2000-01-01"),
                    "end_time": pd.Timestamp("2000-01-01"),
                    "filepath": "right.tif",
                },
            ]
        )

        assert len(left.intersect(right)) == len(right.intersect(left))

    def test_intersect_rejects_unknown_engine(
        self, two_tile_catalog: InMemoryGeoCatalog
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported intersect engine"):
            two_tile_catalog.intersect(two_tile_catalog, engine="missing")  # type: ignore[arg-type]

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


class TestIterRows:
    def test_yields_catalog_rows_in_order(
        self, two_tile_catalog: InMemoryGeoCatalog
    ) -> None:
        rows = list(two_tile_catalog.iter_rows())

        assert len(rows) == 2
        for i, row in enumerate(rows):
            assert row.filepath == two_tile_catalog.gdf["filepath"].iloc[i]
            assert row.geometry == two_tile_catalog.gdf.geometry.iloc[i]
            assert row.interval == two_tile_catalog.gdf.index[i]
            assert row.crs == two_tile_catalog.gdf.crs
            assert row.extras == {}

    def test_extras_include_only_non_reserved_columns(self) -> None:
        catalog = _build(
            [
                {
                    "geometry": shapely.geometry.box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-01-01"),
                    "end_time": pd.Timestamp("2024-01-02"),
                    "filepath": "tile_A.tif",
                    "sensor": "S2A",
                    "cloud_pct": 10,
                },
                {
                    "geometry": shapely.geometry.box(1, 1, 2, 2),
                    "start_time": pd.Timestamp("2024-01-02"),
                    "end_time": pd.Timestamp("2024-01-03"),
                    "filepath": "tile_B.tif",
                    "sensor": "S2B",
                    "cloud_pct": 20,
                },
            ]
        )

        rows = list(catalog.iter_rows())

        assert rows[0].extras == {"sensor": "S2A", "cloud_pct": 10}
        assert rows[1].extras == {"sensor": "S2B", "cloud_pct": 20}

    def test_extras_preserve_pandas_scalar_types(self) -> None:
        """Datetime extras must yield ``pd.Timestamp``, not ``np.datetime64``.

        Regression: an earlier vectorised implementation used
        ``Series.to_numpy(copy=False)`` for extras, which silently coerced
        pandas extension scalars and made ``CatalogRow.extras`` diverge
        from the DuckDB backend (which uses ``Series.iloc[i]``).
        """
        catalog = _build(
            [
                {
                    "geometry": shapely.geometry.box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-01-01"),
                    "end_time": pd.Timestamp("2024-01-02"),
                    "filepath": "tile_A.tif",
                    "observed_at": pd.Timestamp("2024-01-01 12:00"),
                },
            ]
        )

        rows = list(catalog.iter_rows())

        assert isinstance(rows[0].extras["observed_at"], pd.Timestamp)
        assert rows[0].extras["observed_at"] == pd.Timestamp("2024-01-01 12:00")

    def test_uses_interval_as_filepath_fallback(self) -> None:
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [shapely.geometry.box(0, 0, 1, 1)],
                "start_time": [pd.Timestamp("2024-01-01")],
                "end_time": [pd.Timestamp("2024-01-02")],
            },
            geometry="geometry",
            crs="EPSG:32629",
        )
        catalog = InMemoryGeoCatalog(gdf, backend="raster")

        row = next(catalog.iter_rows())

        assert row.filepath == str(catalog.gdf.index[0])

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
