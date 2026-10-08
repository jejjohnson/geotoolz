"""Behavioural catalog tests, run on every backend (#235).

Each test here runs three times — on an `InMemoryGeoCatalog`, on a
`DuckDBGeoCatalog` wrapping it, and on a `DuckDBGeoCatalog` opened from
the GeoParquet it writes — so a divergence between backends fails the
suite. Backend-specific behaviour (constructor validation, intersect
engines, DuckDB lifecycle) stays in ``test_inmemory.py`` /
``test_duckdb.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import shapely.geometry

from geocatalog import GeoSlice, intersect, query, union
from geocatalog._src.base import GeoCatalog
from geocatalog.backends import InMemoryGeoCatalog


_convert: Callable[[InMemoryGeoCatalog], GeoCatalog] = lambda cat: cat


@pytest.fixture(autouse=True)
def _bind_backend(
    as_backend: Callable[[InMemoryGeoCatalog], GeoCatalog],
) -> Iterator[None]:
    """Route every `_build` in this module through the backend under test."""
    global _convert
    _convert = as_backend
    yield
    _convert = lambda cat: cat


def _build(rows: list[dict], crs: str = "EPSG:32629") -> GeoCatalog:
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return _convert(InMemoryGeoCatalog(gdf, kind="raster"))


@pytest.fixture
def two_tile_catalog() -> GeoCatalog:
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


def _row(geom: shapely.Geometry, start: str, end: str, path: str) -> dict:
    return {
        "geometry": geom,
        "start_time": pd.Timestamp(start),
        "end_time": pd.Timestamp(end),
        "filepath": path,
    }


class TestProperties:
    def test_total_bounds(self, two_tile_catalog: GeoCatalog) -> None:
        assert two_tile_catalog.total_bounds == (0.0, 0.0, 300.0, 100.0)

    def test_temporal_extent(self, two_tile_catalog: GeoCatalog) -> None:
        ext = two_tile_catalog.temporal_extent
        assert ext.left == pd.Timestamp("2024-01-01")
        assert ext.right == pd.Timestamp("2024-01-03")

    def test_len(self, two_tile_catalog: GeoCatalog) -> None:
        assert len(two_tile_catalog) == 2

    def test_query_rejects_crs_alongside_slice(
        self, two_tile_catalog: GeoCatalog
    ) -> None:
        sl = GeoSlice(
            (0.0, 0.0, 100.0, 100.0),
            pd.Interval(
                pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03"), closed="both"
            ),
            (10.0, 10.0),
            "EPSG:32629",
        )
        with pytest.raises(TypeError, match="carries its own crs"):
            two_tile_catalog.query(sl, crs="EPSG:4326")

    def test_temporal_extent_empty_is_none(self, two_tile_catalog: GeoCatalog) -> None:
        empty = two_tile_catalog.query(bounds=(1e6, 1e6, 2e6, 2e6), crs="EPSG:32629")
        assert len(empty) == 0
        assert empty.temporal_extent is None


class TestQuery:
    def test_query_filters_by_bbox(self, two_tile_catalog: GeoCatalog) -> None:
        out = two_tile_catalog.query(bounds=(0, 0, 50, 50), crs="EPSG:32629")
        assert len(out) == 1
        assert out.gdf["filepath"].iloc[0] == "tile_A.tif"

    def test_query_filters_by_time(self, two_tile_catalog: GeoCatalog) -> None:
        out = two_tile_catalog.query(time=("2024-01-02 12:00", "2024-01-03 12:00"))
        assert len(out) == 1
        assert out.gdf["filepath"].iloc[0] == "tile_B.tif"

    def test_query_by_slice(self, two_tile_catalog: GeoCatalog) -> None:
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
        self, two_tile_catalog: GeoCatalog
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
        self, two_tile_catalog: GeoCatalog
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
    def test_intersect_spatiotemporal(self, two_tile_catalog: GeoCatalog) -> None:
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
        self, two_tile_catalog: GeoCatalog
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

    def test_intersect_spatial_only(self, two_tile_catalog: GeoCatalog) -> None:
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

    def test_union(self, two_tile_catalog: GeoCatalog) -> None:
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

    def test_union_reprojects(self, two_tile_catalog: GeoCatalog) -> None:
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
    def test_yields_catalog_rows_in_order(self, two_tile_catalog: GeoCatalog) -> None:
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
        catalog = _convert(InMemoryGeoCatalog(gdf, kind="raster"))

        row = next(catalog.iter_rows())

        assert row.filepath == str(catalog.gdf.index[0])


class TestIterSlices:
    def test_yields_one_per_row(self, two_tile_catalog: GeoCatalog) -> None:
        slices = list(two_tile_catalog.iter_slices(resolution=(10.0, 10.0)))
        assert len(slices) == 2
        for s in slices:
            assert isinstance(s, GeoSlice)
            assert s.resolution == (10.0, 10.0)

    def test_slice_bounds_match_footprints(self, two_tile_catalog: GeoCatalog) -> None:
        slices = list(two_tile_catalog.iter_slices(resolution=(10.0, 10.0)))
        np.testing.assert_allclose(slices[0].bounds, (0, 0, 100, 100))
        np.testing.assert_allclose(slices[1].bounds, (200, 0, 300, 100))


class TestQueryFreeFunction:
    def test_delegates(self, two_tile_catalog: GeoCatalog) -> None:
        out = query(two_tile_catalog, bounds=(0, 0, 50, 50), crs="EPSG:32629")
        assert len(out) == 1


class TestIntersectFixes:
    """#230: GeometryCollection clips, index-column sync, empty schema."""

    def test_geometry_collection_clip_keeps_polygon_part(self) -> None:
        left = _build([_row(shapely.box(0, 0, 2, 2), "2024-01-01", "2024-01-03", "l")])
        # Overlaps [1, 2] x [0, 2] and touches the left edge at x = 0, so the
        # clip is GeometryCollection(Polygon, LineString).
        right_geom = shapely.MultiPolygon(
            [shapely.box(1, 0, 3, 2), shapely.box(-1, 0, 0, 2)]
        )
        right = _build([_row(right_geom, "2024-01-01", "2024-01-03", "r")])
        out = left.intersect(right)
        assert len(out) == 1
        geom = out.gdf.geometry.iloc[0]
        assert geom.geom_type == "Polygon"
        assert geom.equals(shapely.box(1, 0, 2, 2))

    def test_time_columns_follow_clipped_index(self) -> None:
        left = _build([_row(shapely.box(0, 0, 2, 2), "2024-01-01", "2024-01-03", "l")])
        right = _build([_row(shapely.box(1, 0, 3, 2), "2024-01-02", "2024-01-05", "r")])
        out = left.intersect(right)
        assert out.gdf.index[0] == pd.Interval(
            pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), closed="both"
        )
        # The in-memory backend also keeps time *columns* when it was built
        # from them; where present they must follow the clipped index.
        if "start_time" in out.gdf.columns:
            assert out.gdf["start_time"].iloc[0] == pd.Timestamp("2024-01-02")
            assert out.gdf["end_time"].iloc[0] == pd.Timestamp("2024-01-03")
        assert "_right_start_time" not in out.gdf.columns
        assert "_right_end_time" not in out.gdf.columns
        assert out.gdf["_right_filepath"].iloc[0] == "r"

    def test_empty_intersect_keeps_schema(self, two_tile_catalog: GeoCatalog) -> None:
        far = _build(
            [_row(shapely.box(1e6, 1e6, 2e6, 2e6), "2024-01-01", "2024-01-02", "x")]
        )
        out = two_tile_catalog.intersect(far)
        assert len(out) == 0
        # Same columns as a non-empty intersect: left + `_right_` columns.
        assert set(two_tile_catalog.gdf.columns) <= set(out.gdf.columns)
        assert "_right_filepath" in out.gdf.columns
        assert "_right_start_time" not in out.gdf.columns
        assert out.gdf.crs == two_tile_catalog.gdf.crs
        assert out.temporal_extent is None


class TestUnionGeometryName:
    def test_differently_named_geometry_column(
        self, two_tile_catalog: GeoCatalog
    ) -> None:
        other_gdf = gpd.GeoDataFrame(
            {
                "footprint": [shapely.box(500, 0, 600, 100)],
                "start_time": [pd.Timestamp("2024-01-05")],
                "end_time": [pd.Timestamp("2024-01-06")],
                "filepath": ["tile_C.tif"],
            },
            geometry="footprint",
            crs="EPSG:32629",
        )
        other = _convert(InMemoryGeoCatalog(other_gdf, kind="raster"))
        merged = two_tile_catalog.union(other)
        assert len(merged) == 3
        assert merged.gdf.geometry.notna().all()
        assert "footprint" not in merged.gdf.columns
        assert merged.total_bounds == (0.0, 0.0, 600.0, 100.0)
