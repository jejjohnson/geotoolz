"""Unit tests for `RasterToPointCloud`, `PointCloudToRaster`,
`VectorToRasterAgg`.

These three operators round out the numpy-first cross-modality
coregister surface. `RasterToPoints` / `PointsToRaster` are their
shapely-first siblings (already merged); this PR adds the
point-cloud (numpy XY arrays) + vector-with-aggregation pair.
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import toy_geotensor
from georeader.geotensor import GeoTensor
from pipekit import Operator
from shapely.geometry import Point, box

from geotoolz.geom.coregister import (
    PointCloudToRaster,
    PointsToRaster,
    RasterToPointCloud,
    RasterToPoints,
    VectorToRasterAgg,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gt(values: np.ndarray, *, transform=None, crs="EPSG:32629") -> GeoTensor:
    return toy_geotensor(
        values, transform=transform, crs=crs, fill_value_default=np.nan
    )


PX_X = lambda j: 500_005.0 + 10.0 * j
PX_Y = lambda i: 3_999_995.0 - 10.0 * i


# ---------------------------------------------------------------------------
# RasterToPointCloud
# ---------------------------------------------------------------------------


class TestRasterToPointCloudConstruction:
    def test_is_operator_with_config(self) -> None:
        op = RasterToPointCloud(k=3, max_radius=50.0, method="idw")
        assert isinstance(op, Operator)
        cfg = op.get_config()
        assert cfg["k"] == 3
        assert cfg["max_radius"] == 50.0
        assert cfg["method"] == "idw"

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match=r"k must be >= 1"):
            RasterToPointCloud(k=0)

    def test_nearest_or_bilinear_require_k_eq_1(self) -> None:
        # Picking k>1 with nearest/bilinear is conceptually nonsense
        # (those are k=1 concepts); fail fast.
        with pytest.raises(ValueError, match=r"only supports k=1"):
            RasterToPointCloud(k=3, method="nearest")
        with pytest.raises(ValueError, match=r"only supports k=1"):
            RasterToPointCloud(k=3, method="bilinear")
        # IDW with k>1 is the documented use case (the whole point of
        # supporting k>1 at all); also accept k=1 (degenerate IDW that
        # equals nearest).
        RasterToPointCloud(k=3, method="idw")
        RasterToPointCloud(k=1, method="idw")


class TestRasterToPointCloudNearest:
    def test_2d_raster_nearest_via_ndarray(self) -> None:
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        raster = _gt(values)
        cloud = np.array(
            [
                [PX_X(2), PX_Y(3)],  # → cell (3, 2) = 32
                [PX_X(7), PX_Y(0)],  # → cell (0, 7) = 7
            ]
        )
        result = RasterToPointCloud()(raster, cloud)
        np.testing.assert_array_equal(result, [32.0, 7.0])

    def test_3d_raster_preserves_band_axis(self) -> None:
        b0 = np.arange(100, dtype=np.float32).reshape(10, 10)
        b1 = b0 + 1000.0
        raster = _gt(np.stack([b0, b1]))
        cloud = np.array([[PX_X(2), PX_Y(3)]])
        result = RasterToPointCloud()(raster, cloud)
        # Shape: (bands, N).
        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result.ravel(), [32.0, 1032.0])

    def test_ignores_z_column(self) -> None:
        # (N, 3) inputs are accepted; the Z column is dropped.
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        raster = _gt(values)
        cloud = np.array([[PX_X(2), PX_Y(3), 999.0]])
        result = RasterToPointCloud()(raster, cloud)
        np.testing.assert_array_equal(result, [32.0])

    def test_max_radius_masks_far_points(self) -> None:
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        raster = _gt(values)
        # First point sits at a pixel centre; second is 100m away.
        cloud = np.array(
            [
                [PX_X(2), PX_Y(3)],
                [PX_X(2) + 100, PX_Y(3) + 100],
            ]
        )
        result = RasterToPointCloud(max_radius=20.0)(raster, cloud)
        assert result[0] == pytest.approx(32.0)
        assert np.isnan(result[1])

    def test_empty_cloud_returns_empty(self) -> None:
        raster = _gt(np.zeros((4, 4), dtype=np.float32))
        result = RasterToPointCloud()(raster, np.zeros((0, 2)))
        assert result.shape == (0,)


class TestRasterToPointCloudIDW:
    def test_idw_blends_k_neighbours(self) -> None:
        # Use a constant raster — IDW reduces to that constant.
        raster = _gt(np.full((10, 10), 7.0, dtype=np.float32))
        cloud = np.array([[PX_X(2) + 2, PX_Y(3) + 2]])
        out = RasterToPointCloud(k=4, method="idw")(raster, cloud)
        np.testing.assert_allclose(out, [7.0], rtol=1e-9)

    def test_idw_max_radius_gates(self) -> None:
        # Far point with all neighbours beyond the radius → NaN.
        raster = _gt(np.full((10, 10), 7.0, dtype=np.float32))
        cloud = np.array([[600_000.0, 4_500_000.0]])
        out = RasterToPointCloud(k=4, method="idw", max_radius=10.0)(raster, cloud)
        assert np.isnan(out[0])


class TestRasterToPointCloudInputForms:
    def test_geoseries_input(self) -> None:
        gpd = pytest.importorskip("geopandas")
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        raster = _gt(values)
        gs = gpd.GeoSeries([Point(PX_X(2), PX_Y(3))], crs="EPSG:32629")
        result = RasterToPointCloud()(raster, gs)
        np.testing.assert_array_equal(result, [32.0])

    def test_non_point_geometry_rejected(self) -> None:
        from shapely.geometry import LineString

        raster = _gt(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="Point geometries"):
            RasterToPointCloud()(raster, [LineString([(0, 0), (1, 1)])])

    def test_unsupported_input_raises(self) -> None:
        raster = _gt(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(TypeError, match="Unsupported cloud"):
            RasterToPointCloud()(raster, 42)  # type: ignore[arg-type]

    def test_tuple_input_clear_error_message(self) -> None:
        # (xy, values) is PointCloudToRaster's input form, not ours.
        # Surface a clear error instead of "requires Point geometries"
        # which is what the generic iterable branch would emit.
        raster = _gt(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(TypeError, match="PointCloudToRaster"):
            RasterToPointCloud()(raster, (np.array([[0, 0]]), np.array([1.0])))

    def test_geoseries_crs_mismatch_rejected(self) -> None:
        # Silent CRS mismatch would sample the wrong pixels;
        # validate when both sides have a CRS set.
        gpd = pytest.importorskip("geopandas")
        raster = _gt(np.zeros((4, 4), dtype=np.float32))  # EPSG:32629
        gs = gpd.GeoSeries([Point(-9, 38)], crs="EPSG:4326")
        with pytest.raises(ValueError, match="CRS"):
            RasterToPointCloud()(raster, gs)

    def test_idw_k_clamped_to_pixel_count(self) -> None:
        # 2x2 raster has only 4 pixels; ask for k=10. Without the
        # clamp this crashes inside KDTree's fancy-index lookup.
        raster = _gt(np.full((2, 2), 5.0, dtype=np.float32))
        cloud = np.array([[PX_X(0), PX_Y(0)]])
        out = RasterToPointCloud(k=10, method="idw")(raster, cloud)
        # IDW on a constant raster returns the constant regardless
        # of how many neighbours we ask for.
        np.testing.assert_allclose(out, [5.0], rtol=1e-9)


# ---------------------------------------------------------------------------
# PointCloudToRaster
# ---------------------------------------------------------------------------


class TestPointCloudToRasterConstruction:
    def test_is_operator_with_config(self) -> None:
        op = PointCloudToRaster(method="idw", power=1.5, k=4, max_radius=50.0)
        assert isinstance(op, Operator)
        cfg = op.get_config()
        assert cfg["method"] == "idw"
        assert cfg["power"] == 1.5
        assert cfg["k"] == 4

    def test_invalid_method_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"binned_stat.*idw"):
            PointCloudToRaster(method="kdtree")  # type: ignore[arg-type]

    def test_k_validation(self) -> None:
        with pytest.raises(ValueError, match=r"k must be >= 1"):
            PointCloudToRaster(k=0)


class TestPointCloudToRasterBinnedStat:
    def test_mean_at_known_cells(self) -> None:
        like = _gt(np.zeros((10, 10), dtype=np.float32))
        xy = np.array(
            [
                [PX_X(2), PX_Y(3)],
                [PX_X(2), PX_Y(3)],  # duplicate at the same cell
                [PX_X(0), PX_Y(0)],
            ]
        )
        values = np.array([10.0, 20.0, 99.0])
        result = PointCloudToRaster(stat="mean")(  # tuple form
            (xy, values), like
        )
        arr = np.asarray(result)
        assert arr[3, 2] == pytest.approx(15.0)
        assert arr[0, 0] == pytest.approx(99.0)
        assert np.isnan(arr[5, 5])

    def test_structured_array_input(self) -> None:
        like = _gt(np.zeros((10, 10), dtype=np.float32))
        cloud = np.zeros(2, dtype=[("x", "f8"), ("y", "f8"), ("value", "f8")])
        cloud["x"] = [PX_X(2), PX_X(0)]
        cloud["y"] = [PX_Y(3), PX_Y(0)]
        cloud["value"] = [10.0, 99.0]
        result = PointCloudToRaster(stat="sum")(cloud, like)
        arr = np.asarray(result)
        assert arr[3, 2] == pytest.approx(10.0)
        assert arr[0, 0] == pytest.approx(99.0)

    def test_unsupported_input_raises(self) -> None:
        like = _gt(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(TypeError, match=r"expects cloud"):
            PointCloudToRaster()("not a tuple", like)  # type: ignore[arg-type]

    def test_mismatched_lengths_raise(self) -> None:
        like = _gt(np.zeros((4, 4), dtype=np.float32))
        xy = np.array([[PX_X(0), PX_Y(0)], [PX_X(1), PX_Y(1)]])
        values = np.array([1.0])  # only one
        with pytest.raises(ValueError, match=r"same N"):
            PointCloudToRaster()((xy, values), like)


class TestPointCloudToRasterIDW:
    def test_idw_constant_field_reduces_to_constant(self) -> None:
        # A single value sample: IDW everywhere returns that value
        # (degenerate-but-correct k=1 IDW).
        like = _gt(np.zeros((6, 6), dtype=np.float32))
        xy = np.array([[PX_X(3), PX_Y(3)]])
        values = np.array([42.0])
        result = PointCloudToRaster(method="idw", k=1)((xy, values), like)
        arr = np.asarray(result)
        np.testing.assert_allclose(arr, 42.0, rtol=1e-9)

    def test_idw_empty_cloud_returns_all_nan(self) -> None:
        # Empty cloud → all-NaN raster on the `like` grid.
        # Without the guard, KDTree(empty) / query(k=0) would crash.
        like = _gt(np.zeros((4, 4), dtype=np.float32))
        xy = np.zeros((0, 2))
        values = np.zeros((0,))
        result = PointCloudToRaster(method="idw", k=4)((xy, values), like)
        arr = np.asarray(result)
        assert arr.shape == (4, 4)
        assert np.all(np.isnan(arr))

    def test_idw_with_max_radius_masks_distant_pixels(self) -> None:
        like = _gt(np.zeros((10, 10), dtype=np.float32))
        # Single point at upper-left corner; only the closest 2-3
        # pixels should be within 25m, the rest masked.
        xy = np.array([[PX_X(0), PX_Y(0)]])
        values = np.array([99.0])
        result = PointCloudToRaster(method="idw", k=1, max_radius=25.0)(
            (xy, values), like
        )
        arr = np.asarray(result)
        assert arr[0, 0] == pytest.approx(99.0)
        # Far cells (5+ pixels away → >50m) should be NaN.
        assert np.isnan(arr[5, 5])


# ---------------------------------------------------------------------------
# VectorToRasterAgg
# ---------------------------------------------------------------------------


class TestVectorToRasterAggConstruction:
    def test_is_operator_with_config(self) -> None:
        op = VectorToRasterAgg(agg="mean", attribute="class_id", all_touched=True)
        assert isinstance(op, Operator)
        cfg = op.get_config()
        assert cfg == {"agg": "mean", "attribute": "class_id", "all_touched": True}

    def test_invalid_agg_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"agg must be"):
            VectorToRasterAgg(agg="random")  # type: ignore[arg-type]

    def test_attribute_required_for_value_aggs(self) -> None:
        with pytest.raises(ValueError, match=r"requires `attribute`"):
            VectorToRasterAgg(agg="mean")
        with pytest.raises(ValueError, match=r"requires `attribute`"):
            VectorToRasterAgg(agg="sum")
        # count + majority can omit attribute (count is feature-count;
        # majority operates on geometry presence).
        VectorToRasterAgg(agg="count")

    def test_majority_not_implemented(self) -> None:
        gpd = pytest.importorskip("geopandas")
        like = _gt(np.zeros((4, 4), dtype=np.float32))
        df = gpd.GeoDataFrame(
            {"class": [1]},
            geometry=[box(PX_X(0), PX_Y(2), PX_X(2), PX_Y(0))],
            crs="EPSG:32629",
        )
        with pytest.raises(NotImplementedError, match="majority"):
            VectorToRasterAgg(agg="majority")(df, like)


class TestVectorToRasterAggValues:
    def _gdf(self):
        gpd = pytest.importorskip("geopandas")
        # Three overlapping 3x3-pixel boxes near the top-left:
        # - feature 0: rows 0-3, cols 0-3, attr=10
        # - feature 1: rows 0-3, cols 2-5, attr=20   (overlaps cols 2-3)
        # - feature 2: rows 4-6, cols 5-7, attr=30   (disjoint)
        geoms = [
            box(PX_X(0) - 5, PX_Y(2) - 5, PX_X(2) + 5, PX_Y(0) + 5),
            box(PX_X(2) - 5, PX_Y(2) - 5, PX_X(4) + 5, PX_Y(0) + 5),
            box(PX_X(5) - 5, PX_Y(5) - 5, PX_X(6) + 5, PX_Y(4) + 5),
        ]
        return gpd.GeoDataFrame(
            {"v": [10.0, 20.0, 30.0]}, geometry=geoms, crs="EPSG:32629"
        )

    def test_count_overlap_doubled(self) -> None:
        like = _gt(np.zeros((10, 10), dtype=np.float32))
        result = VectorToRasterAgg(agg="count")(self._gdf(), like)
        arr = np.asarray(result)
        # The overlap region of features 0+1 (cols 2-3, rows 0-2)
        # has two features → count == 2; the disjoint feature 2 is 1.
        assert arr[1, 2] == 2.0
        assert arr[5, 5] == 1.0
        # Pixels with no features = 0.
        assert arr[5, 0] == 0.0

    def test_sum_of_attribute(self) -> None:
        like = _gt(np.zeros((10, 10), dtype=np.float32))
        result = VectorToRasterAgg(agg="sum", attribute="v")(self._gdf(), like)
        arr = np.asarray(result)
        # Overlap cell: 10 + 20 = 30.
        assert arr[1, 2] == pytest.approx(30.0)
        # Feature 0 only: 10.
        assert arr[1, 0] == pytest.approx(10.0)
        # Feature 2 only: 30.
        assert arr[5, 5] == pytest.approx(30.0)
        # No feature → NaN.
        assert np.isnan(arr[5, 0])

    def test_mean_of_attribute(self) -> None:
        like = _gt(np.zeros((10, 10), dtype=np.float32))
        result = VectorToRasterAgg(agg="mean", attribute="v")(self._gdf(), like)
        arr = np.asarray(result)
        # Overlap cell: mean(10, 20) = 15.
        assert arr[1, 2] == pytest.approx(15.0)
        assert arr[1, 0] == pytest.approx(10.0)

    def test_max_min(self) -> None:
        like = _gt(np.zeros((10, 10), dtype=np.float32))
        gdf = self._gdf()
        rmax = np.asarray(VectorToRasterAgg(agg="max", attribute="v")(gdf, like))
        rmin = np.asarray(VectorToRasterAgg(agg="min", attribute="v")(gdf, like))
        assert rmax[1, 2] == 20.0
        assert rmin[1, 2] == 10.0

    def test_first_last_distinguish(self) -> None:
        # "first" → feature 0 wins (its value 10 lands at overlap).
        # "last"  → feature 1 wins (its value 20 lands at overlap).
        like = _gt(np.zeros((10, 10), dtype=np.float32))
        gdf = self._gdf()
        first = np.asarray(VectorToRasterAgg(agg="first", attribute="v")(gdf, like))
        last = np.asarray(VectorToRasterAgg(agg="last", attribute="v")(gdf, like))
        assert first[1, 2] == 10.0
        assert last[1, 2] == 20.0


class TestGeoDependentPlainArrayRejection:
    """Every coregister op needs transform + CRS on its raster argument.

    Plain ``np.ndarray`` rasters must raise a clear TypeError instead
    of an AttributeError buried in transform access. The guards fire
    before any optional-dependency import, so the `RasterToPoints` /
    `PointsToRaster` checks run without the ``[vector-cube]`` extra.
    """

    def test_plain_array_raster_inputs_raise_type_error(self) -> None:
        arr = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(TypeError, match="GeoTensor"):
            RasterToPointCloud()(arr, np.zeros((1, 2)))
        with pytest.raises(TypeError, match="GeoTensor"):
            PointCloudToRaster()((np.zeros((1, 2)), np.zeros(1)), arr)
        with pytest.raises(TypeError, match="GeoTensor"):
            VectorToRasterAgg(agg="count")(None, arr)
        with pytest.raises(TypeError, match="GeoTensor"):
            RasterToPoints()(arr, [Point(0.0, 0.0)])
        with pytest.raises(TypeError, match="GeoTensor"):
            PointsToRaster(attribute="v")(None, arr)


class TestVectorToRasterAggValidation:
    def test_non_geodataframe_rejected(self) -> None:
        like = _gt(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(TypeError, match="GeoDataFrame"):
            VectorToRasterAgg(agg="count")([], like)  # type: ignore[arg-type]

    def test_crs_mismatch_rejected(self) -> None:
        gpd = pytest.importorskip("geopandas")
        like = _gt(np.zeros((4, 4), dtype=np.float32))  # EPSG:32629
        df = gpd.GeoDataFrame(
            {"v": [1.0]},
            geometry=[box(-9, 38, -8, 39)],
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="CRS"):
            VectorToRasterAgg(agg="mean", attribute="v")(df, like)
