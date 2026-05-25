"""Unit tests for `RasterToPoints` + `PointsToRaster`.

Both operators round-trip values between a `GeoTensor` raster grid
and a `xvec`-indexed xarray DataArray (station / matchup-point
form). They underpin the in-situ-validation half of the matchup
workflow: pulling raster values at AERONET / buoy / drifter
positions, or binning irregular point observations onto a target
raster grid.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from pipekit import Operator
from shapely.geometry import Point


# These tests require the `[vector-cube]` extra; skip the module
# wholesale if it isn't installed.
xvec = pytest.importorskip("xvec")
import xarray as xr

from geotoolz.geom.coregister import PointsToRaster, RasterToPoints


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gt(values: np.ndarray, *, transform=None, crs="EPSG:32629") -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=transform
        or rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs=crs,
        fill_value_default=np.nan,
    )


# Pixel-center coords for a (10, 10) raster with 10 m pixels starting
# at (500_000, 4_000_000) with north-up affine:
#   x_center[j] = 500_000 + 10 * (j + 0.5) = 500_005 + 10*j
#   y_center[i] = 4_000_000 - 10 * (i + 0.5) = 3_999_995 - 10*i
PX_X = lambda j: 500_005.0 + 10.0 * j
PX_Y = lambda i: 3_999_995.0 - 10.0 * i


# ---------------------------------------------------------------------------
# RasterToPoints
# ---------------------------------------------------------------------------


class TestRasterToPoints:
    def test_is_operator_with_config(self) -> None:
        op = RasterToPoints(extract="bilinear", out_var="albedo")
        assert isinstance(op, Operator)
        assert op.get_config() == {"extract": "bilinear", "out_var": "albedo"}

    def test_invalid_extract_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 'nearest' or 'bilinear'"):
            RasterToPoints(extract="cubic")  # type: ignore[arg-type]

    def test_nearest_2d_raster_list_of_points(self) -> None:
        # Build a raster with distinctive cell values so we can read
        # back which cell each point landed in.
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        # values[i, j] = i*10 + j → cell (i=3, j=2) holds 32.
        raster = _gt(values)
        points = [
            Point(PX_X(2), PX_Y(3)),  # should sample 32
            Point(PX_X(7), PX_Y(0)),  # should sample 7
            Point(PX_X(5), PX_Y(5)),  # should sample 55
        ]
        result = RasterToPoints()(raster, points)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.values, [32.0, 7.0, 55.0])

    def test_nearest_2d_raster_geoseries(self) -> None:
        gpd = pytest.importorskip("geopandas")
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        raster = _gt(values)
        gseries = gpd.GeoSeries(
            [Point(PX_X(2), PX_Y(3)), Point(PX_X(7), PX_Y(0))],
            crs="EPSG:32629",
        )
        result = RasterToPoints()(raster, gseries)
        np.testing.assert_array_equal(result.values, [32.0, 7.0])

    def test_nearest_3d_raster_preserves_band_dim(self) -> None:
        # Two-band raster: band 0 = i*10+j, band 1 = (i*10+j) + 1000.
        b0 = np.arange(100, dtype=np.float32).reshape(10, 10)
        b1 = b0 + 1000.0
        raster = _gt(np.stack([b0, b1]))
        points = [Point(PX_X(2), PX_Y(3))]
        result = RasterToPoints()(raster, points)
        # Shape: (band, geometry) = (2, 1).
        assert "band" in result.dims
        assert "geometry" in result.dims
        np.testing.assert_array_equal(
            result.sel(geometry=result["geometry"][0]).values, [32.0, 1032.0]
        )

    def test_bilinear_interpolates_between_pixels(self) -> None:
        # Use a multiplicative field i*j so bilinear genuinely differs
        # from nearest at the corner midpoint (a linear field would
        # be exact under both modes by accident).
        values = np.fromfunction(lambda i, j: (i * j).astype(np.float32), (10, 10))
        raster = _gt(values)
        mid_x = (PX_X(3) + PX_X(4)) / 2
        mid_y = (PX_Y(3) + PX_Y(4)) / 2
        point = Point(mid_x, mid_y)
        nearest = RasterToPoints(extract="nearest")(raster, [point])
        bilinear = RasterToPoints(extract="bilinear")(raster, [point])
        # Bilinear at the corner midpoint averages the four
        # surrounding pixel-center values {3*3, 3*4, 4*3, 4*4}.
        expected = np.mean(values[3:5, 3:5])
        np.testing.assert_allclose(bilinear.values, [expected], rtol=1e-5)
        # Nearest snaps to one of the corner cells (12, 12, 12, or 16);
        # 12.25 ≠ any of those.
        assert not np.allclose(nearest.values, bilinear.values, rtol=1e-5)

    def test_higher_dim_raster_rejected(self) -> None:
        # 4-D (T, C, H, W) isn't supported by the (x, y) coord builder.
        values = np.zeros((1, 2, 4, 4), dtype=np.float32)
        raster = _gt(values)
        with pytest.raises(ValueError, match=r"2-D .* or 3-D"):
            RasterToPoints()(raster, [Point(PX_X(0), PX_Y(0))])


# ---------------------------------------------------------------------------
# PointsToRaster
# ---------------------------------------------------------------------------


class TestPointsToRaster:
    def test_is_operator_with_config(self) -> None:
        op = PointsToRaster(method="binned_stat", stat="median")
        assert isinstance(op, Operator)
        cfg = op.get_config()
        assert cfg["method"] == "binned_stat"
        assert cfg["stat"] == "median"

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 'binned_stat' or 'idw'"):
            PointsToRaster(method="kdtree")  # type: ignore[arg-type]

    def test_idw_not_implemented(self) -> None:
        like = _gt(np.zeros((4, 4), dtype=np.float32))
        op = PointsToRaster(method="idw")
        gpd = pytest.importorskip("geopandas")
        df = gpd.GeoDataFrame(
            {"v": [1.0]}, geometry=[Point(PX_X(0), PX_Y(0))], crs="EPSG:32629"
        )
        with pytest.raises(NotImplementedError, match=r"idw.* not yet"):
            op(df, like)

    def test_bin_mean_geodataframe(self) -> None:
        gpd = pytest.importorskip("geopandas")
        # 10x10 target grid.
        like = _gt(np.zeros((10, 10), dtype=np.float32))
        # Place two points in cell (i=3, j=2) and one in cell (i=0, j=0).
        df = gpd.GeoDataFrame(
            {"v": [10.0, 20.0, 99.0]},
            geometry=[
                Point(PX_X(2), PX_Y(3)),
                Point(PX_X(2), PX_Y(3)),
                Point(PX_X(0), PX_Y(0)),
            ],
            crs="EPSG:32629",
        )
        result = PointsToRaster(stat="mean", attribute="v")(df, like)
        # GeoTensor's `__getitem__` only supports slice indexing;
        # for element access we go through numpy.
        arr = np.asarray(result)
        # The cell (3, 2) holds mean(10, 20) = 15.
        assert result.shape == (10, 10)
        assert arr[3, 2] == pytest.approx(15.0)
        assert arr[0, 0] == pytest.approx(99.0)
        # Cells with no points are NaN (binned_statistic_2d default).
        assert np.isnan(arr[5, 5])

    def test_bin_count(self) -> None:
        gpd = pytest.importorskip("geopandas")
        like = _gt(np.zeros((10, 10), dtype=np.float32))
        df = gpd.GeoDataFrame(
            {"v": [1.0, 1.0, 1.0]},
            geometry=[
                Point(PX_X(2), PX_Y(3)),
                Point(PX_X(2), PX_Y(3)),
                Point(PX_X(0), PX_Y(0)),
            ],
            crs="EPSG:32629",
        )
        result = PointsToRaster(stat="count", attribute="v")(df, like)
        arr = np.asarray(result)
        assert arr[3, 2] == pytest.approx(2.0)
        assert arr[0, 0] == pytest.approx(1.0)

    def test_geodataframe_requires_attribute(self) -> None:
        gpd = pytest.importorskip("geopandas")
        like = _gt(np.zeros((4, 4), dtype=np.float32))
        df = gpd.GeoDataFrame(
            {"v": [1.0]}, geometry=[Point(PX_X(0), PX_Y(0))], crs="EPSG:32629"
        )
        with pytest.raises(ValueError, match=r"requires.*attribute"):
            PointsToRaster()(df, like)

    def test_unsupported_input_raises(self) -> None:
        like = _gt(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(TypeError, match=r"Unsupported|expects"):
            PointsToRaster()(["not points"], like)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round-trip — pull values out then bin them back in
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_extract_then_rebin_recovers_at_sampled_pixels(self) -> None:
        gpd = pytest.importorskip("geopandas")
        # Single-band raster with a known pattern.
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        raster = _gt(values)

        sample_geoms = [
            Point(PX_X(2), PX_Y(3)),
            Point(PX_X(7), PX_Y(0)),
            Point(PX_X(5), PX_Y(5)),
        ]
        extracted = RasterToPoints()(raster, sample_geoms)

        # Build a GeoDataFrame from the extracted values.
        df = gpd.GeoDataFrame(
            {"v": extracted.values}, geometry=sample_geoms, crs="EPSG:32629"
        )
        rebinned = PointsToRaster(stat="mean", attribute="v")(df, raster)

        # Sampled pixels should hold the original values.
        arr = np.asarray(rebinned)
        for geom, original_val in zip(sample_geoms, extracted.values, strict=True):
            # Recover the row/col from the geometry.
            col = int((geom.x - 500_005.0) / 10.0)
            row = int((3_999_995.0 - geom.y) / 10.0)
            assert arr[row, col] == pytest.approx(float(original_val))
