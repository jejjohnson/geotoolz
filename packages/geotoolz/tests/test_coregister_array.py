"""Tier-A coregistration primitives + pinned operator numerics.

The ``*_pinned`` tests freeze the outputs the coregister operators
produced before their maths moved into
``geotoolz.geom._src.coregister.array`` (#157); the primitive tests
check that each operator is a thin wrapper over its primitive.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor

from geotoolz.geom._src.coregister import array as coreg_array
from geotoolz.geom.coregister import (
    PointCloudToRaster,
    PointsToRaster,
    RasterToPointCloud,
)


_NAN = np.nan
_TRANSFORM = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 30.0)
_XY = np.array([[3.0, 27.0], [14.0, 12.0], [26.0, 21.0], [38.5, 3.5], [55.0, 15.0]])
_VALUES = np.array([1.0, -2.0, 4.0, 0.5, 3.0])


def _raster() -> GeoTensor:
    return GeoTensor(
        np.arange(24, dtype=np.float64).reshape(2, 3, 4) ** 1.5,
        transform=_TRANSFORM,
        crs="EPSG:32630",
        fill_value_default=-1.0,
    )


# ---------------------------------------------------------------------------
# Pinned operator outputs (captured before the move into array.py)
# ---------------------------------------------------------------------------


def test_raster_to_point_cloud_nearest_pinned() -> None:
    out = RasterToPointCloud(max_radius=12.0)(_raster(), _XY)
    expected = [
        [0.0, 11.180339887498949, 2.8284271247461903, 36.4828726939094, _NAN],
        [
            41.569219381653056,
            70.09279563550022,
            52.38320341483518,
            110.30412503619254,
            _NAN,
        ],
    ]
    # Pinned literals: allow last-bit differences across numpy / BLAS builds.
    np.testing.assert_allclose(out, expected, rtol=1e-14)
    assert out.dtype == np.float64


def test_raster_to_point_cloud_bilinear_pinned() -> None:
    # Bilinear sampling goes through xarray (the [vector-cube] extra).
    pytest.importorskip("xarray")
    out = RasterToPointCloud(method="bilinear", max_radius=12.0)(_raster(), _XY)
    expected = [
        [_NAN, 15.572436639063424, 7.870828004235092, _NAN, _NAN],
        [_NAN, 77.30494701376452, 62.57768970522156, _NAN, _NAN],
    ]
    np.testing.assert_allclose(out, expected, rtol=1e-14)


def test_raster_to_point_cloud_idw_pinned() -> None:
    out = RasterToPointCloud(method="idw", k=3, power=1.5, max_radius=12.0)(
        _raster(), _XY
    )
    expected = [
        [
            0.8241542121172545,
            13.942036426844414,
            6.800550660314942,
            33.525254288085335,
            _NAN,
        ],
        [
            44.108871448174405,
            74.5483255858921,
            60.54747293394176,
            105.8145766778349,
            _NAN,
        ],
    ]
    np.testing.assert_allclose(out, expected, rtol=1e-14)


def test_point_cloud_to_raster_binned_pinned() -> None:
    out = np.asarray(PointCloudToRaster(stat="mean")((_XY, _VALUES), _raster()))
    expected = [
        [1.0, _NAN, 4.0, _NAN],
        [_NAN, -2.0, _NAN, _NAN],
        [_NAN, _NAN, _NAN, 0.5],
    ]
    np.testing.assert_array_equal(out, expected)


def test_point_cloud_to_raster_idw_pinned() -> None:
    out = np.asarray(
        PointCloudToRaster(method="idw", k=3, power=1.5, max_radius=20.0)(
            (_XY, _VALUES), _raster()
        )
    )
    expected = [
        [
            0.9264986036624617,
            1.1604446274104894,
            3.2025866634696203,
            3.1503136487832855,
        ],
        [
            -0.0841785019064287,
            -1.1699686929004087,
            2.088027642870646,
            2.4963266871056815,
        ],
        [
            -0.2985247575183518,
            -0.7502860279692615,
            0.5251208593228741,
            0.6306235771641466,
        ],
    ]
    np.testing.assert_allclose(out, expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stat", ["mean", "median", "sum", "count", "max", "min"])
def test_point_cloud_to_raster_wraps_binned_primitive(stat: str) -> None:
    raster = _raster()
    expected = coreg_array.points_to_raster_binned(
        _XY, _VALUES, dst_shape=(3, 4), dst_transform=_TRANSFORM, stat=stat
    )
    out = PointCloudToRaster(stat=stat)((_XY, _VALUES), raster)
    np.testing.assert_array_equal(np.asarray(out), expected)
    assert expected.dtype == np.float64


def test_points_to_raster_wraps_binned_primitive() -> None:
    gpd = pytest.importorskip("geopandas")
    # The binned raster is assembled with xarray (the [vector-cube] extra).
    pytest.importorskip("xarray")
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"v": _VALUES}, geometry=[Point(*p) for p in _XY], crs="EPSG:32630"
    )
    out = PointsToRaster(stat="sum", attribute="v")(gdf, _raster())
    expected = coreg_array.points_to_raster_binned(
        _XY, _VALUES, dst_shape=(3, 4), dst_transform=_TRANSFORM, stat="sum"
    )
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_binned_primitive_handles_west_up_and_south_up_grids() -> None:
    # Flipping both axes of the grid flips the binned output with it.
    flipped = Affine(-10.0, 0.0, 40.0, 0.0, 10.0, 0.0)
    north_up = coreg_array.points_to_raster_binned(
        _XY, _VALUES, dst_shape=(3, 4), dst_transform=_TRANSFORM
    )
    out = coreg_array.points_to_raster_binned(
        _XY, _VALUES, dst_shape=(3, 4), dst_transform=flipped
    )
    np.testing.assert_array_equal(out, north_up[::-1, ::-1])


@pytest.mark.parametrize(
    ("method", "k"), [("nearest", 1), ("bilinear", 1), ("idw", 1), ("idw", 4)]
)
def test_raster_to_point_cloud_wraps_primitive(method: str, k: int) -> None:
    if method == "bilinear":
        # Bilinear sampling goes through xarray (the [vector-cube] extra).
        pytest.importorskip("xarray")
    raster = _raster()
    out = RasterToPointCloud(method=method, k=k, max_radius=12.0, power=1.5)(
        raster, _XY
    )
    expected = coreg_array.raster_to_point_cloud(
        np.asarray(raster),
        _TRANSFORM,
        _XY,
        k=k,
        max_radius=12.0,
        method=method,
        power=1.5,
    )
    np.testing.assert_array_equal(out, expected)


def test_raster_to_point_cloud_primitive_2d_input() -> None:
    values = np.asarray(_raster())[0]
    out = coreg_array.raster_to_point_cloud(values, _TRANSFORM, _XY)
    np.testing.assert_allclose(
        out,
        [
            0.0,
            11.180339887498949,
            2.8284271247461903,
            36.4828726939094,
            18.520259177452132,
        ],
        rtol=1e-14,
    )


def test_raster_to_point_cloud_primitive_empty_cloud_keeps_dtype() -> None:
    values = np.zeros((2, 3, 4), dtype=np.float32)
    out = coreg_array.raster_to_point_cloud(values, _TRANSFORM, np.zeros((0, 2)))
    assert out.shape == (2, 0)
    assert out.dtype == np.float32


def test_raster_to_point_cloud_rejects_4d_raster_by_name() -> None:
    raster = GeoTensor(np.zeros((1, 2, 3, 4)), transform=_TRANSFORM, crs="EPSG:32630")
    with pytest.raises(ValueError, match=r"^RasterToPointCloud accepts"):
        RasterToPointCloud()(raster, _XY)


def test_raster_to_point_cloud_bilinear_errors_name_the_operator() -> None:
    rotated = Affine(8.66, 5.0, 0.0, -5.0, 8.66, 0.0)
    raster = GeoTensor(np.zeros((3, 4)), transform=rotated, crs="EPSG:32630")
    with pytest.raises(ValueError, match=r"^RasterToPointCloud requires"):
        RasterToPointCloud(method="bilinear")(raster, _XY)


def test_point_cloud_to_raster_idw_wraps_primitive() -> None:
    out = PointCloudToRaster(method="idw", k=2, max_radius=15.0)(
        (_XY, _VALUES), _raster()
    )
    expected = coreg_array.points_to_raster_idw(
        _XY, _VALUES, dst_shape=(3, 4), dst_transform=_TRANSFORM, k=2, max_radius=15.0
    )
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_points_to_raster_idw_primitive_empty_cloud_is_all_nan() -> None:
    out = coreg_array.points_to_raster_idw(
        np.zeros((0, 2)), np.zeros(0), dst_shape=(3, 4), dst_transform=_TRANSFORM
    )
    assert out.shape == (3, 4)
    assert np.isnan(out).all()


def test_reproject_like_primitive_matches_reproject_like_operator() -> None:
    from geotoolz.geom import ReprojectLike

    src = GeoTensor(
        np.arange(2 * 8 * 8, dtype=np.float64).reshape(2, 8, 8),
        transform=Affine(5.0, 0.0, 0.0, 0.0, -5.0, 40.0),
        crs="EPSG:32630",
        fill_value_default=np.nan,
    )
    like = GeoTensor(
        np.zeros((3, 4)),
        transform=_TRANSFORM,
        crs="EPSG:32630",
        fill_value_default=np.nan,
    )
    out = coreg_array.reproject_like(
        np.asarray(src),
        src.transform,
        src.crs,
        dst_shape=(3, 4),
        dst_transform=_TRANSFORM,
        dst_crs="EPSG:32630",
        resampling="average",
    )
    expected = ReprojectLike(like=like, resampling="average")(src)
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out, np.asarray(expected))


def test_reproject_like_primitive_integer_grid_offset_is_a_window_read() -> None:
    values = np.arange(36, dtype=np.int16).reshape(6, 6)
    src_transform = Affine(10.0, 0.0, -10.0, 0.0, -10.0, 40.0)
    out = coreg_array.reproject_like(
        values,
        src_transform,
        "EPSG:32630",
        dst_shape=(3, 4),
        dst_transform=_TRANSFORM,
        dst_crs="EPSG:32630",
    )
    np.testing.assert_array_equal(out, values[1:4, 1:5])
    assert out.dtype == np.int16
