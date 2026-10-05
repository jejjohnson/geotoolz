"""Tests for the shared georeferencing helpers (``geotoolz._src.geo``)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from _helpers import DEFAULT_TRANSFORM, toy_geotensor

import geotoolz as gz
from geotoolz._src.geo import (
    grid_matches,
    ground_pixel_size,
    pixel_xy,
    require_geotensor,
    require_projected_crs,
)


def _shifted(dx: float) -> rasterio.Affine:
    t = DEFAULT_TRANSFORM
    return rasterio.Affine(t.a, t.b, t.c + dx, t.d, t.e, t.f)


def _old_xy(transform, row: float, col: float) -> tuple[float, float]:
    """The per-point formula ``feature`` / ``measure`` / ``plume`` used to inline."""
    x = transform.c + transform.a * (col + 0.5) + transform.b * (row + 0.5)
    y = transform.f + transform.d * (col + 0.5) + transform.e * (row + 0.5)
    return float(x), float(y)


# --- require_geotensor -------------------------------------------------------


def test_require_geotensor_returns_georeferenced_input() -> None:
    gt = toy_geotensor(np.zeros((2, 2)))
    assert require_geotensor(gt, "Op") is gt


def test_require_geotensor_message() -> None:
    with pytest.raises(TypeError) as excinfo:
        require_geotensor(np.zeros((2, 2)), "MyOp")
    assert str(excinfo.value) == (
        "MyOp requires a georeferenced GeoTensor input; got a plain array (ndarray)."
    )


def test_require_geotensor_arg_and_hint() -> None:
    with pytest.raises(
        TypeError,
        match=r"^Op requires a georeferenced GeoTensor like; got a plain array "
        r"\(list\)\. Pass it explicitly\.$",
    ):
        require_geotensor([1, 2], "Op", arg="like", hint="Pass it explicitly.")


def test_require_geotensor_rejects_transform_none() -> None:
    class Carrier:
        transform = None

    with pytest.raises(TypeError, match="Carrier"):
        require_geotensor(Carrier(), "Op")


@pytest.mark.parametrize(
    "op",
    [
        gz.measure.FindContours(level=0.5),
        gz.feature.PeakLocalMax(),
        gz.viz.ShadedRelief(),
        gz.augment.RandomShift(max_shift=(1, 1)),
    ],
    ids=lambda op: type(op).__name__,
)
def test_operators_share_require_geotensor_message(op) -> None:
    with pytest.raises(TypeError) as excinfo:
        op(np.zeros((4, 4), dtype=np.float32))
    assert str(excinfo.value) == (
        f"{type(op).__name__} requires a georeferenced GeoTensor input; "
        "got a plain array (ndarray)."
    )


# --- grid_matches ------------------------------------------------------------


def test_grid_matches_identical_grids() -> None:
    a = toy_geotensor(np.zeros((2, 3, 3)))
    assert grid_matches(a, toy_geotensor(np.ones((2, 3, 3))))


def test_grid_matches_default_is_exact() -> None:
    a = toy_geotensor(np.zeros((3, 3)))
    b = toy_geotensor(np.zeros((3, 3)), transform=_shifted(1e-9))
    assert not grid_matches(a, b)


def test_grid_matches_atol_is_absolute_per_coefficient() -> None:
    a = toy_geotensor(np.zeros((3, 3)))
    drift = toy_geotensor(np.zeros((3, 3)), transform=_shifted(1e-9))
    assert grid_matches(a, drift, atol=1e-6)
    # No relative term: a 0.5 m shift on a 500 km origin is not "close"
    # even though it is within numpy's default rtol=1e-5.
    half_metre = toy_geotensor(np.zeros((3, 3)), transform=_shifted(0.5))
    assert not grid_matches(a, half_metre, atol=1e-6)
    assert grid_matches(a, half_metre, atol=0.5)


def test_grid_matches_crs_mismatch() -> None:
    a = toy_geotensor(np.zeros((3, 3)))
    b = toy_geotensor(np.zeros((3, 3)), crs="EPSG:32630")
    assert not grid_matches(a, b, atol=1.0)


def test_grid_matches_spatial_only_ignores_band_axis() -> None:
    a = toy_geotensor(np.zeros((2, 3, 3)))
    b = toy_geotensor(np.zeros((4, 3, 3)))
    assert grid_matches(a, b)
    assert not grid_matches(a, b, spatial_only=False)


def test_grid_matches_spatial_shape_mismatch() -> None:
    a = toy_geotensor(np.zeros((3, 3)))
    b = toy_geotensor(np.zeros((3, 4)))
    assert not grid_matches(a, b)


def test_grid_matches_plain_arrays_compare_shape_only() -> None:
    gt = toy_geotensor(np.zeros((2, 3, 3)))
    assert grid_matches(np.zeros((3, 3)), gt)
    assert grid_matches(gt, np.zeros((5, 3, 3)))
    assert not grid_matches(gt, np.zeros((5, 3, 3)), spatial_only=False)
    assert not grid_matches(np.zeros((3, 3)), np.zeros((3, 4)))


def test_dnbr_rejects_sub_pixel_drift() -> None:
    # dNBR used to accept a 1e-9 origin drift (np.allclose defaults) while
    # the compositing operators rejected it; the shared check is exact.
    pre = toy_geotensor(np.ones((3, 3), dtype=np.float32))
    post = toy_geotensor(np.ones((3, 3), dtype=np.float32), transform=_shifted(1e-9))
    with pytest.raises(ValueError, match="dNBR inputs must share"):
        gz.indices.dNBR()(pre, post)


def test_median_composite_requires_full_shape_match() -> None:
    a = toy_geotensor(np.zeros((2, 3, 3), dtype=np.float32))
    b = toy_geotensor(np.zeros((3, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="MedianComposite: the frame 1 pixel grid"):
        gz.compositing.MedianComposite()([a, b])


# --- pixel_xy ----------------------------------------------------------------


@pytest.mark.parametrize(
    "transform",
    [
        DEFAULT_TRANSFORM,
        rasterio.Affine(0.3, 0.07, -122.123456, -0.05, -0.3, 37.987654),
    ],
)
def test_pixel_xy_matches_old_per_point_formula_exactly(transform) -> None:
    rng = np.random.default_rng(0)
    rows = np.concatenate([np.arange(7), rng.uniform(-1, 50, size=20)])
    cols = np.concatenate([np.arange(7)[::-1], rng.uniform(-1, 50, size=20)])
    xs, ys = pixel_xy(transform, rows, cols)
    expected = np.array(
        [_old_xy(transform, r, c) for r, c in zip(rows, cols, strict=True)]
    )
    np.testing.assert_array_equal(xs, expected[:, 0])
    np.testing.assert_array_equal(ys, expected[:, 1])


def test_pixel_xy_centre_convention_and_broadcast() -> None:
    xs, ys = pixel_xy(DEFAULT_TRANSFORM, np.arange(2)[:, None], np.arange(3)[None])
    assert xs.shape == ys.shape == (2, 3)
    assert xs.dtype == np.float64
    assert (xs[0, 0], ys[0, 0]) == (500_005.0, 3_999_995.0)
    np.testing.assert_allclose(
        np.stack([xs, ys]),
        np.stack(
            DEFAULT_TRANSFORM * (np.arange(3)[None] + 0.5, np.arange(2)[:, None] + 0.5)
        ),
    )


def test_pixel_xy_scalar() -> None:
    x, y = pixel_xy(DEFAULT_TRANSFORM, 0, 0)
    assert (float(x), float(y)) == (500_005.0, 3_999_995.0)


def test_plume_pixel_centers_unchanged() -> None:
    from geotoolz.plume._src.array import pixel_centers

    transform = rasterio.Affine(0.3, 0.07, -122.1, -0.05, -0.3, 37.9)
    xs, ys = pixel_centers((4, 5), transform)
    rows, cols = np.indices((4, 5), dtype=float)
    old = np.vectorize(lambda r, c: _old_xy(transform, r, c))(rows, cols)
    np.testing.assert_array_equal(xs, old[0])
    np.testing.assert_array_equal(ys, old[1])


def test_find_contours_vertices_unchanged() -> None:
    transform = rasterio.Affine(0.3, 0.07, -122.1, -0.05, -0.3, 37.9)
    arr = np.zeros((6, 6), dtype=np.float32)
    arr[2:4, 1:5] = 1.0
    gt = toy_geotensor(arr, transform=transform, fill_value_default=None)
    contours = gz.measure.FindContours(level=0.5)(gt)
    from skimage.measure import find_contours

    (raw,) = find_contours(arr, level=0.5)
    expected = [_old_xy(transform, float(r), float(c)) for r, c in raw]
    np.testing.assert_array_equal(
        np.asarray(contours.geometry.iloc[0].coords), np.asarray(expected)
    )


def test_regionprops_centroids_unchanged() -> None:
    transform = rasterio.Affine(0.3, 0.07, -122.1, -0.05, -0.3, 37.9)
    labels = toy_geotensor(
        np.array([[1, 1, 0], [0, 2, 2], [0, 2, 2]], dtype=np.int32),
        transform=transform,
        fill_value_default=0,
    )
    props = gz.measure.RegionProps()(labels)
    assert len(props) == 2
    for r, c, geom in zip(
        props["centroid-0"], props["centroid-1"], props.geometry, strict=True
    ):
        assert (geom.x, geom.y) == _old_xy(transform, r, c)


def test_peak_local_max_points_unchanged() -> None:
    transform = rasterio.Affine(0.3, 0.07, -122.1, -0.05, -0.3, 37.9)
    arr = np.zeros((9, 9), dtype=np.float32)
    arr[2, 3] = 5.0
    arr[6, 6] = 3.0
    gt = toy_geotensor(arr, transform=transform, fill_value_default=None)
    peaks = gz.feature.PeakLocalMax(min_distance=1)(gt)
    assert len(peaks) == 2
    for row, col, geom in zip(peaks["row"], peaks["col"], peaks.geometry, strict=True):
        assert (geom.x, geom.y) == _old_xy(transform, row, col)


# --- ground_pixel_size / require_projected_crs ------------------------------


@pytest.mark.parametrize(
    ("grid", "expected"),
    [("utm", (10.0, 10.0)), ("non_square", (20.0, 10.0)), ("rotated", (10.0, 10.0))],
)
def test_ground_pixel_size_is_the_step_length(grid, expected) -> None:
    """``(hypot(b, e), hypot(a, d))``: on the rotated grid |a| = |e| = 8.66."""
    transform = toy_geotensor(np.zeros((2, 2)), grid=grid).transform
    np.testing.assert_allclose(ground_pixel_size(transform, "Op"), expected)


def test_ground_pixel_size_rejects_a_sheared_grid() -> None:
    transform = toy_geotensor(np.zeros((2, 2)), grid="sheared").transform
    with pytest.raises(ValueError, match=r"^Op needs perpendicular .* sheared"):
        ground_pixel_size(transform, "Op")


def test_require_projected_crs_messages() -> None:
    geographic = toy_geotensor(np.zeros((2, 2)), grid="geographic")
    with pytest.raises(ValueError, match=r"^Op computes slopes .*geographic"):
        require_projected_crs(geographic, "Op", what="slopes")
    feet = toy_geotensor(np.zeros((2, 2)), crs="EPSG:2263")
    with pytest.raises(ValueError, match="linear units"):
        require_projected_crs(feet, "Op")
    require_projected_crs(feet, "Op", metres=False)
    require_projected_crs(toy_geotensor(np.zeros((2, 2))), "Op")
    require_projected_crs(np.zeros((2, 2)), "Op")
