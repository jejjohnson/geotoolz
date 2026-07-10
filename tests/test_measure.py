"""Tests for measurement operators."""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import toy_geotensor
from georeader.geotensor import GeoTensor
from shapely.geometry import LineString

import geotoolz as gz


def _gt(values: np.ndarray) -> GeoTensor:
    return toy_geotensor(values, fill_value_default=0)


def test_label_connected_components_returns_int_geotensor() -> None:
    mask = _gt(np.array([[1, 0, 1], [1, 0, 0]], dtype=bool))

    labels = gz.measure.LabelConnectedComponents(connectivity=1)(mask)

    assert labels.shape == mask.shape
    assert labels.transform == mask.transform
    assert np.asarray(labels).dtype == np.int32
    assert int(np.asarray(labels).max()) == 2


def test_regionprops_returns_geodataframe_with_region_fields() -> None:
    labels = _gt(np.array([[1, 1, 0], [0, 2, 2]], dtype=np.int32))

    props = gz.measure.RegionProps()(labels)

    assert list(props["label"]) == [1, 2]
    assert "area" in props.columns
    assert props.geometry.notna().all()


def test_regionprops_without_centroid_returns_valid_geodataframe() -> None:
    """``properties`` without ``centroid`` must not raise on construction.

    Regression test: previously ``GeoDataFrame(frame, crs=gt.crs)`` raised
    because no geometry column was attached.
    """
    labels = _gt(np.array([[1, 1, 0], [0, 2, 2]], dtype=np.int32))

    props = gz.measure.RegionProps(properties=("label", "area"))(labels)

    assert list(props["label"]) == [1, 2]
    assert list(props["area"]) == [2, 2]
    # Geometry column exists but is empty/None — no centroid was requested.
    assert "geometry" in props.columns
    assert props.geometry.isna().all()


def test_find_contours_returns_linestrings_in_world_coords() -> None:
    arr = np.zeros((20, 20), dtype=float)
    arr[5:15, 5:15] = 1.0

    contours = gz.measure.FindContours(level=0.5)(_gt(arr))

    assert len(contours) >= 1
    assert all(isinstance(geom, LineString) for geom in contours.geometry)
    assert contours.crs == "EPSG:32629"


def test_find_contours_empty_input_returns_empty_geodataframe() -> None:
    """Constant raster (no contours) must return an empty GDF, not raise."""
    arr = np.zeros((10, 10), dtype=float)

    contours = gz.measure.FindContours(level=0.5)(_gt(arr))

    assert len(contours) == 0
    assert "geometry" in contours.columns
    assert contours.crs == "EPSG:32629"


def test_profile_line_samples_pixel_intensities_between_endpoints() -> None:
    arr = np.tile(np.arange(10, dtype=float), (5, 1))

    profile = gz.measure.ProfileLine(src=(2, 0), dst=(2, 9), linewidth=1, order=0)(
        _gt(arr)
    )

    assert profile.shape[0] >= 9
    assert profile[0] == pytest.approx(0.0)
    assert profile[-1] == pytest.approx(9.0)


def test_ransac_fits_line_model_to_collinear_points() -> None:
    from skimage.measure import LineModelND

    rng = np.random.default_rng(0)
    xs = np.linspace(0.0, 10.0, 50)
    ys = 2.0 * xs + 1.0
    # Inject a handful of outliers.
    ys[::10] += 50.0
    data = np.column_stack([xs, ys])

    model, inliers = gz.measure.RANSAC(
        model_class=LineModelND,
        min_samples=2,
        residual_threshold=1.0,
        max_trials=200,
        rng=rng,
    )(data)

    assert inliers.sum() >= 40
    # ``LineModelND`` exposes ``origin`` / ``direction`` (the ``params``
    # tuple alias is deprecated in skimage 0.26).
    direction = np.asarray(model.direction)
    slope = direction[1] / direction[0]
    assert slope == pytest.approx(2.0, rel=0.05)


def test_shannon_entropy_zero_for_constant_array() -> None:
    constant = _gt(np.zeros((8, 8), dtype=float))
    mixed = _gt(np.arange(64, dtype=float).reshape(8, 8))

    assert gz.measure.ShannonEntropy()(constant) == pytest.approx(0.0)
    assert gz.measure.ShannonEntropy()(mixed) > 0.0


def test_skeleton_length_horizontal_line_equals_length_minus_one() -> None:
    mask = np.zeros((5, 10), dtype=bool)
    mask[2, :] = True
    gt = _gt(mask)

    length = gz.measure.SkeletonLength()(gt)

    # 10-pixel-long line skeletonizes to itself: diameter is 9 edges.
    assert length == pytest.approx(9.0)


def test_skeleton_length_zero_for_empty_or_singleton_mask() -> None:
    empty = _gt(np.zeros((4, 4), dtype=bool))
    singleton = np.zeros((4, 4), dtype=bool)
    singleton[1, 1] = True

    assert gz.measure.SkeletonLength()(empty) == 0.0
    assert gz.measure.SkeletonLength()(_gt(singleton)) == 0.0


def test_skeleton_length_diagonal_uses_8_connectivity() -> None:
    # A 5-pixel diagonal is one path of 4 unit-edge steps under
    # 8-connectivity (not sqrt(2)-weighted — see operator docstring).
    mask = np.eye(5, dtype=bool)
    length = gz.measure.SkeletonLength()(_gt(mask))
    assert length == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("op", "values"),
    [
        pytest.param(
            gz.measure.LabelConnectedComponents(connectivity=1),
            np.array([[1, 0, 1], [1, 0, 0]], dtype=bool),
            id="label-connected-components",
        ),
        pytest.param(
            gz.measure.ProfileLine(src=(1, 0), dst=(1, 4), order=0),
            np.tile(np.arange(5, dtype=float), (3, 1)),
            id="profile-line",
        ),
        pytest.param(
            gz.measure.ShannonEntropy(),
            np.arange(16, dtype=float).reshape(4, 4),
            id="shannon-entropy",
        ),
        pytest.param(gz.measure.SkeletonLength(), np.eye(5, dtype=bool), id="skeleton"),
    ],
)
def test_plain_ndarray_matches_geotensor_path(
    op: gz.Operator, values: np.ndarray
) -> None:
    """Metadata-independent measure ops accept plain ndarrays and match
    the GeoTensor path (array outputs stay plain ndarrays)."""
    out_plain = op(values)
    out_geo = op(_gt(values))

    if isinstance(out_plain, np.ndarray):
        assert type(out_plain) is np.ndarray
        np.testing.assert_array_equal(out_plain, np.asarray(out_geo))
    else:
        assert out_plain == pytest.approx(out_geo)


@pytest.mark.parametrize(
    ("op", "values"),
    [
        pytest.param(
            gz.measure.RegionProps(),
            np.array([[1, 1, 0], [0, 2, 2]], dtype=np.int32),
            id="region-props",
        ),
        pytest.param(
            gz.measure.FindContours(level=0.5),
            np.pad(np.ones((3, 3)), 2),
            id="find-contours",
        ),
    ],
)
def test_geo_dependent_ops_reject_plain_arrays(
    op: gz.Operator, values: np.ndarray
) -> None:
    """Ops whose output geometry needs the transform stay GeoTensor-only."""
    with pytest.raises(TypeError, match="georeferenced GeoTensor"):
        op(values)
