"""Tests for measurement operators."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from shapely.geometry import LineString

import geotoolz as gz


def _gt(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs="EPSG:32629",
        fill_value_default=0,
    )


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

    profile = gz.measure.ProfileLine(
        src=(2, 0), dst=(2, 9), linewidth=1, order=0
    )(_gt(arr))

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
