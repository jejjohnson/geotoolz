"""Tests for feature operators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz


def _gt(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(2.0, 0.0, 100.0, 0.0, -2.0, 200.0),
        crs="EPSG:32629",
        fill_value_default=0,
    )


def test_peak_local_max_returns_scored_points() -> None:
    image = np.zeros((5, 5), dtype=float)
    image[2, 3] = 10.0
    gt = _gt(image)

    points = gz.feature.PeakLocalMax(
        min_distance=1,
        threshold_abs=5.0,
        exclude_border=False,
    )(gt)

    assert len(points) == 1
    assert points.iloc[0]["score"] == 10.0
    assert points.crs == gt.crs


def test_canny_returns_boolean_geotensor() -> None:
    image = np.zeros((10, 10), dtype=float)
    image[:, 5:] = 1.0
    gt = _gt(image)

    edges = gz.feature.Canny(sigma=0.5)(gt)

    assert edges.shape == gt.shape
    assert edges.transform == gt.transform
    assert np.asarray(edges).dtype == bool


def _blob_scene() -> GeoTensor:
    rng = np.random.default_rng(0)
    image = rng.uniform(0.0, 0.1, size=(32, 32))
    yy, xx = np.ogrid[:32, :32]
    image += np.exp(-((yy - 8) ** 2 + (xx - 8) ** 2) / 8.0)
    image += np.exp(-((yy - 22) ** 2 + (xx - 22) ** 2) / 8.0)
    return _gt(image)


def test_blob_log_returns_points_with_radius() -> None:
    gt = _blob_scene()
    points = gz.feature.BlobLoG(min_sigma=1.0, max_sigma=4.0, threshold=0.1)(gt)
    assert {"row", "col", "sigma", "radius"}.issubset(points.columns)
    assert points.crs == gt.crs


def test_blob_dog_uses_sigma_ratio() -> None:
    """``BlobDOG`` must forward ``sigma_ratio`` (and not ``num_sigma``)."""
    gt = _blob_scene()
    op = gz.feature.BlobDOG(
        min_sigma=1.0, max_sigma=4.0, sigma_ratio=1.5, threshold=0.1
    )
    points = op(gt)  # Must not raise TypeError on unexpected kwarg.
    assert "sigma_ratio" in op.get_config()
    assert points.crs == gt.crs


def test_blob_doh_radius_is_sigma_not_sqrt_two_scaled() -> None:
    """``BlobDoH`` reports sigma directly as radius (no sqrt(2) scaling)."""
    gt = _blob_scene()
    points = gz.feature.BlobDoH(
        min_sigma=1.0, max_sigma=8.0, num_sigma=4, threshold=0.001
    )(gt)
    if len(points) > 0:
        np.testing.assert_allclose(
            points["radius"].to_numpy(), points["sigma"].to_numpy()
        )


def test_corner_harris_returns_points() -> None:
    image = np.zeros((20, 20), dtype=float)
    image[5:15, 5:15] = 1.0
    gt = _gt(image)
    points = gz.feature.CornerHarris(min_distance=1, threshold_rel=0.1)(gt)
    assert {"row", "col", "response"}.issubset(points.columns)
    assert points.crs == gt.crs


def test_structure_tensor_returns_eigenvalue_stack() -> None:
    image = np.zeros((16, 16), dtype=float)
    image[:, 8:] = 1.0
    gt = _gt(image)
    out = gz.feature.StructureTensor(sigma=1.0)(gt)
    assert out.shape[-2:] == gt.shape[-2:]


def test_multiscale_basic_features_returns_feature_stack() -> None:
    image = np.linspace(0.0, 1.0, 16 * 16).reshape(16, 16)
    gt = _gt(image)
    out = gz.feature.MultiscaleBasicFeatures(sigma_min=0.5, sigma_max=2.0)(gt)
    assert out.shape[-2:] == gt.shape[-2:]


def test_hog_returns_feature_vector() -> None:
    image = np.linspace(0.0, 1.0, 24 * 24).reshape(24, 24)
    gt = _gt(image)
    descriptor = gz.feature.HOG(
        orientations=4, pixels_per_cell=(8, 8), cells_per_block=(2, 2)
    )(gt)
    assert descriptor.ndim == 1
    assert descriptor.size > 0


def test_hough_lines_returns_dataframe() -> None:
    image = np.zeros((20, 20), dtype=bool)
    image[10, :] = True
    gt = _gt(image.astype(float))
    out = gz.feature.HoughLines(num_peaks=3)(gt)
    assert {"accumulator", "angle", "distance"}.issubset(out.columns)


def test_hough_circles_returns_points_with_radii() -> None:
    image = np.zeros((40, 40), dtype=float)
    yy, xx = np.ogrid[:40, :40]
    ring = (yy - 20) ** 2 + (xx - 20) ** 2
    image[(ring > 64) & (ring < 100)] = 1.0
    gt = _gt(image)
    out = gz.feature.HoughCircles(radii=[8, 9, 10], total_num_peaks=2)(gt)
    assert {"row", "col", "radius", "accumulator"}.issubset(out.columns)
    assert out.crs == gt.crs


def _step_image() -> np.ndarray:
    image = np.zeros((8, 8), dtype=float)
    image[:, 4:] = 1.0
    return image


@pytest.mark.parametrize(
    ("op", "values"),
    [
        pytest.param(gz.feature.Canny(sigma=0.5), _step_image(), id="canny"),
        pytest.param(
            gz.feature.StructureTensor(sigma=1.0),
            _step_image(),
            id="structure-tensor",
        ),
        pytest.param(
            gz.feature.MultiscaleBasicFeatures(sigma_min=0.5, sigma_max=1.0),
            np.linspace(0.0, 1.0, 64).reshape(8, 8),
            id="multiscale-basic-features",
        ),
        pytest.param(
            gz.feature.HOG(
                orientations=4, pixels_per_cell=(4, 4), cells_per_block=(1, 1)
            ),
            np.linspace(0.0, 1.0, 64).reshape(8, 8),
            id="hog",
        ),
        pytest.param(
            gz.feature.HoughLines(num_peaks=2), _step_image(), id="hough-lines"
        ),
    ],
)
def test_plain_ndarray_in_plain_result_out(op: gz.Operator, values: np.ndarray) -> None:
    """Metadata-independent feature ops accept plain ndarrays and match
    the GeoTensor path (array outputs stay plain ndarrays)."""
    out_plain = op(values)
    out_geo = op(_gt(values))

    if isinstance(out_plain, np.ndarray):
        assert type(out_plain) is np.ndarray
        np.testing.assert_array_equal(out_plain, np.asarray(out_geo))
    else:
        pd.testing.assert_frame_equal(out_plain, out_geo)


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(gz.feature.PeakLocalMax(min_distance=1), id="peak-local-max"),
        pytest.param(gz.feature.BlobLoG(min_sigma=1.0, max_sigma=2.0), id="blob-log"),
        pytest.param(gz.feature.CornerHarris(), id="corner-harris"),
        pytest.param(gz.feature.HoughCircles(radii=[2]), id="hough-circles"),
    ],
)
def test_geo_dependent_ops_reject_plain_arrays(op: gz.Operator) -> None:
    """Ops whose output geometry needs the transform stay GeoTensor-only."""
    with pytest.raises(TypeError, match="georeferenced GeoTensor"):
        op(np.zeros((6, 6), dtype=float))
