"""Tests for pure NumPy matched-filter operators."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz


def _gt(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(30.0, 0.0, 100.0, 0.0, -30.0, 200.0),
        crs="EPSG:32629",
    )


def test_matched_filter_module_exports() -> None:
    for name in gz.matched_filter.__all__:
        assert getattr(gz.matched_filter, name) is not None, name


def test_matched_filter_recovers_known_amplitude_and_preserves_metadata() -> None:
    mean = np.array([10.0, 20.0, 30.0])
    target = np.array([1.0, 2.0, -1.0])
    amplitudes = np.array([[0.0, 1.0], [2.0, -0.5]])
    cube = mean[:, None, None] + target[:, None, None] * amplitudes[None, :, :]
    gt = _gt(cube)

    out = gz.matched_filter.MatchedFilter(
        mean=mean,
        cov_op=np.eye(3),
        target=target,
    )(gt)

    assert np.allclose(np.asarray(out), amplitudes)
    assert out.transform == gt.transform
    assert str(out.crs) == "EPSG:32629"


def test_pixel_matches_image_kernel() -> None:
    mean = np.array([1.0, 2.0])
    target = np.array([2.0, 1.0])
    cov = np.array([[2.0, 0.0], [0.0, 1.0]])
    pixel = mean + 3.0 * target

    score = gz.matched_filter.MatchedFilterPixel(mean=mean, cov_op=cov, target=target)(
        pixel
    )

    assert score == pytest.approx(3.0)


def test_estimators_return_expected_numpy_backgrounds() -> None:
    cube = np.array(
        [
            [[1.0, 2.0], [3.0, 100.0]],
            [[10.0, 20.0], [30.0, 1000.0]],
        ]
    )
    gt = _gt(cube)

    assert np.allclose(gz.matched_filter.EstimateMean(method="median")(gt), [2.5, 25.0])
    cov = gz.matched_filter.EstimateCovEmpirical(mean=np.array([0.0, 0.0]), ridge=1e-6)(
        gt
    )

    assert isinstance(cov, gz.matched_filter.NumpyLinearOperator)
    assert cov.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(cov.matrix) > 0)


def test_snr_threshold_and_validation() -> None:
    target = np.array([1.0, 2.0])
    cov = np.eye(2)

    snr = gz.matched_filter.MatchedFilterSNR(amplitude=3.0, cov_op=cov, target=target)()
    threshold = gz.matched_filter.DetectionThreshold(
        false_alarm_rate=0.5, cov_op=cov, target=target
    )()

    assert snr == pytest.approx(3.0 * np.sqrt(5.0))
    assert threshold == pytest.approx(0.0)
    assert gz.matched_filter.ValidateMFInputs(cov_op=cov, target=target)("ok") == "ok"
    with pytest.raises(ValueError, match="target"):
        gz.matched_filter.ValidateMFInputs(cov_op=cov, target=np.zeros(2))()
    with pytest.raises(ValueError, match="non-singular"):
        gz.matched_filter.ValidateMFInputs(cov_op=np.zeros((2, 2)), target=target)()


def test_fit_on_call_populates_reusable_state() -> None:
    mean = np.array([1.0, 2.0])
    target = np.array([0.5, 1.0])
    cube = mean[:, None, None] + target[:, None, None] * np.ones((1, 2, 2))
    op = gz.matched_filter.MatchedFilter(
        target=target, fit_on_call=True, cov_method="empirical"
    )

    out = op(_gt(cube))

    assert np.asarray(out).shape == (2, 2)
    assert op.mean is not None
    assert op.cov_op is not None


def test_streaming_background_matches_empirical_covariance() -> None:
    cube_a = _gt(np.arange(8, dtype=float).reshape(2, 2, 2))
    cube_b = _gt(np.arange(8, 16, dtype=float).reshape(2, 2, 2))

    bg = gz.matched_filter.StreamingBackground(
        cubes=[cube_a, cube_b], cov_kind="empirical"
    )()
    stacked = np.concatenate(
        [np.asarray(cube_a).reshape(2, -1), np.asarray(cube_b).reshape(2, -1)],
        axis=1,
    ).T

    assert np.allclose(bg.mean, stacked.mean(axis=0))
    assert np.allclose(
        bg.cov_op.matrix, np.cov(stacked, rowvar=False) + 1e-8 * np.eye(2)
    )


def test_streaming_background_shrunk_covariance_matches_batch_estimator() -> None:
    cube_a = _gt(np.arange(8, dtype=float).reshape(2, 2, 2))
    cube_b = _gt(np.arange(8, 16, dtype=float).reshape(2, 2, 2))

    bg = gz.matched_filter.StreamingBackground(cubes=[cube_a, cube_b])()
    stacked = np.concatenate(
        [np.asarray(cube_a).reshape(2, -1), np.asarray(cube_b).reshape(2, -1)],
        axis=1,
    ).T
    empirical = np.cov(stacked, rowvar=False) + 1e-8 * np.eye(2)
    expected = gz.matched_filter.estimate_cov_shrunk(
        stacked.T.reshape(2, 2, 4), axis=0
    ).matrix

    assert isinstance(bg, gz.matched_filter.StreamingBackgroundResult)
    assert np.allclose(bg.mean, stacked.mean(axis=0))
    assert np.allclose(bg.cov_op.matrix, expected)
    assert not np.allclose(bg.cov_op.matrix, empirical)


def test_cluster_background_and_dispatch_are_reproducible() -> None:
    target = np.array([1.0, 0.0])
    cube = np.array(
        [
            [[0.0, 0.1], [10.0, 10.1]],
            [[0.0, 0.1], [10.0, 10.1]],
        ]
    )
    gt = _gt(cube)

    bg1 = gz.matched_filter.GMMClusterBackground(n_clusters=2, random_state=4)(gt)
    bg2 = gz.matched_filter.GMMClusterBackground(n_clusters=2, random_state=4)(gt)
    out = gz.matched_filter.ApplyClusterMF(cluster=bg1, target=target)(gt)
    samples = np.asarray(gt).reshape(2, -1).T
    labels = bg1.labels.reshape(-1)
    expected = np.array(
        [
            gz.matched_filter.apply_pixel(
                sample,
                mean=bg1.means[label],
                cov_op=bg1.cov_ops[label],
                target=target,
            )
            for sample, label in zip(samples, labels, strict=True)
        ]
    ).reshape(bg1.labels.shape)

    assert np.array_equal(bg1.labels, bg2.labels)
    assert np.allclose(bg1.means, bg2.means)
    for cov1, cov2 in zip(bg1.cov_ops, bg2.cov_ops, strict=True):
        assert np.allclose(cov1.matrix, cov2.matrix)
    assert np.allclose(np.asarray(out), expected)
    assert np.asarray(out).shape == (2, 2)
    assert out.transform == gt.transform
