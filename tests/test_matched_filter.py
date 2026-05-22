"""Tests for pure NumPy matched-filter operators."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz


def _make_geotensor(values: np.ndarray) -> GeoTensor:
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
    gt = _make_geotensor(cube)

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
    gt = _make_geotensor(cube)

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

    out = op(_make_geotensor(cube))

    assert np.asarray(out).shape == (2, 2)
    assert op.mean is not None
    assert op.cov_op is not None


def test_streaming_background_matches_empirical_covariance() -> None:
    cube_a = _make_geotensor(np.arange(8, dtype=float).reshape(2, 2, 2))
    cube_b = _make_geotensor(np.arange(8, 16, dtype=float).reshape(2, 2, 2))

    bg = gz.matched_filter.StreamingBackground(cov_kind="empirical")(
        [cube_a, cube_b]
    )
    stacked = np.concatenate(
        [np.asarray(cube_a).reshape(2, -1), np.asarray(cube_b).reshape(2, -1)],
        axis=1,
    ).T

    assert np.allclose(bg.mean, stacked.mean(axis=0))
    assert np.allclose(
        bg.cov_op.matrix, np.cov(stacked, rowvar=False) + 1e-8 * np.eye(2)
    )


def test_shrink_covariance_ledoit_wolf_properties() -> None:
    n_features = 5
    base = np.diag([4.0, 3.0, 2.0, 1.5, 1.0])
    off = 0.4
    cov = base.copy()
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                cov[i, j] = off * np.sqrt(cov[i, i] * cov[j, j])

    # Scaled-identity input is already at the target, so shrinkage = 0.
    scaled_identity = 2.0 * np.eye(n_features)
    shrunk_id = gz.matched_filter.shrink_covariance(
        scaled_identity, method="ledoit_wolf", n_samples=100
    )
    np.testing.assert_allclose(shrunk_id, scaled_identity)

    # Larger n_samples => less shrinkage => closer to empirical.
    shrunk_small = gz.matched_filter.shrink_covariance(
        cov, method="ledoit_wolf", n_samples=10
    )
    shrunk_large = gz.matched_filter.shrink_covariance(
        cov, method="ledoit_wolf", n_samples=10_000
    )
    err_small = float(np.linalg.norm(shrunk_small - cov))
    err_large = float(np.linalg.norm(shrunk_large - cov))
    assert err_small > err_large

    # Result is a convex combination of cov and the scaled-identity target.
    mu = float(np.trace(cov) / n_features)
    target = mu * np.eye(n_features)
    for shrunk in (shrunk_small, shrunk_large):
        np.testing.assert_allclose(np.trace(shrunk), np.trace(cov), atol=1e-10)
        lower = np.minimum(cov, target)
        upper = np.maximum(cov, target)
        assert np.all(shrunk >= lower - 1e-10)
        assert np.all(shrunk <= upper + 1e-10)


def test_streaming_background_uses_streaming_mean_and_shrunk_covariance() -> None:
    cube_a = _make_geotensor(np.arange(8, dtype=float).reshape(2, 2, 2))
    cube_b = _make_geotensor(np.arange(8, 16, dtype=float).reshape(2, 2, 2))

    bg = gz.matched_filter.StreamingBackground()([cube_a, cube_b])

    # Mean equals the concatenated per-pixel mean.
    stacked_pixels = np.concatenate(
        [np.asarray(cube_a).reshape(2, -1), np.asarray(cube_b).reshape(2, -1)],
        axis=1,
    ).T  # shape (n_pixels, n_features)
    assert isinstance(bg, gz.matched_filter.StreamingBackgroundResult)
    np.testing.assert_allclose(bg.mean, stacked_pixels.mean(axis=0))

    # Shrunk covariance lies on the segment between the empirical cov
    # and the scaled-identity target (convex combination).
    empirical = np.cov(stacked_pixels, rowvar=False)
    mu = float(np.trace(empirical) / empirical.shape[0])
    target = mu * np.eye(empirical.shape[0])
    shrunk = bg.cov_op.matrix
    np.testing.assert_allclose(np.trace(shrunk), np.trace(empirical), atol=1e-6)
    lower = np.minimum(empirical, target)
    upper = np.maximum(empirical, target)
    assert np.all(shrunk >= lower - 1e-6)
    assert np.all(shrunk <= upper + 1e-6)


def test_cluster_background_and_dispatch_are_reproducible() -> None:
    target = np.array([1.0, 0.0])
    cube = np.array(
        [
            [[0.0, 0.1], [10.0, 10.1]],
            [[0.0, 0.1], [10.0, 10.1]],
        ]
    )
    gt = _make_geotensor(cube)

    bg1 = gz.matched_filter.GMMClusterBackground(n_clusters=2, random_state=4)(gt)
    bg2 = gz.matched_filter.GMMClusterBackground(n_clusters=2, random_state=4)(gt)
    out = gz.matched_filter.ApplyClusterMF(target=target)(gt, bg1)
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


def test_fit_on_call_false_preserves_explicit_mean() -> None:
    # When fit_on_call is False and the user supplies an explicit mean but
    # leaves cov_op=None, the mean must NOT be silently overwritten on
    # first apply: only the missing covariance is fit on the cube.
    fixed_mean = np.array([100.0, 200.0])
    target = np.array([1.0, 0.0])
    cube = np.ones((2, 4, 4)) * fixed_mean[:, None, None]

    op = gz.matched_filter.MatchedFilter(
        mean=fixed_mean,
        target=target,
        fit_on_call=False,
        cov_method="empirical",
    )
    op(_make_geotensor(cube))

    np.testing.assert_allclose(op.mean, fixed_mean)
    assert op.cov_op is not None  # cov was fit on the cube


def test_linear_target_from_obs_matches_finite_difference() -> None:
    a = np.array([0.5, -1.0, 2.0])

    def obs_model(x: np.ndarray) -> np.ndarray:
        # Linear model: maps a (bands, h, w) cube to bands by summing
        # over space; tangent-linear derivative wrt a uniform perturbation
        # is a scaled by the spatial size.
        return (x * a[:, None, None]).sum(axis=(1, 2))

    cube = np.zeros((3, 2, 3))
    gt = _make_geotensor(cube)

    target = gz.matched_filter.LinearTargetFromObs(
        obs_model=obs_model, pattern="uniform"
    )(gt)

    expected = a * 2 * 3
    np.testing.assert_allclose(target, expected, rtol=1e-6, atol=1e-8)


def test_nonlinear_target_from_obs_amplitude_difference() -> None:
    def obs_model(x: np.ndarray) -> np.ndarray:
        # Nonlinear (quadratic) per-band response, reduced spatially.
        return (x**2).mean(axis=(1, 2))

    base = np.full((2, 3, 3), 1.0)
    gt = _make_geotensor(base)

    target = gz.matched_filter.NonlinearTargetFromObs(
        obs_model=obs_model, amplitude=0.5, pattern="uniform"
    )(gt)

    # y(base) = 1, y(base + 0.5) = 2.25, difference per band = 1.25
    np.testing.assert_allclose(target, np.full(2, 1.25), rtol=1e-6)


def test_column_enhancement_end_to_end_wires_components() -> None:
    rng = np.random.default_rng(0)
    bands, h, w = 4, 5, 6
    bg_mean = np.array([10.0, 11.0, 12.0, 13.0])
    cube = bg_mean[:, None, None] + rng.normal(size=(bands, h, w)) * 0.01
    gt = _make_geotensor(cube)

    out = gz.matched_filter.ColumnEnhancement(
        gas="CH4", sensor="EMIT", obs_model=None, cov_method="ledoit_wolf"
    )(gt)

    arr = np.asarray(out)
    assert arr.shape == (h, w)
    # With no obs_model, target is uniform-1 and outputs should be small
    # mean-zero residuals around the background mean.
    assert abs(arr.mean()) < 0.1
    assert out.transform == gt.transform


def test_matched_filter_null_distribution_at_known_false_alarm_rate() -> None:
    # Sanity check on the analytical FAR threshold: under a Gaussian null,
    # the fraction of MF scores above DetectionThreshold(false_alarm_rate=p)
    # should be approximately p.
    rng = np.random.default_rng(123)
    bands = 6
    cov_matrix = np.eye(bands)
    target = np.array([1.0, -0.5, 0.3, 0.0, 0.2, -0.1])
    mean = np.zeros(bands)

    n_pixels = 20000
    samples = rng.multivariate_normal(mean=mean, cov=cov_matrix, size=n_pixels)
    cube = samples.T.reshape(bands, n_pixels, 1)
    gt = _make_geotensor(cube)

    scores = np.asarray(
        gz.matched_filter.MatchedFilter(
            mean=mean, cov_op=cov_matrix, target=target
        )(gt)
    ).reshape(-1)

    for far in (0.05, 0.01):
        threshold = gz.matched_filter.DetectionThreshold(
            false_alarm_rate=far, cov_op=cov_matrix, target=target
        )()
        empirical_far = float((scores > threshold).mean())
        # Three-sigma binomial tolerance for n=20000.
        tol = 3.0 * np.sqrt(far * (1 - far) / n_pixels)
        assert abs(empirical_far - far) < tol, (far, empirical_far, tol)


def test_operator_get_configs_are_json_safe_and_round_trippable() -> None:
    import json

    target = np.array([1.0, 2.0])
    cov = np.eye(2)
    mean = np.array([0.0, 0.0])

    mf = gz.matched_filter
    ops_with_args: list[tuple[type, dict]] = [
        (mf.MatchedFilter, {"target": target, "cov_op": cov, "mean": mean}),
        (mf.MatchedFilterSNR,
         {"amplitude": 1.0, "cov_op": cov, "target": target}),
        (mf.DetectionThreshold,
         {"false_alarm_rate": 0.05, "cov_op": cov, "target": target}),
        (mf.ValidateMFInputs, {"cov_op": cov, "target": target}),
        (mf.StreamingBackground, {}),
        (mf.ApplyClusterMF, {"target": target}),
        (mf.EstimateMean, {}),
        (mf.EstimateCovEmpirical, {}),
        (mf.EstimateCovShrunk, {}),
    ]
    for cls, kwargs in ops_with_args:
        op = cls(**kwargs)
        cfg = op.get_config()
        # Config must round-trip through JSON (i.e. be free of numpy arrays).
        json.dumps(cfg)
        # And must be sufficient to reconstruct the operator without errors.
        cls(**{**kwargs, **{k: cfg[k] for k in cfg if k in kwargs}})
