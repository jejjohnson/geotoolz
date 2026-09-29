"""Tests for pure NumPy matched-filter operators."""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import fill_pixel_mask, toy_geotensor
from georeader.geotensor import GeoTensor

import geotoolz as gz


def test_matched_filter_module_exports() -> None:
    for name in gz.matched_filter.__all__:
        assert getattr(gz.matched_filter, name) is not None, name


def test_matched_filter_recovers_known_amplitude_and_preserves_metadata() -> None:
    mean = np.array([10.0, 20.0, 30.0])
    target = np.array([1.0, 2.0, -1.0])
    amplitudes = np.array([[0.0, 1.0], [2.0, -0.5]])
    cube = mean[:, None, None] + target[:, None, None] * amplitudes[None, :, :]
    gt = toy_geotensor(cube)

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
    gt = toy_geotensor(cube)

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

    out = op(toy_geotensor(cube))

    assert np.asarray(out).shape == (2, 2)
    assert op.mean is not None
    assert op.cov_op is not None


def test_streaming_background_matches_empirical_covariance() -> None:
    cube_a = toy_geotensor(np.arange(8, dtype=float).reshape(2, 2, 2))
    cube_b = toy_geotensor(np.arange(8, 16, dtype=float).reshape(2, 2, 2))

    bg = gz.matched_filter.StreamingBackground(cov_kind="empirical")([cube_a, cube_b])
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
    cube_a = toy_geotensor(np.arange(8, dtype=float).reshape(2, 2, 2))
    cube_b = toy_geotensor(np.arange(8, 16, dtype=float).reshape(2, 2, 2))

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
    gt = toy_geotensor(cube)

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
    op(toy_geotensor(cube))

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
    gt = toy_geotensor(cube)

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
    gt = toy_geotensor(base)

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
    gt = toy_geotensor(cube)

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
    gt = toy_geotensor(cube)

    scores = np.asarray(
        gz.matched_filter.MatchedFilter(mean=mean, cov_op=cov_matrix, target=target)(gt)
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
        (mf.MatchedFilterSNR, {"amplitude": 1.0, "cov_op": cov, "target": target}),
        (
            mf.DetectionThreshold,
            {"false_alarm_rate": 0.05, "cov_op": cov, "target": target},
        ),
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


_NDARRAY_TARGET = np.array([1.0, -0.5, 0.25])


def _ndarray_cube() -> np.ndarray:
    """Tiny bimodal (3, 4, 4) cube so GMM clustering splits cleanly."""
    rng = np.random.default_rng(7)
    cube = 0.01 * rng.normal(size=(3, 4, 4))
    cube += np.array([5.0, 6.0, 7.0])[:, None, None]
    cube[:, :, 2:] += 10.0
    return cube


def _run_matched_filter(cube):
    return gz.matched_filter.MatchedFilter(
        mean=np.array([5.0, 6.0, 7.0]), cov_op=np.eye(3), target=_NDARRAY_TARGET
    )(cube)


def _run_apply_cluster_mf(cube):
    bg = gz.matched_filter.GMMClusterBackground(n_clusters=2, random_state=0)(cube)
    return gz.matched_filter.ApplyClusterMF(target=_NDARRAY_TARGET)(cube, bg)


def _run_column_enhancement(cube):
    return gz.matched_filter.ColumnEnhancement(obs_model=None)(cube)


@pytest.mark.parametrize(
    "run_op",
    [_run_matched_filter, _run_apply_cluster_mf, _run_column_enhancement],
    ids=["matched_filter", "apply_cluster_mf", "column_enhancement"],
)
def test_score_map_ops_support_plain_ndarray_carrier(run_op) -> None:
    # Plain ndarray in -> plain ndarray out (NOT a GeoTensor), with values
    # identical to the GeoTensor path.
    cube = _ndarray_cube()

    out_arr = run_op(cube)
    out_gt = run_op(toy_geotensor(cube))

    assert type(out_arr) is np.ndarray
    assert isinstance(out_gt, GeoTensor)
    np.testing.assert_allclose(out_arr, np.asarray(out_gt))


def test_estimator_ops_accept_plain_ndarray_input() -> None:
    # Estimator operators return non-carrier results (vectors, linear
    # operators, background dataclasses); plain-array input must give the
    # same values as the GeoTensor path.
    cube = _ndarray_cube()
    gt = toy_geotensor(cube)

    np.testing.assert_allclose(
        gz.matched_filter.EstimateMean(method="median")(cube),
        gz.matched_filter.EstimateMean(method="median")(gt),
    )
    cov_arr = gz.matched_filter.EstimateCovShrunk()(cube)
    cov_gt = gz.matched_filter.EstimateCovShrunk()(gt)
    assert isinstance(cov_arr, gz.matched_filter.NumpyLinearOperator)
    np.testing.assert_allclose(cov_arr.matrix, cov_gt.matrix)

    adaptive = gz.matched_filter.AdaptiveWindowBackground(window_size=3)(cube)
    assert adaptive.mean.shape == cube.shape

    streaming = gz.matched_filter.StreamingBackground(cov_kind="empirical")(
        [cube, cube]
    )
    np.testing.assert_allclose(streaming.mean, cube.reshape(3, -1).mean(axis=1))


def test_obs_model_operators_are_forbid_in_yaml() -> None:
    """``obs_model`` is a required callable: no config form exists (#140)."""
    from geotoolz.matched_filter import LinearTargetFromObs, NonlinearTargetFromObs

    assert LinearTargetFromObs.forbid_in_yaml is True
    assert NonlinearTargetFromObs.forbid_in_yaml is True


def test_column_enhancement_obs_model_refuses_reload() -> None:
    """A dropped ``obs_model`` used to reload as a uniform target (#140)."""
    import json

    from pipekit import Operator

    from geotoolz.matched_filter import ColumnEnhancement

    def obs_model(state: object) -> object:
        return state

    op = ColumnEnhancement(obs_model=obs_model)
    assert op.get_config()["obs_model"] == {"callable": "obs_model"}
    with pytest.raises(RuntimeError, match="non-primitive"):
        Operator.from_state(json.loads(json.dumps(op.state)))

    plain = ColumnEnhancement()
    assert Operator.from_state(plain.state).get_config() == plain.get_config()


def _fill_cube(fill: float) -> tuple[GeoTensor, np.ndarray, np.ndarray]:
    """(3, 4, 4) cube with fill pixels, its valid mask, and the valid-only cube."""
    rng = np.random.default_rng(3)
    values = rng.normal(loc=[[[1.0]], [[2.0]], [[3.0]]], size=(3, 4, 4))
    gt = toy_geotensor(values, fill_value_default=fill, with_fill_pixels=True)
    valid = ~fill_pixel_mask(values.shape)
    # The same spectra with the fill pixels absent, as a (3, n_valid, 1) cube.
    clean = values[:, valid][:, :, None]
    return gt, valid, clean


@pytest.mark.parametrize("fill", [-9999.0, np.nan], ids=["fill-9999", "fill-nan"])
@pytest.mark.parametrize(
    "case",
    [
        "estimate_mean",
        "cov_empirical",
        "cov_shrunk",
        "cov_lowrank",
        "matched_filter",
        "cluster",
        "adaptive_window",
        "streaming",
    ],
)
def test_fill_pixels_are_excluded(case: str, fill: float) -> None:
    """Fill pixels never enter a fitted statistic (#145); score maps hold NaN
    there and declare ``fill_value_default=NaN`` whatever the input fill (#146)."""
    mf = gz.matched_filter
    gt, valid, clean = _fill_cube(fill)
    target = np.array([1.0, 0.5, -1.0])

    def assert_fill(out: GeoTensor) -> None:
        assert np.isnan(out.fill_value_default)
        values = np.asarray(out)
        assert np.isnan(values[~valid]).all()
        assert np.isfinite(values[valid]).all()

    if case == "estimate_mean":
        for method in ("mean", "median", "trimmed", "huber"):
            np.testing.assert_allclose(
                mf.EstimateMean(method=method)(gt),
                mf.EstimateMean(method=method)(clean),
            )
    elif case == "cov_empirical":
        np.testing.assert_allclose(
            mf.EstimateCovEmpirical()(gt).matrix,
            mf.EstimateCovEmpirical()(clean).matrix,
        )
    elif case == "cov_shrunk":
        np.testing.assert_allclose(
            mf.EstimateCovShrunk()(gt).matrix, mf.EstimateCovShrunk()(clean).matrix
        )
    elif case == "cov_lowrank":
        np.testing.assert_allclose(
            mf.EstimateCovLowRank(rank=2)(gt).matrix,
            mf.EstimateCovLowRank(rank=2)(clean).matrix,
        )
    elif case == "matched_filter":
        # Issue reproduction: the fitted mean was [-615.6, -615.6, -615.6].
        op = mf.MatchedFilter(target=target, mean_method="mean")
        out = op(gt)
        np.testing.assert_allclose(op.mean, clean.reshape(3, -1).mean(axis=1))
        ref = mf.MatchedFilter(target=target, mean_method="mean")(clean)
        scores = np.asarray(out)
        np.testing.assert_allclose(scores[valid], np.asarray(ref).ravel())
        assert_fill(out)
    elif case == "cluster":
        cluster = mf.GMMClusterBackground(n_clusters=2)(gt)
        assert (cluster.labels[~valid] == -1).all()
        assert (cluster.labels[valid] >= 0).all()
        assert np.isfinite(cluster.means).all()
        assert_fill(mf.ApplyClusterMF(target=target)(gt, cluster))
    elif case == "adaptive_window":
        bg = mf.AdaptiveWindowBackground(window_size=3)(gt)
        assert np.isnan(bg.mean[:, ~valid]).all()
        assert np.isnan(bg.variance[:, ~valid]).all()
        # Pixel (1, 1): its 3x3 window holds the (0, 0) fill pixel.
        window = np.asarray(gt)[:, 0:3, 0:3].reshape(3, -1)[:, 1:]
        np.testing.assert_allclose(bg.mean[:, 1, 1], window.mean(axis=1))
        np.testing.assert_allclose(bg.variance[:, 1, 1], window.var(axis=1, ddof=1))
    elif case == "streaming":
        result = mf.StreamingBackground(cov_kind="empirical")([gt, gt])
        expected = mf.StreamingBackground(cov_kind="empirical")([clean, clean])
        np.testing.assert_allclose(result.mean, expected.mean)
        np.testing.assert_allclose(result.cov_op.matrix, expected.cov_op.matrix)


def test_4d_time_stack() -> None:
    """The spectral axis defaults to -3; 2-D maps are rejected (#147)."""
    from _helpers import time_stack

    stack = time_stack((2, 3, 6, 6))
    samples = np.moveaxis(np.asarray(stack), 1, -1).reshape(-1, 3)
    mean = gz.matched_filter.EstimateMean(method="mean")(stack)
    np.testing.assert_allclose(mean, samples.mean(axis=0))
    scores = gz.matched_filter.MatchedFilter(target=np.ones(3))(stack)
    assert isinstance(scores, GeoTensor)
    assert scores.shape == (2, 1, 6, 6)
    with pytest.raises(ValueError, match="MatchedFilter: spectral axis -3"):
        gz.matched_filter.MatchedFilter(target=np.ones(3))(
            toy_geotensor(np.ones((6, 6)))
        )
    with pytest.raises(ValueError, match="AdaptiveWindowBackground accepts 3-D"):
        gz.matched_filter.AdaptiveWindowBackground(window_size=3)(stack)
