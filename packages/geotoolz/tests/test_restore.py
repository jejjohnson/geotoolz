"""Tests for `geotoolz.restore`."""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import fill_pixel_mask, toy_geotensor
from georeader.geotensor import GeoTensor
from scipy import ndimage

import geotoolz as gz
from geotoolz.restore import (
    MNF,
    BilateralDenoise,
    DenoisePCA,
    DespeckleFrost,
    DespeckleLee,
    DespeckleRefinedLee,
    DestripeColumn,
    GapFillIDW,
    GapFillInpaintBiharmonic,
    GapFillLaplacian,
    GapFillNearest,
    GaussianDenoise,
    InverseMNF,
    MedianDenoise,
    MomentMatching,
    NLMeans,
    OutlierMask,
    ReplaceOutliers,
    bilateral_denoise,
    despeckle_frost,
    despeckle_lee,
    despeckle_refined_lee,
    destripe_column,
    gap_fill_idw,
    gap_fill_laplacian,
    gap_fill_nearest,
    median_denoise,
    nl_means,
    outlier_mask,
    pca_denoise,
)
from geotoolz.restore._src.array import fit_pca


LEE_VARIANCE_REDUCTION_THRESHOLD = 0.5
DESTRIPE_RMSE_TOLERANCE = 0.01


def test_restore_namespace_is_available() -> None:
    assert gz.restore.DespeckleLee is DespeckleLee
    assert gz.DespeckleLee is DespeckleLee


def test_despeckle_lee_reduces_multiplicative_speckle_variance() -> None:
    rng = np.random.default_rng(0)
    clean = np.ones((64, 64), dtype=float)
    noisy = clean * rng.gamma(shape=1.0, scale=1.0, size=clean.shape)
    # Single-look intensity: Cu = 1.
    out = despeckle_lee(noisy, window=9, cu=1.0)
    assert np.nanvar(out) <= LEE_VARIANCE_REDUCTION_THRESHOLD * np.nanvar(noisy)
    np.testing.assert_allclose(np.nanmean(out), np.nanmean(noisy), rtol=0.05)
    assert np.isfinite(out[[0, -1], :]).all()
    assert np.isfinite(out[:, [0, -1]]).all()


def test_despeckle_operator_preserves_metadata() -> None:
    rng = np.random.default_rng(1)
    gt = toy_geotensor(rng.random((2, 8, 8)).astype(np.float32))
    out = DespeckleLee(window=3)(gt)
    assert isinstance(out, GeoTensor)
    assert out.shape == gt.shape
    assert out.transform == gt.transform
    assert str(out.crs) == "EPSG:32629"
    assert DespeckleFrost(window=3)(gt).shape == gt.shape
    assert DespeckleRefinedLee(window=3)(gt).shape == gt.shape


def test_destripe_column_recovers_flat_image() -> None:
    base = np.ones((32, 32), dtype=float)
    stripe = np.linspace(-0.2, 0.2, 32)
    striped = base + stripe[None, :]
    out = destripe_column(striped, method="mean", axis="column")
    rmse = np.sqrt(np.nanmean((out - base) ** 2))
    assert rmse < DESTRIPE_RMSE_TOLERANCE


def test_destripe_operator_preserves_metadata() -> None:
    gt = toy_geotensor(np.ones((2, 8, 8), dtype=np.float32))
    out = DestripeColumn()(gt)
    assert isinstance(out, GeoTensor)
    assert out.shape == gt.shape
    assert out.transform == gt.transform
    assert MomentMatching(window=3)(gt).shape == gt.shape


def test_mnf_inverse_with_all_components_is_identity() -> None:
    rng = np.random.default_rng(2)
    arr = rng.normal(size=(4, 10, 10)).astype(np.float32)
    gt = toy_geotensor(arr)
    forward = MNF(n_components=4)
    scores = forward(gt)
    restored = InverseMNF(forward=forward)(scores)
    np.testing.assert_allclose(np.asarray(restored), arr, atol=1e-5)
    assert np.all(np.diff(forward.snr_) <= 0)
    reduced_forward = MNF(n_components=2)
    reduced_scores = reduced_forward(gt)
    reduced = InverseMNF(forward=reduced_forward)(reduced_scores)
    assert reduced.shape == gt.shape


def test_denoise_pca_reconstructs_original_shape() -> None:
    rng = np.random.default_rng(3)
    arr = rng.normal(size=(3, 6, 6)).astype(np.float32)
    gt = toy_geotensor(arr)
    out = DenoisePCA(n_components=2)(gt)
    assert out.shape == gt.shape
    assert out.transform == gt.transform


def test_pca_denoise_raises_on_all_nan_band() -> None:
    rng = np.random.default_rng(7)
    arr = rng.normal(size=(3, 5, 5))
    arr[1] = np.nan
    with pytest.raises(ValueError, match=r"entirely NaN.*1"):
        pca_denoise(arr, n_components=2)


def test_gap_fill_laplacian_uses_local_not_periodic_neighbours() -> None:
    # Construct an array where opposite edges differ strongly. If the
    # Laplacian solver used periodic wrap-around (np.roll), the corner
    # fill would pull from the far-edge interior pixels; with edge
    # boundaries it should stay close to the local corner neighbours.
    arr = np.zeros((5, 5), dtype=float)
    arr[:, -1] = 100.0  # right column has high values
    arr[0, 0] = np.nan  # NaN at the top-left corner
    filled = gap_fill_laplacian(arr, iterations=500)
    # Local neighbours of (0, 0) are arr[0, 1] = 0 and arr[1, 0] = 0;
    # wrap-around neighbours would include arr[0, -1] = 100.
    assert filled[0, 0] < 1.0


def test_gaussian_denoise_preserves_nan_mask_and_metadata() -> None:
    arr = np.arange(25, dtype=float).reshape(5, 5)
    arr[2, 2] = np.nan
    gt = toy_geotensor(arr)
    out = GaussianDenoise(sigma=1.0)(gt)
    # The NaN pixel is nodata: it holds the carrier's fill value.
    assert np.asarray(out)[2, 2] == gt.fill_value_default
    assert np.isnan(np.asarray(GaussianDenoise(sigma=1.0)(arr))[2, 2])
    assert out.transform == gt.transform


def test_single_band_denoisers_run_and_preserve_shape() -> None:
    arr = np.arange(25, dtype=float).reshape(5, 5)
    gt = toy_geotensor(arr)
    for op in [
        MedianDenoise(size=3),
        BilateralDenoise(sigma_color=10.0, sigma_space=1.0),
        NLMeans(patch_size=3, patch_distance=3, h=10.0),
    ]:
        out = op(gt)
        assert out.shape == gt.shape
        assert out.transform == gt.transform
    assert median_denoise(arr, size=3).shape == arr.shape
    assert bilateral_denoise(arr, sigma_color=10.0, sigma_space=1.0).shape == arr.shape
    assert nl_means(arr, patch_size=3, patch_distance=3, h=10.0).shape == arr.shape


def test_gap_fill_biharmonic_preserves_non_nan_pixels() -> None:
    arr = np.arange(25, dtype=float).reshape(5, 5)
    arr[2, 2] = np.nan
    gt = toy_geotensor(arr)
    out = GapFillInpaintBiharmonic()(gt)
    assert np.isfinite(np.asarray(out)[2, 2])
    np.testing.assert_array_equal(
        np.asarray(out)[np.isfinite(arr)], arr[np.isfinite(arr)]
    )


def test_gap_fill_laplacian_fills_nan() -> None:
    arr = np.arange(25, dtype=float).reshape(5, 5)
    arr[2, 2] = np.nan
    gt = toy_geotensor(arr)
    out = GapFillLaplacian()(gt)
    assert np.isfinite(np.asarray(out)[2, 2])
    assert np.isfinite(gap_fill_laplacian(arr)[2, 2])


def test_gap_fill_nearest_and_idw_fill_missing_pixel() -> None:
    arr = np.array([[1.0, 2.0], [3.0, np.nan]])
    nearest = gap_fill_nearest(arr)
    idw = gap_fill_idw(arr, power=2.0, radius=2)
    assert nearest[1, 1] == 3.0
    assert np.isfinite(idw[1, 1])


def test_gap_fill_idw_high_power_matches_nearest_operator() -> None:
    arr = np.array([[1.0, 2.0], [3.0, np.nan]])
    gt = toy_geotensor(arr)
    nearest = GapFillNearest(max_distance=2)(gt)
    idw = GapFillIDW(power=128.0, radius=2)(gt)
    boundary = GapFillIDW(power=64.0, radius=2)(gt)
    np.testing.assert_array_equal(np.asarray(idw), np.asarray(nearest))
    np.testing.assert_array_equal(np.asarray(boundary), np.asarray(nearest))
    np.testing.assert_array_equal(
        gap_fill_idw(arr, power=64.0, radius=2),
        gap_fill_nearest(arr, max_distance=2),
    )


def test_outlier_mask_and_replacement() -> None:
    arr = np.ones((5, 5), dtype=float)
    arr[1, 2] = 10.0
    arr[3, 4] = -8.0
    mask = outlier_mask(arr, method="mad", k=3.0)
    assert int(mask.sum()) == 2
    gt = toy_geotensor(arr)
    op_mask = OutlierMask(method="mad", k=3.0)(gt)
    np.testing.assert_array_equal(np.asarray(op_mask), mask)
    replaced = ReplaceOutliers(method="mad", k=3.0, fill="median")(gt)
    assert np.asarray(replaced)[1, 2] == 1.0


# ----------------------------------------------------------------------------
# Known-answer tests for gap-fill primitives.
# A single NaN surrounded by 1s should be filled with ~1 by every method
# regardless of the underlying algorithm — a sanity check on the inpainting
# contract documented in the module-level docstring.
# ----------------------------------------------------------------------------
def test_gap_fill_methods_recover_isolated_nan() -> None:
    arr = np.ones((5, 5), dtype=float)
    arr[2, 2] = np.nan
    gt = toy_geotensor(arr)
    for op in [
        GapFillNearest(),
        GapFillIDW(power=2.0, radius=2),
        GapFillLaplacian(),
        GapFillInpaintBiharmonic(),
    ]:
        out = np.asarray(op(gt))
        assert np.isfinite(out[2, 2]), f"{op!r} left the NaN unfilled"
        np.testing.assert_allclose(out[2, 2], 1.0, atol=1e-6)


def test_gap_fill_biharmonic_does_not_double_apply() -> None:
    """The operator must not modify finite pixels — the primitive already
    preserves originals, so a second ``np.where(isfinite, ...)`` was
    redundant and is no longer applied."""
    arr = np.linspace(0.0, 1.0, 25, dtype=float).reshape(5, 5)
    arr_with_nan = arr.copy()
    arr_with_nan[2, 2] = np.nan
    gt = toy_geotensor(arr_with_nan)
    out = np.asarray(GapFillInpaintBiharmonic()(gt))
    finite_mask = np.isfinite(arr_with_nan)
    np.testing.assert_array_equal(out[finite_mask], arr_with_nan[finite_mask])


def test_bilateral_preserves_strong_edge() -> None:
    """A bilateral filter should retain a sharp edge better than a Gaussian.

    Build a step image with two flat regions; the bilateral output should
    have a smaller maximum deviation from the original at the edge than
    a comparable Gaussian smoother.
    """
    rng = np.random.default_rng(7)
    image = np.where(np.arange(32)[None, :] < 16, 0.0, 1.0) * np.ones((32, 32))
    noisy = image + 0.02 * rng.standard_normal(image.shape)
    gt = toy_geotensor(noisy)
    gauss = np.asarray(GaussianDenoise(sigma=2.0)(gt))
    bilateral = np.asarray(BilateralDenoise(sigma_color=0.05, sigma_space=2.0)(gt))
    edge_col = 15
    gauss_edge_error = np.abs(gauss[:, edge_col] - image[:, edge_col]).max()
    bilat_edge_error = np.abs(bilateral[:, edge_col] - image[:, edge_col]).max()
    assert bilat_edge_error < gauss_edge_error


def test_destripe_column_propagates_window_to_moment_matching() -> None:
    """``DestripeColumn`` must round-trip its ``window`` parameter into the
    underlying primitive so ``method="moment_matching"`` actually uses it."""
    rng = np.random.default_rng(11)
    arr = rng.standard_normal((24, 24))
    op_default = DestripeColumn(method="moment_matching")
    op_wide = DestripeColumn(method="moment_matching", window=11)
    out_default = np.asarray(op_default(toy_geotensor(arr.copy())))
    out_wide = np.asarray(op_wide(toy_geotensor(arr.copy())))
    # Different smoothing windows must produce different outputs.
    assert not np.allclose(out_default, out_wide)


def test_outlier_mask_operator_returns_bool_dtype() -> None:
    arr = np.ones((4, 4), dtype=float)
    arr[0, 0] = 100.0
    out = OutlierMask(method="mad", k=3.0)(toy_geotensor(arr))
    assert np.asarray(out).dtype == np.dtype(bool)


# ----------------------------------------------------------------------------
# Tier-B contract: every Operator subclass should report a JSON-safe
# ``get_config`` and round-trip through that config (except for
# :class:`InverseMNF`, which holds a runtime reference and is flagged
# ``forbid_in_yaml=True``).
# ----------------------------------------------------------------------------
def test_operator_configs_are_json_safe() -> None:
    import json

    operators = [
        DespeckleLee(window=5, cu=0.523),
        DespeckleFrost(window=5, damping=2.0),
        DespeckleRefinedLee(window=5),
        DestripeColumn(method="median", axis="row", window=15),
        MomentMatching(window=11),
        DenoisePCA(n_components=2, axis=0),
        MNF(n_components=2, axis=0),
        GaussianDenoise(sigma=1.0),
        MedianDenoise(size=3),
        BilateralDenoise(sigma_color=0.1, sigma_space=2.0),
        NLMeans(patch_size=3, patch_distance=3, h=0.1),
        GapFillIDW(power=2.0, radius=4),
        GapFillNearest(max_distance=5),
        OutlierMask(method="zscore", k=3.0),
        ReplaceOutliers(method="mad", k=3.0, fill="interp"),
    ]
    for op in operators:
        config = op.get_config()
        # Round-trip through JSON; will raise if any value is not JSON-safe.
        rehydrated = json.loads(json.dumps(config))
        clone = type(op)(**rehydrated)
        assert clone.get_config() == config


def test_inverse_mnf_is_forbidden_in_yaml() -> None:
    """``InverseMNF`` holds a runtime reference to a fitted MNF, so it must
    flag itself as non-serialisable and report an empty config."""
    forward = MNF(n_components=2)
    inverse = InverseMNF(forward=forward)
    assert inverse.forbid_in_yaml is True
    assert inverse.get_config() == {}


# ----------------------------------------------------------------------------
# numpy-compat contract: every restore primitive is metadata-independent
# per-pixel/window math, so the operators must accept a plain np.ndarray
# and return a plain np.ndarray with the same values as the GeoTensor path.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "make_op",
    [
        pytest.param(lambda: DespeckleLee(window=3), id="DespeckleLee"),
        pytest.param(lambda: DestripeColumn(method="median"), id="DestripeColumn"),
        pytest.param(lambda: DenoisePCA(n_components=1), id="DenoisePCA"),
        pytest.param(lambda: MNF(n_components=1), id="MNF"),
        pytest.param(lambda: GaussianDenoise(sigma=1.0), id="GaussianDenoise"),
        pytest.param(lambda: MedianDenoise(size=3), id="MedianDenoise"),
        pytest.param(lambda: GapFillNearest(), id="GapFillNearest"),
        pytest.param(lambda: GapFillLaplacian(), id="GapFillLaplacian"),
        pytest.param(lambda: OutlierMask(method="mad", k=3.0), id="OutlierMask"),
        pytest.param(
            lambda: ReplaceOutliers(method="mad", k=3.0, fill="interp"),
            id="ReplaceOutliers",
        ),
    ],
)
def test_operators_accept_plain_ndarray(make_op) -> None:
    rng = np.random.default_rng(5)
    arr = rng.random((2, 6, 6))
    arr[0, 1, 1] = np.nan
    out_arr = make_op()(arr)
    assert type(out_arr) is np.ndarray
    # NaN fill so nodata pixels hold NaN on both carriers.
    out_gt = make_op()(toy_geotensor(arr, fill_value_default=np.nan))
    assert isinstance(out_gt, GeoTensor)
    np.testing.assert_array_equal(out_arr, np.asarray(out_gt))


def test_mnf_round_trip_accepts_plain_ndarray() -> None:
    rng = np.random.default_rng(6)
    arr = rng.normal(size=(3, 5, 5))
    forward = MNF(n_components=3)
    scores = forward(arr)
    assert type(scores) is np.ndarray
    restored = InverseMNF(forward=forward)(scores)
    assert type(restored) is np.ndarray
    np.testing.assert_allclose(restored, arr, atol=1e-10)


# ----------------------------------------------------------------------------
# Nodata (fill pixels) handling
# ----------------------------------------------------------------------------
def _fill_scene() -> tuple[GeoTensor, np.ndarray, np.ndarray]:
    """(3, 6, 7) positive scene with -9999 fill pixels, clean values, fill mask."""
    rng = np.random.default_rng(11)
    values = rng.uniform(1.0, 2.0, size=(3, 6, 7))
    gt = toy_geotensor(values, fill_value_default=-9999, with_fill_pixels=True)
    return gt, values, fill_pixel_mask(values.shape)


def _nan_at(values: np.ndarray, fill: np.ndarray) -> np.ndarray:
    out = values.copy()
    out[..., fill] = np.nan
    return out


_FILTERS = [
    pytest.param(lambda: DespeckleLee(window=3), id="DespeckleLee"),
    pytest.param(lambda: DespeckleFrost(window=3), id="DespeckleFrost"),
    pytest.param(lambda: DespeckleRefinedLee(window=3), id="DespeckleRefinedLee"),
    pytest.param(lambda: DestripeColumn(method="mean"), id="DestripeColumn"),
    pytest.param(lambda: MomentMatching(window=3), id="MomentMatching"),
    pytest.param(lambda: DenoisePCA(n_components=1), id="DenoisePCA"),
    pytest.param(lambda: MNF(n_components=2), id="MNF"),
    pytest.param(lambda: GaussianDenoise(sigma=1.0), id="GaussianDenoise"),
    pytest.param(lambda: MedianDenoise(size=3), id="MedianDenoise"),
    pytest.param(lambda: BilateralDenoise(sigma_space=1.0), id="BilateralDenoise"),
    pytest.param(lambda: NLMeans(), id="NLMeans"),
    pytest.param(lambda: ReplaceOutliers(fill="interp"), id="ReplaceOutliers"),
]


@pytest.mark.parametrize("make_op", _FILTERS)
def test_fill_pixels_are_excluded(make_op) -> None:
    """Fill pixels never enter a filter / fit and hold the output fill.

    Valid pixels must equal the result on the same data with the fill
    pixels marked missing (NaN) -- e.g. ``GaussianDenoise`` must not smear
    -9999 into the neighbours of a fill pixel.
    """
    gt, values, fill = _fill_scene()
    op = make_op()
    result = op(gt)
    out = np.asarray(result)

    if isinstance(op, MNF):
        # Component scores are a new quantity: NaN nodata (#146).
        assert np.isnan(result.fill_value_default)
        assert np.isnan(out[:, fill]).all()
    else:
        assert result.fill_value_default == -9999
        assert np.all(out[:, fill] == -9999)
    expected = np.asarray(make_op()(_nan_at(values, fill)))
    np.testing.assert_allclose(out[:, ~fill], expected[:, ~fill])
    # No fill leakage: every valid output stays in the data's range.
    assert out[:, ~fill].min() > -10.0


@pytest.mark.parametrize(
    "make_op",
    [
        pytest.param(lambda: GapFillNearest(), id="GapFillNearest"),
        pytest.param(lambda: GapFillIDW(radius=3), id="GapFillIDW"),
        pytest.param(lambda: GapFillLaplacian(), id="GapFillLaplacian"),
        pytest.param(lambda: GapFillInpaintBiharmonic(), id="GapFillBiharmonic"),
    ],
)
def test_fill_pixels_are_gaps_for_gap_fill(make_op) -> None:
    """Gap-fill operators fill fill pixels from valid neighbours only."""
    gt, values, fill = _fill_scene()
    out = np.asarray(make_op()(gt))

    expected = np.asarray(make_op()(_nan_at(values, fill)))
    np.testing.assert_allclose(out, expected)
    assert np.all(np.isfinite(out[:, fill]))
    assert np.all((out[:, fill] >= 1.0) & (out[:, fill] <= 2.0))
    np.testing.assert_array_equal(out[:, ~fill], values[:, ~fill])


def test_gap_fill_unfilled_gaps_hold_fill_value() -> None:
    values = np.ones((1, 6, 6))
    gt = toy_geotensor(values, fill_value_default=-9999, with_fill_pixels=True)
    out = np.asarray(GapFillNearest(max_distance=0)(gt))
    fill = fill_pixel_mask(values.shape)
    assert np.all(out[:, fill] == -9999)


def test_fill_pixels_are_excluded_from_masks_and_pca_fit() -> None:
    gt, values, fill = _fill_scene()

    # OutlierMask: the -9999 fill is not an outlier and does not skew stats.
    outliers = OutlierMask(method="zscore", k=3.0)(gt)
    assert not np.asarray(outliers).any()
    # A boolean flag declares False, not the input's -9999 (#146).
    assert outliers.fill_value_default is False

    # MNF: fitted on valid pixels only; the scores' NaN nodata carries
    # through the inverse.
    forward = MNF(n_components=3)
    scores = forward(gt)
    reference = MNF(n_components=3)
    reference(_nan_at(values, fill))
    np.testing.assert_allclose(forward.snr_, reference.snr_)
    restored = np.asarray(InverseMNF(forward=forward)(scores))
    assert np.isnan(restored[:, fill]).all()
    np.testing.assert_allclose(restored[:, ~fill], values[:, ~fill], atol=1e-10)


def test_4d_time_stack() -> None:
    """PCA transforms fit on the band axis (-3); 2-D maps are rejected (#147)."""
    from _helpers import time_stack

    stack = time_stack((2, 4, 6, 6))
    forward = MNF(n_components=2)
    scores = forward(stack)
    assert scores.shape == (2, 2, 6, 6)
    restored = InverseMNF(forward=forward)(scores)
    assert restored.shape == stack.shape
    full = MNF()
    np.testing.assert_allclose(
        np.asarray(InverseMNF(forward=full)(full(stack))), np.asarray(stack)
    )
    denoised = DenoisePCA(n_components=4)(stack)
    np.testing.assert_allclose(np.asarray(denoised), np.asarray(stack))
    # Rows of a 2-D map are never treated as bands.
    with pytest.raises(ValueError, match="DenoisePCA accepts 3-D"):
        DenoisePCA(n_components=1)(toy_geotensor(np.ones((4, 4))))
    with pytest.raises(ValueError, match="MNF accepts 3-D"):
        MNF(n_components=1)(toy_geotensor(np.ones((4, 4))))


# ----------------------------------------------------------------------------
# Reference tests for the named algorithms (#158): Lee (1980), Frost (1982),
# moment-matching destriping (Gadallah et al. 2000) and MNF (Green et al.
# 1988), each checked against a brute-force / closed-form reference.
# ----------------------------------------------------------------------------
def _windows(arr: np.ndarray, window: int):
    """Yield ``(row, col, neighbourhood, distances)`` with edge-repeat padding."""
    half = window // 2
    padded = np.pad(arr, ((half, window - 1 - half),) * 2, mode="edge")
    offsets = np.arange(window) - half
    dist = np.hypot(offsets[:, None], offsets[None, :])
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            yield row, col, padded[row : row + window, col : col + window], dist


def _lee_reference(arr: np.ndarray, window: int, cu: float) -> np.ndarray:
    """Per-pixel Lee (1980): x̂ = z̄ + k·(z − z̄), k = σ_x² / (σ_x² + Cu²·z̄²)."""
    out = np.full(arr.shape, np.nan)
    for row, col, hood, _ in _windows(arr, window):
        if np.isnan(arr[row, col]):
            continue
        mean, var = np.nanmean(hood), np.nanvar(hood)
        noise_var = cu**2 * mean**2
        signal_var = max((var - noise_var) / (1 + cu**2), 0.0)
        gain = signal_var / (signal_var + noise_var) if signal_var > 0 else 0.0
        out[row, col] = mean + gain * (arr[row, col] - mean)
    return out


def _frost_reference(arr: np.ndarray, window: int, damping: float) -> np.ndarray:
    """Per-pixel Frost (1982): Σ m·z / Σ m with m(t) = exp(−K·Cv²·|t|)."""
    out = np.full(arr.shape, np.nan)
    for row, col, hood, dist in _windows(arr, window):
        if np.isnan(arr[row, col]):
            continue
        valid = np.isfinite(hood)
        mean, var = np.nanmean(hood), np.nanvar(hood)
        cv_sq = var / mean**2 if mean != 0 else 0.0
        weights = np.exp(-damping * cv_sq * dist) * valid
        out[row, col] = np.sum(weights * np.where(valid, hood, 0.0)) / weights.sum()
    return out


def _speckled_scene(seed: int = 0) -> np.ndarray:
    """Two-level 4-look speckled scene with a bright block and one NaN."""
    rng = np.random.default_rng(seed)
    clean = np.where(np.arange(20)[None, :] < 9, 2.0, 8.0) * np.ones((18, 20))
    clean[5:8, 3:6] = 30.0
    noisy = clean * rng.gamma(shape=4.0, scale=0.25, size=clean.shape)
    noisy[4, 11] = np.nan
    return noisy


@pytest.mark.parametrize("window", [3, 4, 7])
def test_lee_matches_per_pixel_reference(window: int) -> None:
    noisy = _speckled_scene()
    out = despeckle_lee(noisy, window=window, cu=0.5)
    np.testing.assert_allclose(out, _lee_reference(noisy, window, 0.5), atol=1e-9)
    np.testing.assert_allclose(
        despeckle_refined_lee(noisy, window=window),
        _lee_reference(noisy, window, 0.523),
        atol=1e-9,
    )


def test_lee_gain_approaches_one_at_edges() -> None:
    """Lee's gain is ~1 at a step edge / point target and 0 on flat speckle.

    The old gain ``0.5·σ²/(σ² + Cu²·z̄²)`` could never exceed 0.5.
    """
    low, high, window, cu = 10.0, 100.0, 7, 0.1
    step = np.where(np.arange(32)[None, :] < 16, low, high) * np.ones((32, 32))
    out = despeckle_lee(step, window=window, cu=cu)
    # Closed form at the last low column: its window holds 4 low + 3 high.
    p = 3 / 7
    mean = low + p * (high - low)
    ci_sq = p * (1 - p) * (high - low) ** 2 / mean**2
    gain = (ci_sq - cu**2) / (ci_sq + cu**4)
    assert gain > 0.95
    np.testing.assert_allclose(out[:, 15], mean + gain * (low - mean), rtol=1e-12)
    assert np.all(np.abs(out[:, 15] - low) < 0.05 * (high - low))
    assert np.all(np.abs(out[:, 16] - high) < 0.05 * (high - low))

    # A point target passes through almost unchanged with the default Cu.
    point = np.ones((15, 15))
    point[7, 7] = 1000.0
    assert despeckle_lee(point)[7, 7] > 0.99 * 1000.0

    # Flat 4-look intensity speckle with the matching Cu = 1/√4: the local
    # Ci² is below Cu² for most pixels, so the gain is exactly 0 there.
    rng = np.random.default_rng(3)
    flat = rng.gamma(shape=4.0, scale=0.25, size=(96, 96))
    out = despeckle_lee(flat, window=window, cu=0.5)
    local_mean = ndimage.uniform_filter(flat, window, mode="nearest")
    effective_gain = (out - local_mean) / (flat - local_mean)
    assert np.median(effective_gain) < 1e-9
    assert np.percentile(effective_gain, 90) < 0.3


@pytest.mark.parametrize(("window", "damping"), [(3, 2.0), (4, 1.0), (7, 2.0)])
def test_frost_matches_per_pixel_reference(window: int, damping: float) -> None:
    noisy = _speckled_scene(1)
    out = despeckle_frost(noisy, window=window, damping=damping)
    np.testing.assert_allclose(out, _frost_reference(noisy, window, damping), atol=1e-9)


def test_frost_preserves_step_edge() -> None:
    """The Frost kernel narrows at an edge (high Cv²) and widens on flat areas.

    The old weighting did the opposite: at the edge it fell back to the box
    mean (0/100 step -> 38.6/64.7).
    """
    low, high = 10.0, 100.0
    step = np.where(np.arange(32)[None, :] < 16, low, high) * np.ones((32, 32))
    out = despeckle_frost(step, window=7, damping=2.0)
    box = ndimage.uniform_filter(step, 7, mode="nearest")
    frost_contrast = out[:, 16] - out[:, 15]
    box_contrast = box[:, 16] - box[:, 15]
    assert np.all(frost_contrast > 3 * box_contrast)
    assert np.all(frost_contrast > 0.4 * (high - low))
    # One pixel into the low side the kernel is already ~a delta.
    np.testing.assert_allclose(out[:, 14], low, atol=1.0)
    # Flat regions are untouched; more damping keeps more contrast.
    np.testing.assert_allclose(out[:, :12], low)
    sharper = despeckle_frost(step, window=7, damping=10.0)
    assert np.all(sharper[:, 16] - sharper[:, 15] > frost_contrast)

    # On homogeneous speckle the kernel is wide and the variance drops.
    rng = np.random.default_rng(4)
    flat = rng.gamma(shape=4.0, scale=0.25, size=(64, 64))
    assert np.var(despeckle_frost(flat, window=7)) < 0.1 * np.var(flat)


def _moment_matching_reference(arr: np.ndarray, window: int | None) -> np.ndarray:
    """Per-column gain/offset: (z − μ_j)·σᵣ_j/σ_j + μᵣ_j."""
    mu, sd = arr.mean(axis=0), arr.std(axis=0)
    width = arr.shape[1]
    out = np.empty_like(arr)
    for col in range(width):
        if window is None:
            neighbours = np.arange(width)
        else:
            half = window // 2
            neighbours = np.clip(
                np.arange(col - half, col - half + window), 0, width - 1
            )
        ref_mu, ref_sd = mu[neighbours].mean(), sd[neighbours].mean()
        out[:, col] = (arr[:, col] - mu[col]) * ref_sd / sd[col] + ref_mu
    return out


def test_moment_matching_preserves_detail() -> None:
    """Moment matching rescales each column; it must not blur the image.

    The old implementation replaced every pixel with its window mean
    (white-noise std 1.03 -> 0.053).
    """
    rng = np.random.default_rng(8)
    noise = rng.standard_normal((128, 96))
    out = destripe_column(noise, method="moment_matching")
    assert abs(out.std() / noise.std() - 1.0) < 0.03
    assert np.corrcoef(out.ravel(), noise.ravel())[0, 1] > 0.99

    # Synthetic detector stripes: per-column gain and offset.
    gain = rng.uniform(0.5, 1.5, size=96)
    offset = rng.uniform(-2.0, 2.0, size=96)
    striped = noise * gain + offset
    for window in (None, 1, 9, 21):
        out = destripe_column(striped, method="moment_matching", window=window)
        np.testing.assert_allclose(
            out, _moment_matching_reference(striped, window), atol=1e-10
        )
    out = destripe_column(striped, method="moment_matching", window=None)
    # Every column now has the same moments ...
    np.testing.assert_allclose(out.mean(axis=0), out.mean(), atol=1e-12)
    np.testing.assert_allclose(out.std(axis=0), out.std(axis=0)[0], rtol=1e-12)
    # ... and is an exact affine image of the clean detail.
    standardised = (noise - noise.mean(axis=0)) / noise.std(axis=0)
    expected = standardised * striped.std(axis=0).mean() + striped.mean(axis=0).mean()
    np.testing.assert_allclose(out, expected, atol=1e-10)
    # Rows: the transposed problem.
    np.testing.assert_allclose(
        destripe_column(striped.T, method="moment_matching", axis="row", window=9),
        _moment_matching_reference(striped, 9).T,
        atol=1e-10,
    )


def _mnf_scene(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """5-band cube = 2 smooth sources mixed into 5 bands + known coloured noise.

    Returns ``(noisy, clean, sources, noise_cov)``.
    """
    rng = np.random.default_rng(seed)
    bands, size = 5, 128
    sources = np.stack(
        [
            ndimage.gaussian_filter(rng.standard_normal((size, size)), 16.0)
            for _ in range(2)
        ]
    )
    sources /= sources.std(axis=(1, 2), keepdims=True)
    sources *= np.array([2.0, 1.0])[:, None, None]
    mixing = rng.normal(size=(bands, 2))
    clean = np.einsum("bk,khw->bhw", mixing, sources)
    chol = np.tril(rng.normal(scale=0.3, size=(bands, bands)))
    chol[np.diag_indices(bands)] = [0.3, 0.8, 0.5, 1.2, 0.6]
    noise = np.einsum("bc,chw->bhw", chol, rng.standard_normal((bands, size, size)))
    return clean + noise, clean, sources, chol @ chol.T


def test_mnf_matches_generalized_eigenproblem() -> None:
    """MNF == eigh(Σ, Σ_N) with Σ_N from horizontal shift differences."""
    from scipy.linalg import eigh

    noisy, _, _, _ = _mnf_scene()
    forward = MNF()
    forward(noisy)
    state = forward._state
    assert state is not None

    samples = noisy.reshape(noisy.shape[0], -1)
    sigma = np.cov(samples)
    diff = (noisy[:, :, 1:] - noisy[:, :, :-1]).reshape(noisy.shape[0], -1)
    noise_cov = np.cov(diff) / 2.0
    eigvals, eigvecs = eigh(sigma, noise_cov)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    np.testing.assert_allclose(state["noise_covariance"], noise_cov, rtol=1e-10)
    np.testing.assert_allclose(forward.eigenvalues_, eigvals, rtol=1e-8)
    np.testing.assert_allclose(forward.snr_, eigvals - 1.0, rtol=1e-8, atol=1e-10)
    components = np.asarray(state["components"])
    signs = np.sign(np.sum(components * eigvecs, axis=0))
    np.testing.assert_allclose(components * signs, eigvecs, atol=1e-8)
    # Scores have unit noise variance: Aᵀ·Σ_N·A = I.
    np.testing.assert_allclose(
        components.T @ noise_cov @ components, np.eye(5), atol=1e-8
    )


def test_mnf_recovers_signal_subspace_ordered_by_snr() -> None:
    from scipy.linalg import eigh

    noisy, clean, sources, noise_cov = _mnf_scene()
    forward = MNF()
    scores = np.asarray(forward(noisy))
    eigenvalues = np.asarray(forward.eigenvalues_)
    # Shift differences recover the true noise covariance (up to the small
    # leak of the smooth signal's pixel-to-pixel increments).
    estimated = np.asarray(forward._state["noise_covariance"])
    assert np.abs(estimated - noise_cov).max() < 0.02 * np.trace(noise_cov)
    # Two signal components, three pure-noise components (λ ≈ 1, SNR ≈ 0).
    true_eigs = eigh(np.cov(clean.reshape(5, -1)) + noise_cov, noise_cov)[0][::-1]
    np.testing.assert_allclose(eigenvalues[:2], true_eigs[:2], rtol=0.15)
    assert eigenvalues[1] > 10 * eigenvalues[2]
    np.testing.assert_allclose(eigenvalues[2:], 1.0, atol=0.1)
    # The top-2 scores span the source subspace.
    design = scores[:2].reshape(2, -1).T
    for source in sources:
        target = source.ravel()
        coef, *_ = np.linalg.lstsq(design, target - target.mean(), rcond=None)
        residual = target - target.mean() - design @ coef
        assert 1 - residual.var() / target.var() > 0.9
    # Inverting from the signal components denoises.
    reduced = MNF(n_components=2)
    denoised = np.asarray(InverseMNF(forward=reduced)(reduced(noisy)))
    rmse_denoised = np.sqrt(np.mean((denoised - clean) ** 2))
    rmse_noisy = np.sqrt(np.mean((noisy - clean) ** 2))
    assert rmse_denoised < 0.5 * rmse_noisy
    # Unlike PCA, MNF ranks by SNR, not variance: with coloured band noise
    # the 2-component PCA reconstruction keeps far more noise.
    pca = np.asarray(DenoisePCA(n_components=2)(noisy))
    assert rmse_denoised < 0.7 * np.sqrt(np.mean((pca - clean) ** 2))


def test_mnf_rejects_noise_free_band() -> None:
    arr = np.random.default_rng(0).normal(size=(3, 8, 8))
    arr[1] = 1.0
    with pytest.raises(ValueError, match="noise covariance is singular"):
        MNF()(arr)


def test_fit_pca_explained_variance_matches_sklearn() -> None:
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(12)
    cube = np.einsum(
        "bk,khw->bhw", rng.normal(size=(4, 4)), rng.normal(size=(4, 9, 11))
    )
    cube[2, 3, 4] = np.nan  # excluded from the fit
    state = fit_pca(cube, n_components=3)
    samples = cube.reshape(4, -1).T
    reference = PCA(n_components=3).fit(samples[np.isfinite(samples).all(axis=1)])
    np.testing.assert_allclose(
        state["explained_variance"], reference.explained_variance_, rtol=1e-10
    )
    components = np.asarray(state["components"])
    signs = np.sign(np.sum(components * reference.components_.T, axis=0))
    np.testing.assert_allclose(components * signs, reference.components_.T, atol=1e-10)


def test_gap_fill_laplacian_operator_exposes_iterations() -> None:
    arr = np.zeros((9, 9))
    arr[:, 7:] = 10.0
    arr[2:7, 2:7] = np.nan
    one = np.asarray(GapFillLaplacian(iterations=1)(arr))
    many = np.asarray(GapFillLaplacian(iterations=500)(arr))
    np.testing.assert_array_equal(one, gap_fill_laplacian(arr, iterations=1))
    np.testing.assert_array_equal(many, gap_fill_laplacian(arr, iterations=500))
    assert not np.allclose(one, many)
    assert GapFillLaplacian(iterations=7).get_config() == {"iterations": 7}
