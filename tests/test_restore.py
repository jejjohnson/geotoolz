"""Tests for `geotoolz.restore`."""

from __future__ import annotations

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

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
    SaturationFlag,
    bilateral_denoise,
    despeckle_lee,
    destripe_column,
    gap_fill_idw,
    gap_fill_laplacian,
    gap_fill_nearest,
    median_denoise,
    nl_means,
    outlier_mask,
)


def _toy_geotensor(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs="EPSG:32629",
        fill_value_default=np.nan,
    )


def test_restore_namespace_is_available() -> None:
    assert gz.restore.DespeckleLee is DespeckleLee
    assert gz.DespeckleLee is DespeckleLee


def test_despeckle_lee_reduces_multiplicative_speckle_variance() -> None:
    rng = np.random.default_rng(0)
    clean = np.ones((64, 64), dtype=float)
    noisy = clean * rng.gamma(shape=1.0, scale=1.0, size=clean.shape)
    out = despeckle_lee(noisy, window=9)
    assert np.nanvar(out) <= 0.5 * np.nanvar(noisy)
    np.testing.assert_allclose(np.nanmean(out), np.nanmean(noisy), rtol=0.05)
    assert np.isfinite(out[[0, -1], :]).all()
    assert np.isfinite(out[:, [0, -1]]).all()


def test_despeckle_operator_preserves_metadata() -> None:
    rng = np.random.default_rng(1)
    gt = _toy_geotensor(rng.random((2, 8, 8)).astype(np.float32))
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
    assert rmse < 0.01


def test_destripe_operator_preserves_metadata() -> None:
    gt = _toy_geotensor(np.ones((2, 8, 8), dtype=np.float32))
    out = DestripeColumn()(gt)
    assert isinstance(out, GeoTensor)
    assert out.shape == gt.shape
    assert out.transform == gt.transform
    assert MomentMatching(window=3)(gt).shape == gt.shape


def test_mnf_inverse_with_all_components_is_identity() -> None:
    rng = np.random.default_rng(2)
    arr = rng.normal(size=(4, 10, 10)).astype(np.float32)
    gt = _toy_geotensor(arr)
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
    gt = _toy_geotensor(arr)
    out = DenoisePCA(n_components=2)(gt)
    assert out.shape == gt.shape
    assert out.transform == gt.transform


def test_gaussian_denoise_preserves_nan_mask_and_metadata() -> None:
    arr = np.arange(25, dtype=float).reshape(5, 5)
    arr[2, 2] = np.nan
    gt = _toy_geotensor(arr)
    out = GaussianDenoise(sigma=1.0)(gt)
    assert np.isnan(np.asarray(out)[2, 2])
    assert out.transform == gt.transform


def test_single_band_denoisers_run_and_preserve_shape() -> None:
    arr = np.arange(25, dtype=float).reshape(5, 5)
    gt = _toy_geotensor(arr)
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
    gt = _toy_geotensor(arr)
    out = GapFillInpaintBiharmonic()(gt)
    assert np.isfinite(np.asarray(out)[2, 2])
    np.testing.assert_array_equal(
        np.asarray(out)[np.isfinite(arr)], arr[np.isfinite(arr)]
    )


def test_gap_fill_laplacian_fills_nan() -> None:
    arr = np.arange(25, dtype=float).reshape(5, 5)
    arr[2, 2] = np.nan
    gt = _toy_geotensor(arr)
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
    gt = _toy_geotensor(arr)
    nearest = GapFillNearest(max_distance=2)(gt)
    idw = GapFillIDW(power=128.0, radius=2)(gt)
    np.testing.assert_array_equal(np.asarray(idw), np.asarray(nearest))
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
    gt = _toy_geotensor(arr)
    op_mask = OutlierMask(method="mad", k=3.0)(gt)
    np.testing.assert_array_equal(np.asarray(op_mask), mask)
    replaced = ReplaceOutliers(method="mad", k=3.0, fill="median")(gt)
    assert np.asarray(replaced)[1, 2] == 1.0


def test_saturation_flag_uses_dtype_max() -> None:
    arr = np.array([[0, 255]], dtype=np.uint8)
    out = SaturationFlag()(_toy_geotensor(arr))
    np.testing.assert_array_equal(np.asarray(out), [[False, True]])
    thresholded = SaturationFlag(threshold=0.5)(
        _toy_geotensor(np.array([[0.25, 0.75]]))
    )
    np.testing.assert_array_equal(np.asarray(thresholded), [[False, True]])
