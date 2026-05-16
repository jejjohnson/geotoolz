"""Tests for `geotoolz.spectral` band-space operators."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geotoolz import spectral


def _toy_geotensor(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs="EPSG:32629",
        fill_value_default=-9999,
        attrs={
            "band_names": ["B2", "B4", "B8", "B11"],
            "wavelengths": [490.0, 665.0, 842.0, 1610.0],
        },
    )


def test_select_bands_by_name_and_index_match() -> None:
    gt = _toy_geotensor(np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3))
    by_name = spectral.SelectBands(indexes=["B8", "B4"])(gt)
    by_index = spectral.SelectBands(indexes=[2, 1])(gt)

    np.testing.assert_array_equal(np.asarray(by_name), np.asarray(by_index))
    assert by_name.transform == gt.transform
    assert str(by_name.crs) == str(gt.crs)
    assert by_name.attrs["band_names"] == ["B8", "B4"]
    assert by_name.attrs["wavelengths"] == [842.0, 665.0]


def test_reorder_bands_preserves_band_metadata() -> None:
    gt = _toy_geotensor(np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2))
    out = spectral.ReorderBands(order=["B11", "B2"])(gt)

    np.testing.assert_array_equal(np.asarray(out), np.asarray(gt)[[3, 0]])
    assert out.attrs["band_names"] == ["B11", "B2"]
    assert out.attrs["wavelengths"] == [1610.0, 490.0]


def test_band_math_matches_normalized_difference() -> None:
    gt = _toy_geotensor(
        np.array([[[0.2]], [[0.4]], [[0.8]], [[0.6]]], dtype=np.float32)
    )

    via_math = spectral.BandMath(expression="(B8 - B4) / (B8 + B4 + 1e-6)")(gt)
    via_op = spectral.NormalizedDifference(a="B8", b="B4", eps=1e-6)(gt)

    np.testing.assert_allclose(np.asarray(via_math), np.asarray(via_op), rtol=1e-6)


def test_band_math_unknown_band_raises_clear_error() -> None:
    gt = _toy_geotensor(np.ones((4, 2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="Unknown band name"):
        spectral.BandMath(expression="B99 + B8")(gt)


def test_band_math_allows_safe_numpy_functions() -> None:
    gt = _toy_geotensor(
        np.array([[[0.2]], [[0.4]], [[0.8]], [[0.6]]], dtype=np.float32)
    )

    out = spectral.BandMath(expression="sqrt(B8)")(gt)

    np.testing.assert_allclose(np.asarray(out), np.sqrt(0.8))


def test_band_ratio_by_name() -> None:
    gt = _toy_geotensor(
        np.array([[[0.2]], [[0.4]], [[0.8]], [[0.6]]], dtype=np.float32)
    )

    out = spectral.BandRatio(numerator="B8", denominator="B4", eps=0.0)(gt)

    np.testing.assert_allclose(np.asarray(out), 2.0)


def test_apply_srf_preserves_values_when_source_equals_target() -> None:
    wavelengths = np.array([490.0, 560.0, 665.0], dtype=float)
    values = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
    gt = GeoTensor(
        values=values,
        transform=rasterio.Affine.identity(),
        crs="EPSG:32629",
        fill_value_default=-9999,
    )

    out = spectral.ApplySRF(
        target_center_wavelengths=wavelengths,
        target_fwhm=np.ones(3),
        source_wavelengths=wavelengths,
    )(gt)
    np.testing.assert_allclose(np.asarray(out), values, atol=1e-5)

    flat = GeoTensor(
        values=np.ones((3, 2, 2), dtype=np.float32) * 0.3,
        transform=gt.transform,
        crs=gt.crs,
        fill_value_default=-9999,
    )
    flat_out = spectral.ApplySRF(
        target_center_wavelengths=wavelengths,
        target_fwhm=np.ones(3),
        source_wavelengths=wavelengths,
    )(flat)
    np.testing.assert_allclose(np.asarray(flat_out), 0.3, rtol=0.01)
    assert flat_out.transform == flat.transform
    assert str(flat_out.crs) == str(flat.crs)


def test_gaussian_srf_reads_source_wavelengths_from_attrs() -> None:
    gt = _toy_geotensor(np.ones((4, 2, 2), dtype=np.float32))

    out = spectral.GaussianSRF(
        target_center_wavelengths=[490.0, 665.0],
        target_fwhm=[1.0, 1.0],
        band_names=["blue", "red"],
    )(gt)

    assert out.shape == (2, 2, 2)
    assert out.attrs["band_names"] == ["blue", "red"]


def test_continuum_removal_flat_and_absorption() -> None:
    wavelengths = np.array([2100.0, 2200.0, 2300.0])
    flat = GeoTensor(
        values=np.ones((3, 1, 1), dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32629",
        attrs={"wavelengths": wavelengths.tolist()},
    )
    np.testing.assert_allclose(
        np.asarray(spectral.ContinuumRemoval(method="convex_hull")(flat)),
        1.0,
    )

    absorption = GeoTensor(
        values=np.array([[[1.0]], [[0.8]], [[1.0]]], dtype=np.float32),
        transform=flat.transform,
        crs=flat.crs,
        attrs={"wavelengths": wavelengths.tolist()},
    )
    removed = spectral.ContinuumRemoval(method="convex_hull")(absorption)
    np.testing.assert_allclose(np.asarray(removed)[1, 0, 0], 0.8, rtol=0.05)


def test_continuum_removal_linear_method() -> None:
    gt = GeoTensor(
        values=np.array([[[1.0]], [[0.75]], [[0.5]]], dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32629",
        attrs={"wavelengths": [1.0, 2.0, 3.0]},
    )

    np.testing.assert_allclose(
        np.asarray(spectral.ContinuumRemoval(method="linear")(gt)),
        1.0,
    )


def test_spectral_binning_and_smoothing_preserve_spatial_metadata() -> None:
    gt = _toy_geotensor(np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2))

    binned = spectral.SpectralBinning(
        target_wavelengths=[577.5, 1226.0],
        width=[200.0, 900.0],
        method="mean",
    )(gt)
    assert binned.shape == (2, 2, 2)
    assert binned.transform == gt.transform
    np.testing.assert_allclose(
        np.asarray(binned)[0], np.mean(np.asarray(gt)[:2], axis=0)
    )

    smoothed = spectral.SpectralSmoothing(method="moving_average", window=3)(gt)
    assert smoothed.shape == gt.shape
    assert smoothed.transform == gt.transform


def test_spectral_binning_median_and_weighted_mean() -> None:
    gt = _toy_geotensor(np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2))

    median = spectral.SpectralBinning(
        target_wavelengths=[577.5],
        width=200.0,
        method="median",
    )(gt)
    weighted = spectral.SpectralBinning(
        target_wavelengths=[577.5],
        width=200.0,
        method="weighted_mean",
    )(gt)

    assert median.shape == (1, 2, 2)
    assert weighted.shape == (1, 2, 2)


def test_spectral_smoothing_savgol_and_gaussian() -> None:
    gt = _toy_geotensor(np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2))

    savgol = spectral.SpectralSmoothing(method="savgol", window=3, polyorder=1)(gt)
    gaussian = spectral.SpectralSmoothing(method="gaussian", window=3)(gt)

    assert savgol.shape == gt.shape
    assert gaussian.shape == gt.shape


def test_stack_and_split_bands() -> None:
    gt = _toy_geotensor(np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2))
    split = spectral.SplitBands()(gt)
    assert len(split) == 4
    assert split[0].shape == (1, 2, 2)
    assert split[0].attrs["band_names"] == ["B2"]

    stacked = spectral.StackBands()(split[:2])
    np.testing.assert_array_equal(np.asarray(stacked), np.asarray(gt)[:2])
    assert stacked.transform == gt.transform


def test_spectral_get_config_serialization() -> None:
    ops_and_configs = [
        (spectral.SelectBands(indexes=[0]), {"indexes": [0], "axis": 0}),
        (spectral.ReorderBands(order=[0]), {"order": [0], "axis": 0}),
        (spectral.StackBands(), {"axis": 0, "along": "band"}),
        (spectral.SplitBands(), {"names": None, "axis": 0}),
        (
            spectral.BandMath(expression="B0"),
            {"expression": "B0", "band_names": None, "axis": 0},
        ),
        (
            spectral.NormalizedDifference(a=0, b=1),
            {"a": 0, "b": 1, "eps": 1e-6, "axis": 0},
        ),
        (
            spectral.BandRatio(numerator=0, denominator=1),
            {"numerator": 0, "denominator": 1, "eps": 1e-6, "axis": 0},
        ),
        (
            spectral.ApplySRF(
                target_center_wavelengths=[1.0],
                target_fwhm=[1.0],
                source_wavelengths=[1.0],
            ),
            {
                "target_center_wavelengths": [1.0],
                "target_fwhm": [1.0],
                "source_wavelengths": [1.0],
                "band_names": None,
            },
        ),
        (
            spectral.GaussianSRF(target_center_wavelengths=[1.0], target_fwhm=[1.0]),
            {
                "target_center_wavelengths": [1.0],
                "target_fwhm": [1.0],
                "source_wavelengths": None,
                "band_names": None,
            },
        ),
        (
            spectral.ContinuumRemoval(wavelengths=[1.0, 2.0, 3.0]),
            {"method": "convex_hull", "wavelengths": [1.0, 2.0, 3.0], "axis": 0},
        ),
        (
            spectral.SpectralBinning(target_wavelengths=[1.0], width=1.0),
            {
                "target_wavelengths": [1.0],
                "width": [1.0],
                "method": "mean",
                "source_wavelengths": None,
                "axis": 0,
            },
        ),
        (
            spectral.SpectralSmoothing(),
            {"method": "savgol", "window": 7, "polyorder": 2, "axis": 0},
        ),
    ]

    for op, expected in ops_and_configs:
        assert op.get_config() == expected
