"""Tests for `geotoolz.normalize`."""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz
from geotoolz.normalize import (
    AsinhScale,
    HistogramMatch,
    HistogramStretch,
    LogScale,
    MinMaxScaler,
    Normalize,
    PerBandStats,
    PercentileClip,
    PowerScale,
    RobustScaler,
    StandardScaler,
    ZeroOne,
)


def _toy_geotensor(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs="EPSG:32629",
        fill_value_default=np.nan,
    )


@pytest.fixture
def scene() -> GeoTensor:
    arr = np.stack(
        [
            np.arange(100, dtype=float).reshape(10, 10),
            np.arange(100, 200, dtype=float).reshape(10, 10),
        ]
    )
    arr[0, 0, 0] = np.nan
    return _toy_geotensor(arr)


def assert_metadata_preserved(out: GeoTensor, src: GeoTensor) -> None:
    assert out.shape == src.shape
    assert out.transform == src.transform
    assert str(out.crs) == str(src.crs)


def test_per_band_stats_caches_nan_aware_stats(scene: GeoTensor) -> None:
    op = PerBandStats(percentiles=[2.0, 98.0])
    out = op(scene)

    assert out is scene
    np.testing.assert_allclose(
        op.stats["mean"], np.nanmean(np.asarray(scene), axis=(-2, -1))
    )
    assert op.stats["percentiles"][0][0] == pytest.approx(
        np.nanpercentile(np.asarray(scene)[0], 2.0)
    )


def test_standard_scaler_fit_inverse_state_roundtrip(scene: GeoTensor) -> None:
    scaler = StandardScaler(fit_on_call=True)
    scaled = scaler(scene)
    restored = scaler.inverse(scaled)

    assert_metadata_preserved(scaled, scene)
    np.testing.assert_allclose(np.asarray(restored), np.asarray(scene), equal_nan=True)

    state = json.loads(json.dumps(scaler.state))
    restored_scaler = gz.Operator.from_state(state)
    np.testing.assert_allclose(
        np.asarray(restored_scaler(scene)), np.asarray(scaled), equal_nan=True
    )


def test_normalize_convenience_matches_standard_scaler(scene: GeoTensor) -> None:
    arr = np.asarray(scene)
    mean = np.nanmean(arr, axis=(-2, -1))
    std = np.nanstd(arr, axis=(-2, -1))

    expected = StandardScaler(mean=mean, std=std)(scene)
    actual = Normalize(mean=mean, std=std)(scene)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), equal_nan=True)


def test_minmax_scaler_fit_on_call_yields_zero_one_non_nan(scene: GeoTensor) -> None:
    out = MinMaxScaler(fit_on_call=True)(scene)
    arr = np.asarray(out)

    assert_metadata_preserved(out, scene)
    np.testing.assert_allclose(np.nanmin(arr, axis=(-2, -1)), [0.0, 0.0])
    np.testing.assert_allclose(np.nanmax(arr, axis=(-2, -1)), [1.0, 1.0])
    assert np.isnan(arr[0, 0, 0])


def test_robust_scaler_fit_ignores_nan(scene: GeoTensor) -> None:
    out = RobustScaler(fit_on_call=True)(scene)
    arr = np.asarray(out)

    np.testing.assert_allclose(np.nanmedian(arr, axis=(-2, -1)), [0.0, 0.0])
    assert np.isnan(arr[0, 0, 0])


def test_percentile_clip_counts_boundary_pixels() -> None:
    gt = _toy_geotensor(np.arange(100, dtype=float).reshape(1, 10, 10))
    out = PercentileClip(lower=2.0, upper=98.0)(gt)
    arr = np.asarray(out)

    assert np.count_nonzero(arr == 0.0) == 2
    assert np.count_nonzero(arr == 1.0) == 2
    assert np.nanmin(arr) == pytest.approx(0.0)
    assert np.nanmax(arr) == pytest.approx(1.0)
    assert 0.0 < arr[0, 5, 0] < 1.0


def test_histogram_stretch_maps_to_output_range(scene: GeoTensor) -> None:
    out = HistogramStretch(lower=0.0, upper=100.0, out_range=(0.0, 255.0))(scene)

    np.testing.assert_allclose(np.nanmin(np.asarray(out), axis=(-2, -1)), [0.0, 0.0])
    np.testing.assert_allclose(
        np.nanmax(np.asarray(out), axis=(-2, -1)), [255.0, 255.0]
    )


def test_histogram_match_approximates_reference_cdf() -> None:
    source = _toy_geotensor(np.linspace(0.0, 1.0, 200).reshape(1, 20, 10))
    reference = _toy_geotensor(np.linspace(10.0, 20.0, 200).reshape(1, 20, 10))

    matched = HistogramMatch(reference=reference)(source)

    np.testing.assert_allclose(
        np.nanquantile(np.asarray(matched), [0.25, 0.5, 0.75]),
        [12.5, 15.0, 17.5],
        atol=0.1,
    )


def test_nonlinear_scales_handle_zero_without_infinity(scene: GeoTensor) -> None:
    gt = _toy_geotensor(np.array([[[0.0, 1.0], [4.0, np.nan]]]))

    for op in (LogScale(), AsinhScale(), PowerScale()):
        out = op(gt)
        arr = np.asarray(out)
        assert_metadata_preserved(out, gt)
        assert np.isfinite(arr[0, 0, 0])
        assert np.isnan(arr[0, 1, 1])


def test_zero_one_global_and_module_export(scene: GeoTensor) -> None:
    out = ZeroOne(per_band=False)(scene)

    assert gz.normalize.ZeroOne is ZeroOne
    assert np.nanmin(out) == pytest.approx(0.0)
    assert np.nanmax(out) == pytest.approx(1.0)
