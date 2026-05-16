"""Tests for `geotoolz.normalize`."""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz
from geotoolz.normalize import (
    CLAHE,
    AsinhScale,
    HistogramMatch,
    HistogramStretch,
    LogScale,
    MinMaxScaler,
    Normalize,
    PerBandStats,
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


def test_standard_scaler_fit_inverse_state_roundtrip(
    scene: GeoTensor,
) -> None:
    scaler = StandardScaler(fit_on_call=True)
    scaled = scaler(scene)
    restored = scaler.inverse(scaled)

    assert_metadata_preserved(scaled, scene)
    np.testing.assert_allclose(np.asarray(restored), np.asarray(scene), equal_nan=True)

    state = json.loads(json.dumps(scaler.state))
    restored_scaler = gz.Operator.from_state(state)
    np.testing.assert_allclose(
        np.asarray(restored_scaler(scene)),
        np.asarray(scaled),
        equal_nan=True,
    )


def test_standard_scaler_fixed_stats_analytic() -> None:
    """Per-band fixed stats: ``(x - mu) / sigma`` matches analytic."""
    arr = np.stack(
        [
            np.array([[0.0, 2.0], [4.0, 6.0]]),  # mu=3, std=sqrt(5)
            np.array([[10.0, 11.0], [12.0, 13.0]]),  # mu=11.5, sd=sqrt(1.25)
        ]
    )
    gt = _toy_geotensor(arr)
    mu = np.array([3.0, 11.5])
    sd = np.array([np.sqrt(5.0), np.sqrt(1.25)])
    out = np.asarray(StandardScaler(mean=mu, std=sd)(gt))

    expected = (arr - mu[:, None, None]) / sd[:, None, None]
    np.testing.assert_allclose(out, expected)


def test_standard_scaler_zero_sigma_no_inf() -> None:
    """A constant band with ``sigma=0`` falls back to divisor=1."""
    arr = np.stack([np.full((4, 4), 7.0), np.arange(16, dtype=float).reshape(4, 4)])
    gt = _toy_geotensor(arr)
    op = StandardScaler(fit_on_call=True)
    out = np.asarray(op(gt))
    assert np.all(np.isfinite(out))
    # Constant band -> std==0 -> output == arr - mu == 0
    np.testing.assert_allclose(out[0], 0.0)


def test_normalize_convenience_matches_standard_scaler(
    scene: GeoTensor,
) -> None:
    arr = np.asarray(scene)
    mean = np.nanmean(arr, axis=(-2, -1))
    std = np.nanstd(arr, axis=(-2, -1))

    expected = StandardScaler(mean=mean, std=std)(scene)
    actual = Normalize(mean=mean, std=std)(scene)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), equal_nan=True)


def test_normalize_state_roundtrip_through_json(scene: GeoTensor) -> None:
    """``Normalize`` state must round-trip through JSON."""
    op = Normalize(mean=[5.0, 105.0], std=[2.0, 3.0])
    state = json.loads(json.dumps(op.state))
    restored = gz.Operator.from_state(state)
    np.testing.assert_allclose(
        np.asarray(restored(scene)), np.asarray(op(scene)), equal_nan=True
    )


def test_minmax_scaler_fit_on_call_yields_zero_one_non_nan(
    scene: GeoTensor,
) -> None:
    out = MinMaxScaler(fit_on_call=True)(scene)
    arr = np.asarray(out)

    assert_metadata_preserved(out, scene)
    np.testing.assert_allclose(np.nanmin(arr, axis=(-2, -1)), [0.0, 0.0])
    np.testing.assert_allclose(np.nanmax(arr, axis=(-2, -1)), [1.0, 1.0])
    assert np.isnan(arr[0, 0, 0])


def test_minmax_scaler_out_range_analytic() -> None:
    """Verify ``minmax`` mapping into a non-default output range."""
    arr = np.array([[[0.0, 1.0], [2.0, 4.0]]])
    gt = _toy_geotensor(arr)
    out = np.asarray(MinMaxScaler(vmin=[0.0], vmax=[4.0], out_range=(0.0, 255.0))(gt))
    # 0->0, 4->255, 1-> 63.75, 2->127.5
    np.testing.assert_allclose(out, np.array([[[0.0, 63.75], [127.5, 255.0]]]))


def test_robust_scaler_fit_ignores_nan(scene: GeoTensor) -> None:
    out = RobustScaler(fit_on_call=True)(scene)
    arr = np.asarray(out)

    np.testing.assert_allclose(np.nanmedian(arr, axis=(-2, -1)), [0.0, 0.0])
    assert np.isnan(arr[0, 0, 0])


def test_robust_scaler_fixed_stats_analytic() -> None:
    """Per-band ``(x - median) / iqr`` matches analytic."""
    arr = np.stack(
        [
            np.arange(8, dtype=float).reshape(2, 4),
            np.arange(8, 16, dtype=float).reshape(2, 4),
        ]
    )
    gt = _toy_geotensor(arr)
    # Band 0 median=3.5, iqr=4; band 1 median=11.5, iqr=4
    op = RobustScaler(median=[3.5, 11.5], iqr=[4.0, 4.0])
    out = np.asarray(op(gt))
    expected = (arr - np.array([3.5, 11.5])[:, None, None]) / 4.0
    np.testing.assert_allclose(out, expected)


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


def test_histogram_match_forbidden_in_yaml() -> None:
    """HistogramMatch holds a live reference; must opt out of YAML."""
    assert HistogramMatch.forbid_in_yaml is True


def test_clahe_preserves_nan_mask_and_metadata() -> None:
    gt = _toy_geotensor(np.linspace(0.0, 1.0, 100).reshape(1, 10, 10))
    np.asarray(gt)[0, 0, 0] = np.nan

    out = CLAHE(kernel_size=(4, 4), clip_limit=0.03)(gt)

    assert_metadata_preserved(out, gt)
    assert np.isnan(np.asarray(out)[0, 0, 0])
    assert np.nanmin(np.asarray(out)) >= 0.0
    assert np.nanmax(np.asarray(out)) <= 1.0


def test_nonlinear_scales_handle_zero_without_infinity(
    scene: GeoTensor,
) -> None:
    del scene  # fixture provides a sanity GeoTensor; we use a tiny one below
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


def test_get_config_is_json_safe() -> None:
    """All normaliser configs must JSON-serialise without errors."""
    ops = [
        StandardScaler(mean=np.array([1.0, 2.0]), std=np.array([0.5, 0.6])),
        RobustScaler(median=np.array([1.0, 2.0]), iqr=np.array([3.0, 4.0])),
        MinMaxScaler(
            vmin=np.array([0.0, 0.0]),
            vmax=np.array([1.0, 1.0]),
            out_range=(0.0, 255.0),
        ),
        Normalize(mean=np.array([1.0]), std=np.array([0.5])),
        HistogramStretch(),
        LogScale(),
        AsinhScale(),
        PowerScale(),
        ZeroOne(),
    ]
    for op in ops:
        json.dumps(op.get_config())  # must not raise


# ---------------------------------------------------------------------------
# hydra-zen round-trip
# ---------------------------------------------------------------------------


try:
    import hydra_zen
except ImportError:  # pragma: no cover - exercised via the [hydra] extra
    hydra_zen = None  # type: ignore[assignment]


@pytest.mark.skipif(hydra_zen is None, reason="requires hydra-zen extra")
@pytest.mark.parametrize(
    "op",
    [
        Normalize(mean=[0.1, 0.2], std=[0.05, 0.07]),
        StandardScaler(mean=[0.1, 0.2], std=[0.05, 0.07]),
        RobustScaler(median=[1.0, 2.0], iqr=[0.5, 0.6]),
        MinMaxScaler(vmin=[0.0, 0.0], vmax=[1.0, 1.0]),
        HistogramStretch(lower=2.0, upper=98.0),
        LogScale(),
        AsinhScale(a=0.5),
        PowerScale(gamma=0.4),
        ZeroOne(per_band=True),
    ],
)
def test_normalize_hydra_zen_roundtrip(op: object) -> None:
    cfg = hydra_zen.builds(type(op), **op.get_config())  # type: ignore[attr-defined]
    restored = hydra_zen.instantiate(cfg)
    assert type(restored) is type(op)
    assert restored.get_config() == op.get_config()  # type: ignore[attr-defined]


def test_normalize_geotensor_metadata_preserved(scene: GeoTensor) -> None:
    """Tier-B operators must preserve transform / CRS / shape."""
    ops = [
        StandardScaler(mean=[5.0, 105.0], std=[2.0, 3.0]),
        RobustScaler(median=[50.0, 150.0], iqr=[25.0, 25.0]),
        MinMaxScaler(vmin=[0.0, 100.0], vmax=[99.0, 199.0]),
        Normalize(mean=[5.0, 105.0], std=[2.0, 3.0]),
        HistogramStretch(),
        LogScale(),
        AsinhScale(),
        PowerScale(),
        ZeroOne(),
    ]
    for op in ops:
        out = op(scene)
        assert_metadata_preserved(out, scene)
