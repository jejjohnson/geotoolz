"""Tests for `geotoolz.normalize`."""

from __future__ import annotations

import json

import numpy as np
import pytest
from _helpers import fill_pixel_mask, toy_geotensor
from georeader.geotensor import GeoTensor

import geotoolz as gz
from geotoolz._src.valid import mask_invalid_to_nan
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
    histogram_match,
)


def _toy_geotensor(values: np.ndarray) -> GeoTensor:
    # Shared factory, but with NaN fill — the NaN-awareness tests below
    # rely on NaN marking invalid pixels.
    return toy_geotensor(values, fill_value_default=np.nan)


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
    # A pixel with a NaN in any band is excluded from every band's stats.
    masked = mask_invalid_to_nan(scene)
    np.testing.assert_allclose(op.stats["mean"], np.nanmean(masked, axis=(-2, -1)))
    assert op.stats["percentiles"][0][0] == pytest.approx(
        np.nanpercentile(masked[0], 2.0)
    )


def test_standard_scaler_fit_inverse_state_roundtrip(
    scene: GeoTensor,
) -> None:
    scaler = StandardScaler(fit_on_call=True)
    scaled = scaler(scene)
    restored = scaler.inverse(scaled)

    assert_metadata_preserved(scaled, scene)
    # The pixel NaN in band 0 is invalid in every band (any-band rule).
    np.testing.assert_allclose(
        np.asarray(restored), mask_invalid_to_nan(scene), equal_nan=True
    )

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


def test_histogram_match_pools_frames_per_band_on_a_time_stack() -> None:
    """A (T, C, H, W) stack matches each band to its own reference band,
    with one CDF over every frame -- never a frame taken for a band.
    """
    rng = np.random.default_rng(0)
    reference = _toy_geotensor(
        np.stack([rng.uniform(lo, lo + 1.0, (8, 8)) for lo in (0.0, 10.0, 100.0)])
    )
    stack = _toy_geotensor(rng.uniform(0.0, 1.0, (2, 3, 8, 8)))
    stack.values[1] *= 2.0
    out = np.asarray(HistogramMatch(reference=reference)(stack))
    for band, lo in enumerate((0.0, 10.0, 100.0)):
        assert np.all((out[:, band] >= lo) & (out[:, band] <= lo + 1.0))
        # One CDF over both frames of the band.
        pooled = histogram_match(
            np.asarray(stack)[None, :, band], reference.values[band]
        )[0]
        np.testing.assert_array_equal(out[:, band], pooled)
    # The brighter second frame stays brighter after the pooled match.
    assert np.all(out[1].mean(axis=(-2, -1)) > out[0].mean(axis=(-2, -1)))


def test_histogram_match_forbidden_in_yaml() -> None:
    """HistogramMatch holds a live reference; must opt out of YAML."""
    assert HistogramMatch.forbid_in_yaml is True


def test_clahe_preserves_nan_mask_and_metadata() -> None:
    gt = _toy_geotensor(np.linspace(0.0, 1.0, 100).reshape(1, 10, 10))
    np.asarray(gt)[0, 0, 0] = np.nan

    out = CLAHE(window=(4, 4), clip_limit=0.03)(gt)

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
        CLAHE(window=(4, 4), clip_limit=0.03),
        CLAHE(window=None),
    ]
    for op in ops:
        json.dumps(op.get_config())  # must not raise


def test_clahe_config_normalises_tuple_kernel_to_list() -> None:
    """Tuple ``window`` must serialise as a list for config round-trips."""
    op = CLAHE(window=(8, 8))
    cfg = op.get_config()
    assert cfg["window"] == [8, 8]
    # Round-trip via list input rebuilds the operator equivalently.
    rebuilt = CLAHE(**cfg)
    assert rebuilt.window == (8, 8)
    assert rebuilt.get_config() == cfg


# ---------------------------------------------------------------------------
# Plain-ndarray carrier support
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [
        PerBandStats(percentiles=[2.0, 98.0]),
        StandardScaler(mean=[2.0, 5.0], std=[1.5, 2.5]),
        RobustScaler(median=[2.0, 5.0], iqr=[1.0, 2.0]),
        MinMaxScaler(vmin=[0.0, 0.0], vmax=[9.0, 9.0]),
        Normalize(mean=[2.0, 5.0], std=[1.5, 2.5]),
        HistogramStretch(out_range=(0.0, 255.0)),
        HistogramMatch(reference=np.linspace(10.0, 20.0, 16).reshape(4, 4)),
        LogScale(),
        AsinhScale(),
        PowerScale(),
        ZeroOne(),
        CLAHE(window=(4, 4), clip_limit=0.03),
    ],
    ids=lambda op: type(op).__name__,
)
def test_operators_accept_plain_ndarray(op: object) -> None:
    """Plain ndarray in -> plain ndarray out, values equal to the GeoTensor path."""
    arr = np.linspace(0.0, 9.0, 2 * 4 * 4).reshape(2, 4, 4)
    out = op(arr)  # type: ignore[operator]
    assert type(out) is np.ndarray
    gt_out = op(_toy_geotensor(arr))  # type: ignore[operator]
    np.testing.assert_allclose(out, np.asarray(gt_out), equal_nan=True)


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
        CLAHE(window=(4, 4), clip_limit=0.03),
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


# ---------------------------------------------------------------------------
# Nodata (fill pixels) handling
# ---------------------------------------------------------------------------


def _fill_scene() -> tuple[GeoTensor, np.ndarray, np.ndarray]:
    """(2, 4, 5) scene with -9999 fill pixels, its clean values and fill mask."""
    rng = np.random.default_rng(0)
    values = rng.uniform(0.5, 9.0, size=(2, 4, 5))
    gt = toy_geotensor(values, fill_value_default=-9999, with_fill_pixels=True)
    return gt, values, fill_pixel_mask(values.shape)


_REFERENCE = np.linspace(10.0, 20.0, 20).reshape(4, 5)


@pytest.mark.parametrize(
    "make_op",
    [
        lambda: StandardScaler(fit_on_call=True),
        lambda: RobustScaler(fit_on_call=True),
        lambda: MinMaxScaler(fit_on_call=True),
        lambda: Normalize(mean=[2.0, 5.0], std=[1.5, 2.5]),
        lambda: HistogramStretch(out_range=(0.0, 255.0)),
        lambda: HistogramMatch(reference=_REFERENCE),
        lambda: LogScale(),
        lambda: AsinhScale(),
        lambda: PowerScale(),
        lambda: ZeroOne(),
        lambda: ZeroOne(per_band=False),
        lambda: CLAHE(window=(2, 2), clip_limit=0.03),
    ],
    ids=lambda f: type(f()).__name__,
)
def test_fill_pixels_are_excluded(make_op) -> None:
    """Fill pixels never enter a fit and map to the output fill value.

    Valid pixels must equal the result on the same data with the fill
    pixels marked missing (NaN), i.e. statistics over valid pixels only.
    Normalized values live on a new scale, so the output fill is NaN
    rather than the input's -9999 (#146).
    """
    gt, values, fill = _fill_scene()
    result = make_op()(gt)
    out = np.asarray(result)

    assert np.isnan(result.fill_value_default)
    assert np.isnan(out[:, fill]).all()
    reference = values.copy()
    reference[:, fill] = np.nan
    expected = np.asarray(make_op()(reference))
    np.testing.assert_allclose(out[:, ~fill], expected[:, ~fill])


def test_fill_pixels_are_excluded_from_fitted_stats() -> None:
    gt, values, fill = _fill_scene()
    valid = values[:, ~fill]

    scaler = StandardScaler(fit_on_call=True)
    scaler(gt)
    np.testing.assert_allclose(scaler.mean_, valid.mean(axis=1))
    np.testing.assert_allclose(scaler.std_, valid.std(axis=1))

    stats = PerBandStats()
    assert stats(gt) is gt
    np.testing.assert_allclose(stats.stats["min"], valid.min(axis=1))

    minmax = MinMaxScaler(fit_on_call=True)
    minmax(gt)
    np.testing.assert_allclose(minmax.vmin_, valid.min(axis=1))

    # inverse() keeps fill pixels of the scaled carrier as nodata (NaN).
    restored = np.asarray(scaler.inverse(scaler(gt)))
    assert np.isnan(restored[:, fill]).all()
    np.testing.assert_allclose(restored[:, ~fill], valid)


def test_4d_time_stack() -> None:
    """Per-band stats are (C,) on a stack; CLAHE equalises each slice (#147)."""
    from _helpers import frames, time_stack

    from geotoolz.normalize._src.array import reshape_stat, stat_axes

    stack = time_stack((2, 3, 8, 8))
    values = np.asarray(stack)
    assert stat_axes(values) == (-4, -2, -1)
    scaler = StandardScaler(fit_on_call=True)
    out = scaler(stack)
    assert np.shape(scaler.mean_) == (3,)
    np.testing.assert_allclose(scaler.mean_, values.mean(axis=(0, 2, 3)))
    np.testing.assert_allclose(np.asarray(out).mean(axis=(0, 2, 3)), 0.0, atol=1e-12)

    clahe = CLAHE(window=(4, 4))
    equalised = clahe(stack)
    for t, frame in enumerate(frames(stack)):
        np.testing.assert_allclose(np.asarray(equalised)[t], np.asarray(clahe(frame)))

    with pytest.raises(ValueError, match="does not match the kept axes"):
        reshape_stat(np.zeros(2), values, stat_axes(values))
    with pytest.raises(ValueError, match="does not match the kept axes"):
        Normalize(mean=[0.0, 0.0], std=[1.0, 1.0])(stack)


# ----------------------------------------------------------------------------
# Fitted-operator contract (#143): fit / transform / inverse seams.
# ----------------------------------------------------------------------------
_SCALERS = [
    pytest.param(StandardScaler, ("mean_", "std_"), id="StandardScaler"),
    pytest.param(RobustScaler, ("median_", "iqr_"), id="RobustScaler"),
    pytest.param(MinMaxScaler, ("vmin_", "vmax_"), id="MinMaxScaler"),
]


@pytest.mark.parametrize(("cls", "fitted"), _SCALERS)
def test_scalers_satisfy_fittable_transformer(cls, fitted) -> None:
    from pipekit.protocols import FittableTransformer

    op = cls()
    assert isinstance(op, FittableTransformer)
    assert callable(op.inverse)
    assert all(getattr(op, name) is None for name in fitted)


@pytest.mark.parametrize(("cls", "fitted"), _SCALERS)
def test_scaler_fit_then_transform_matches_fit_on_call(
    cls, fitted, scene: GeoTensor
) -> None:
    op = cls()
    with pytest.raises(ValueError, match=r"fit\(\)"):
        op.transform(scene)
    assert op.fit(scene) is op
    assert all(getattr(op, name) is not None for name in fitted)
    np.testing.assert_allclose(
        np.asarray(op.transform(scene)),
        np.asarray(cls(fit_on_call=True)(scene)),
        equal_nan=True,
    )


@pytest.mark.parametrize(("cls", "fitted"), _SCALERS)
def test_scaler_fitted_state_excluded_from_config(
    cls, fitted, scene: GeoTensor
) -> None:
    """Statistics learned on call never reach ``get_config()`` / ``state``."""
    op = cls(fit_on_call=True)
    before = op.get_config()
    op(scene)
    assert all(getattr(op, name) is not None for name in fitted)
    assert op.get_config() == before
    # The config still names only constructor parameters, all unset.
    assert all(op.get_config()[name.rstrip("_")] is None for name in fitted)
    json.dumps(op.state, allow_nan=False)


@pytest.mark.parametrize(("cls", "fitted"), _SCALERS)
def test_scaler_inverse_round_trips(cls, fitted, scene: GeoTensor) -> None:
    op = cls().fit(scene)
    restored = op.inverse(op.transform(scene))
    np.testing.assert_allclose(
        np.asarray(restored), mask_invalid_to_nan(scene), equal_nan=True
    )


@pytest.mark.parametrize(("cls", "fitted"), _SCALERS)
def test_scaler_fit_on_call_fits_once_under_thread_map(cls, fitted) -> None:
    """Concurrent first calls under ``ThreadMap`` fit exactly once.

    Each scene has different statistics; a slow ``fit`` widens the race
    window. Every output must equal ``transform`` with the one published
    fit -- no call may see half-written or another call's statistics.
    """
    import threading
    import time

    from pipekit.parallel import ThreadMap

    rng = np.random.default_rng(0)
    scenes = [rng.normal(loc=i, scale=i + 1, size=(2, 6, 6)) for i in range(8)]
    op = cls(fit_on_call=True)
    calls = []
    fit = op.fit
    lock = threading.Lock()

    def slow_fit(x):
        with lock:
            calls.append(x)
        time.sleep(0.05)
        return fit(x)

    op.fit = slow_fit
    outputs = ThreadMap(op, n_workers=8)(scenes)

    assert len(calls) == 1
    for scene, out in zip(scenes, outputs, strict=True):
        np.testing.assert_allclose(out, op.transform(scene))


def test_fitted_scaler_pickles_and_deep_copies_with_its_fit_lock(
    scene: GeoTensor,
) -> None:
    """The per-instance fit lock must not block ``ProcessMap`` / ``deepcopy``."""
    import copy
    import pickle

    op = StandardScaler(fit_on_call=True)
    expected = np.asarray(op(scene))  # first call creates the fit lock
    for clone in (pickle.loads(pickle.dumps(op)), copy.deepcopy(op)):
        np.testing.assert_allclose(clone.mean_, op.mean_)
        np.testing.assert_allclose(np.asarray(clone(scene)), expected, equal_nan=True)
