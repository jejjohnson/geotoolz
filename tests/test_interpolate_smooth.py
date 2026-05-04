"""Tests for ``xr_toolz.interpolate`` smoothers (F3.3, D12)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.interpolate import (
    array as ia,
    gaussian_smooth,
    lowpass_filter,
    moving_average,
)
from xr_toolz.interpolate.operators import (
    GaussianSmooth,
    LowpassFilter,
    MovingAverage,
)


# ---------------------------------------------------------------------------
# Tier A — moving_average
# ---------------------------------------------------------------------------


def test_array_moving_average_constant_input_unchanged():
    x = np.full(20, 3.5)
    out = ia.moving_average(x, axis=-1, window=5, min_periods=1)
    np.testing.assert_allclose(out, 3.5)


def test_array_moving_average_default_min_periods_nans_edges():
    """With default ``min_periods=window`` the edges lack a full window."""
    x = np.full(20, 3.5)
    out = ia.moving_average(x, axis=-1, window=5)  # default min_periods=window
    # Centered window of 5 leaves 2 NaN at each end.
    assert np.all(np.isnan(out[:2]))
    assert np.all(np.isnan(out[-2:]))
    np.testing.assert_allclose(out[2:-2], 3.5)


def test_array_moving_average_centered_window_matches_manual():
    x = np.arange(10, dtype=float)
    out = ia.moving_average(x, axis=-1, window=3, center=True)
    # Interior: simple mean of (x[i-1], x[i], x[i+1]).
    np.testing.assert_allclose(out[1:-1], np.arange(1, 9, dtype=float))


def test_array_moving_average_min_periods_propagates_nan():
    x = np.arange(10, dtype=float)
    out = ia.moving_average(x, axis=-1, window=4, center=False, min_periods=4)
    # First three outputs lack a full window → NaN.
    assert np.all(np.isnan(out[:3]))
    assert not np.any(np.isnan(out[3:]))


def test_array_moving_average_invalid_window_raises():
    with pytest.raises(ValueError):
        ia.moving_average(np.zeros(5), axis=-1, window=0)


# ---------------------------------------------------------------------------
# Tier A — gaussian_smooth
# ---------------------------------------------------------------------------


def test_array_gaussian_preserves_constant():
    x = np.full(50, 2.0)
    np.testing.assert_allclose(ia.gaussian_smooth(x, axis=-1, sigma=2.0), 2.0)


def test_array_gaussian_attenuates_high_freq():
    n = 256
    t = np.arange(n)
    fast = np.sin(2 * np.pi * t / 4)  # period-4 sinusoid
    smooth = ia.gaussian_smooth(fast, axis=-1, sigma=4.0)
    # Sigma=4 over period-4 should aggressively attenuate.
    assert smooth.std() < 0.1 * fast.std()


def test_array_gaussian_invalid_sigma_raises():
    with pytest.raises(ValueError):
        ia.gaussian_smooth(np.zeros(5), axis=-1, sigma=0.0)


# ---------------------------------------------------------------------------
# Tier A — lowpass_filter
# ---------------------------------------------------------------------------


def test_array_lowpass_attenuates_above_cutoff():
    """Pass-band sinusoid survives; stop-band sinusoid is attenuated."""
    n = 1024
    t = np.arange(n)
    pass_band = np.sin(2 * np.pi * t / 64)  # period 64 → freq 1/64 ≈ 0.016 (< cutoff)
    stop_band = np.sin(2 * np.pi * t / 4)  # period 4 → freq 0.25 (> cutoff)

    cutoff = 0.05  # fraction of Nyquist (Nyquist=1)
    pass_out = ia.lowpass_filter(pass_band, axis=-1, cutoff=cutoff, order=4)
    stop_out = ia.lowpass_filter(stop_band, axis=-1, cutoff=cutoff, order=4)

    # Trim edge transients before measuring amplitude.
    assert pass_out[100:-100].std() > 0.9 * pass_band.std()
    assert stop_out[100:-100].std() < 0.05 * stop_band.std()


# ---------------------------------------------------------------------------
# Tier B — Dataset wrappers
# ---------------------------------------------------------------------------


@pytest.fixture
def ds_signal():
    n = 128
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 8) + 0.5 * np.sin(2 * np.pi * t / 2)
    return xr.Dataset({"x": (("time",), x)}, coords={"time": t})


def test_tier_b_moving_average_preserves_shape(ds_signal):
    out = moving_average(ds_signal, dim="time", window=5)
    assert out["x"].shape == ds_signal["x"].shape
    assert out["x"].dims == ("time",)


def test_tier_b_gaussian_smooth_matches_tier_a(ds_signal):
    out = gaussian_smooth(ds_signal, dim="time", sigma=3.0)
    expected = ia.gaussian_smooth(ds_signal["x"].values, axis=-1, sigma=3.0)
    np.testing.assert_allclose(out["x"].values, expected)


def test_tier_b_lowpass_filter_matches_tier_a(ds_signal):
    out = lowpass_filter(ds_signal, dim="time", cutoff=0.1, order=4)
    expected = ia.lowpass_filter(ds_signal["x"].values, axis=-1, cutoff=0.1, order=4)
    np.testing.assert_allclose(out["x"].values, expected)


def test_tier_b_passes_through_non_dim_variables():
    ds = xr.Dataset(
        {
            "x": (("time",), np.arange(10, dtype=float)),
            "static": ((), 7.0),
        },
        coords={"time": np.arange(10)},
    )
    out = moving_average(ds, dim="time", window=3)
    assert float(out["static"]) == 7.0


def test_tier_b_unknown_dim_raises(ds_signal):
    with pytest.raises(ValueError):
        moving_average(ds_signal, dim="bogus", window=3)


# ---------------------------------------------------------------------------
# Tier C — Operator wrappers
# ---------------------------------------------------------------------------


def test_tier_c_moving_average_matches_tier_b(ds_signal):
    op = MovingAverage("time", window=5)
    np.testing.assert_allclose(
        op(ds_signal)["x"].values,
        moving_average(ds_signal, dim="time", window=5)["x"].values,
    )


def test_tier_c_gaussian_smooth_matches_tier_b(ds_signal):
    op = GaussianSmooth("time", sigma=2.5)
    np.testing.assert_allclose(
        op(ds_signal)["x"].values,
        gaussian_smooth(ds_signal, dim="time", sigma=2.5)["x"].values,
    )


def test_tier_c_lowpass_filter_matches_tier_b(ds_signal):
    op = LowpassFilter("time", cutoff=0.1)
    np.testing.assert_allclose(
        op(ds_signal)["x"].values,
        lowpass_filter(ds_signal, dim="time", cutoff=0.1)["x"].values,
    )


def test_tier_c_get_config_is_serializable():
    cfg = LowpassFilter("time", cutoff=0.2, order=6, btype="high").get_config()
    assert cfg == {
        "dim": "time",
        "cutoff": 0.2,
        "order": 6,
        "btype": "high",
    }


# ---------------------------------------------------------------------------
# Validation + dtype preservation (review feedback)
# ---------------------------------------------------------------------------


def test_array_smoothers_preserve_complex_dtype():
    """Complex inputs must not be silently coerced to float."""
    n = 64
    t = np.arange(n)
    z = np.exp(1j * 2 * np.pi * t / n)
    assert np.iscomplexobj(ia.gaussian_smooth(z, axis=-1, sigma=1.0))
    assert np.iscomplexobj(ia.lowpass_filter(z, axis=-1, cutoff=0.1))


def test_array_lowpass_band_requires_pair_cutoff():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256)
    out = ia.lowpass_filter(x, axis=-1, cutoff=(0.05, 0.4), btype="bandpass")
    assert out.shape == x.shape
    with pytest.raises(ValueError):
        ia.lowpass_filter(x, axis=-1, cutoff=0.1, btype="bandpass")
    with pytest.raises(ValueError):
        ia.lowpass_filter(x, axis=-1, cutoff=(0.4, 0.05), btype="bandpass")


def test_array_lowpass_unknown_btype_raises():
    with pytest.raises(ValueError):
        ia.lowpass_filter(np.zeros(16), axis=-1, cutoff=0.2, btype="bogus")


def test_array_lowpass_invalid_cutoff_raises():
    with pytest.raises(ValueError):
        ia.lowpass_filter(np.zeros(16), axis=-1, cutoff=1.5)
    with pytest.raises(ValueError):
        ia.lowpass_filter(np.zeros(16), axis=-1, cutoff=0.0)


def test_array_moving_average_rejects_non_integer_window():
    with pytest.raises(TypeError):
        ia.moving_average(np.zeros(8), axis=-1, window=1.9)  # type: ignore[arg-type]


def test_tier_c_moving_average_rejects_non_integer_window():
    with pytest.raises(TypeError):
        MovingAverage("time", window=1.9)  # type: ignore[arg-type]


def test_tier_c_lowpass_band_filter_passes_band():
    n = 1024
    t = np.arange(n)
    # Period 16 → 0.0625 cycles/sample → 0.125 in normalized (Nyquist=1) units.
    band_signal = np.sin(2 * np.pi * t / 16)
    ds = xr.Dataset({"x": (("time",), band_signal)}, coords={"time": t})
    op = LowpassFilter("time", cutoff=(0.08, 0.20), order=4, btype="bandpass")
    out = op(ds)
    assert out["x"].values[100:-100].std() > 0.9 * band_signal.std()
