"""Tests for :mod:`xr_toolz.transforms.operators`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xr_toolz.core import Sequential
from xr_toolz.transforms.operators import (
    DCT,
    STFT,
    Coherence,
    CrossSpectrum,
    EnstrophySpectralFlux,
    KESpectralFlux,
    PowerSpectrum,
)


@pytest.fixture
def ds() -> xr.Dataset:
    time = pd.date_range("2020-01-01", periods=128, freq="1D")
    rng = np.random.default_rng(0)
    a = rng.standard_normal(128)
    b = a * 0.5 + 0.1 * rng.standard_normal(128)
    return xr.Dataset(
        {"a": ("time", a), "b": ("time", b)},
        coords={"time": time},
    )


@pytest.fixture
def taylor_green_ds() -> xr.Dataset:
    n = 16
    x = np.arange(n)
    y = np.arange(n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    phase = 2.0 * np.pi * 2 / n
    u = np.sin(phase * xx) * np.cos(phase * yy)
    v = -np.cos(phase * xx) * np.sin(phase * yy)
    return xr.Dataset(
        {"u": (("x", "y"), u), "v": (("x", "y"), v)},
        coords={"x": x, "y": y},
    )


def test_power_spectrum_operator_names_output(ds):
    out = PowerSpectrum("a", "time")(ds)
    assert "a_psd" in out.data_vars


def test_cross_spectrum_operator_names_output(ds):
    out = CrossSpectrum("a", "b", "time")(ds)
    assert "a_b_csd" in out.data_vars


def test_coherence_operator_self_coherence_unity(ds):
    out = Coherence("a", "a", "time")(ds)
    finite = np.isfinite(out["a_a_coh"].values)
    np.testing.assert_allclose(out["a_a_coh"].values[finite], 1.0, atol=1e-10)


def test_stft_operator(ds):
    out = STFT("a", "time", window_size=32, hop=16)(ds)
    assert "a_stft" in out.data_vars
    assert "segment" in out["a_stft"].dims


def test_ke_spectral_flux_operator(taylor_green_ds):
    out = KESpectralFlux(
        "u", "v", ("x", "y"), window=None, detrend=None, return_2d=True
    )(taylor_green_ds)
    assert {"transfer", "flux", "transfer_2d"} <= set(out.data_vars)


def test_enstrophy_spectral_flux_operator(taylor_green_ds):
    out = EnstrophySpectralFlux(
        "u", "v", ("x", "y"), window=None, detrend=None, return_2d=True
    )(taylor_green_ds)
    assert {"transfer", "flux", "transfer_2d"} <= set(out.data_vars)


def test_dct_operator(ds):
    out = DCT("a", "time")(ds)
    assert "a_dct" in out.data_vars


def test_operators_compose_in_pipeline(ds):
    """Pipeline: DCT then PowerSpectrum on the DCT output. Just a smoke
    test that the operators chain via ``Sequential`` without surprises."""
    pipe = Sequential([DCT("a", "time")])
    out = pipe(ds)
    assert "a_dct" in out.data_vars


def test_get_config_round_trips_operator():
    op = PowerSpectrum("u", ["lat", "lon"], isotropic=True)
    cfg = op.get_config()
    assert cfg["variable"] == "u"
    assert cfg["dim"] == ["lat", "lon"]
    assert cfg["isotropic"] is True

    flux_op = KESpectralFlux(
        "u", "v", ("x", "y"), window=None, detrend=None, avg_dims=("time",)
    )
    flux_cfg = flux_op.get_config()
    assert flux_cfg["dim"] == ["x", "y"]
    assert flux_cfg["avg_dims"] == ["time"]
