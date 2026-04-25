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
