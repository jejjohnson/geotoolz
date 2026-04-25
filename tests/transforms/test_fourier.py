"""Tests for :mod:`xr_toolz.transforms._src.fourier`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xr_toolz.transforms import (
    coherence,
    cross_spectrum,
    drop_negative_frequencies,
    isotropic_power_spectrum,
    power_spectrum,
    stft,
)


@pytest.fixture
def da_grid_daily() -> xr.DataArray:
    time = pd.date_range("2020-01-01", "2021-12-31", freq="1D")
    lat = np.linspace(-40.0, 40.0, 9)
    lon = np.linspace(-60.0, 60.0, 13)
    rng = np.random.default_rng(0)
    tt = np.arange(len(time))
    base = np.sin(2 * np.pi * tt / 365.25)[:, None, None]
    data = base + 0.1 * rng.standard_normal((len(time), len(lat), len(lon)))
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="ssh",
    )


def test_power_spectrum_1d_along_time(da_grid_daily):
    da_1d = da_grid_daily.isel(lat=4, lon=6)
    out = power_spectrum(da_1d, dim="time")
    assert out.name == "ssh_psd"
    assert any(d.startswith("freq_") for d in out.dims)


def test_power_spectrum_multidim_returns_freq_axes(da_grid_daily):
    da = da_grid_daily.isel(time=slice(0, 64), lat=4)
    out = power_spectrum(da, dim=("time", "lon"))
    assert out.name == "ssh_psd"
    assert any(d.startswith("freq_") for d in out.dims)


def test_isotropic_power_spectrum_returns_freq_r(da_grid_daily):
    da = da_grid_daily.isel(time=4)
    out = isotropic_power_spectrum(da, dim=("lat", "lon"))
    assert out.name == "ssh_iso_psd"
    assert "freq_r" in out.dims


def test_isotropic_requires_two_dims(da_grid_daily):
    with pytest.raises(ValueError, match="exactly 2"):
        power_spectrum(da_grid_daily, dim="time", isotropic=True)


def test_cross_spectrum_naming(da_grid_daily):
    a = da_grid_daily.isel(lat=4, lon=6)
    b = (a * 2.0).rename("ssh2")
    out = cross_spectrum(a, b, dim="time")
    assert out.name == "ssh_ssh2_csd"


def test_coherence_self_is_unity(da_grid_daily):
    """Coherence of a signal with itself is identically 1 wherever the
    auto-spectrum is non-zero."""
    a = da_grid_daily.isel(lat=4, lon=6)
    out = coherence(a, a, dim="time")
    assert out.name == "ssh_ssh_coh"
    finite = np.isfinite(out.values)
    np.testing.assert_allclose(out.values[finite], 1.0, atol=1e-10)


def test_stft_creates_segment_axis(da_grid_daily):
    da_1d = da_grid_daily.isel(lat=4, lon=6)
    out = stft(da_1d, dim="time", window_size=64, hop=32)
    assert "segment" in out.dims
    assert out.name == "ssh_stft"
    # Frequency axis is named freq_<dim> by xrft.
    assert any(d.startswith("freq_") for d in out.dims)


def test_stft_window_too_large_raises(da_grid_daily):
    da_1d = da_grid_daily.isel(time=slice(0, 32), lat=4, lon=6)
    with pytest.raises(ValueError, match="exceeds"):
        stft(da_1d, dim="time", window_size=128)


def test_drop_negative_frequencies_renames_old_conditional_average(da_grid_daily):
    da = da_grid_daily.isel(time=slice(0, 64), lat=4)
    psd = power_spectrum(da, dim=("time", "lon"))
    # Average out the time-frequency axis, restricting to positive lon-freq.
    reduced = drop_negative_frequencies(psd, dims=["freq_time"])
    # Output should drop freq_time and retain only freq_lon > 0.
    assert "freq_time" not in reduced.dims
    assert (reduced["freq_lon"] > 0).all()
