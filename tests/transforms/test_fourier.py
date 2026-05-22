"""Tests for :mod:`xrtoolz.transforms._src.fourier`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrtoolz.transforms import (
    coherence,
    compensated_spectrum,
    cross_spectrum,
    drop_negative_frequencies,
    enstrophy_spectral_flux,
    fit_spectral_slope,
    integral_scale,
    isotropic_power_spectrum,
    ke_spectral_flux,
    power_spectrum,
    rotary_spectrum,
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


@pytest.fixture
def taylor_green_vortex_uv() -> tuple[xr.DataArray, xr.DataArray]:
    n = 32
    mode = 2
    x = np.arange(n)
    y = np.arange(n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    phase = 2.0 * np.pi * mode / n
    u = np.sin(phase * xx) * np.cos(phase * yy)
    v = -np.cos(phase * xx) * np.sin(phase * yy)
    coords = {"x": x, "y": y}
    return (
        xr.DataArray(u, dims=("x", "y"), coords=coords, name="u"),
        xr.DataArray(v, dims=("x", "y"), coords=coords, name="v"),
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


def _rotary_fixture(*, rotation_direction: float = 1.0) -> xr.Dataset:
    """Build one Fourier mode; ``1`` is CCW and ``-1`` is CW."""
    n = 64
    k = 3
    x = np.arange(n, dtype=float)
    phase = 2.0 * np.pi * k * x / n
    return xr.Dataset(
        {
            "u": ("x", np.cos(phase)),
            "v": ("x", rotation_direction * np.sin(phase)),
        },
        coords={"x": x},
    )


def test_rotary_spectrum_ccw_signal_peaks_positive_wavenumber():
    ds = _rotary_fixture(rotation_direction=1.0)
    out = rotary_spectrum(ds["u"], ds["v"], dim="x")
    k = 3.0 / 64.0
    assert float(out["psd_ccw"].idxmax("wavenumber")) == pytest.approx(k)
    assert float(out["psd_cw"].sel(wavenumber=k)) == pytest.approx(0.0, abs=1e-12)
    assert float(out["polarization"].sel(wavenumber=k)) == pytest.approx(-1.0)


def test_rotary_spectrum_cw_signal_has_positive_polarization():
    ds = _rotary_fixture(rotation_direction=-1.0)
    out = rotary_spectrum(ds["u"], ds["v"], dim="x")
    k = 3.0 / 64.0
    assert float(out["psd_cw"].idxmax("wavenumber")) == pytest.approx(k)
    assert float(out["polarization"].sel(wavenumber=k)) == pytest.approx(1.0)


def test_rotary_spectrum_real_only_signal_is_unpolarized():
    ds = _rotary_fixture(rotation_direction=1.0)
    ds["v"] = xr.zeros_like(ds["v"])
    out = rotary_spectrum(ds["u"], ds["v"], dim="x")
    k = 3.0 / 64.0
    assert float(out["polarization"].sel(wavenumber=k)) == pytest.approx(0.0, abs=1e-12)


def test_rotary_spectrum_parseval_matches_velocity_variance():
    ds = _rotary_fixture(rotation_direction=1.0)
    out = rotary_spectrum(ds["u"], ds["v"], dim="x")
    dk = float(out["wavenumber"].diff("wavenumber").median())
    spectral_variance = float((out["psd_cw"] + out["psd_ccw"]).sum() * dk)
    component_variance = float(ds["u"].var("x") + ds["v"].var("x"))
    assert spectral_variance == pytest.approx(component_variance)


def test_rotary_spectrum_avg_dims_reduce_outputs():
    ds = _rotary_fixture(rotation_direction=1.0).expand_dims(lat=[0.0, 1.0])
    ds["u"] = ds["u"] * xr.DataArray([1.0, 2.0], dims="lat", coords={"lat": ds["lat"]})
    ds["v"] = ds["v"] * xr.DataArray([1.0, 2.0], dims="lat", coords={"lat": ds["lat"]})
    out = rotary_spectrum(ds["u"], ds["v"], dim="x", avg_dims=("lat",))
    assert out["psd_ccw"].dims == ("wavenumber",)
    assert out["psd_cw"].dims == ("wavenumber",)
    assert out["polarization"].dims == ("wavenumber",)


def test_rotary_spectrum_handles_dim_without_explicit_coord():
    """xarray allows dims with no coordinate variable; rotary_spectrum
    should fall back to unit spacing rather than KeyError on ds[dim]."""
    n = 32
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "u": ("x", rng.standard_normal(n)),
            "v": ("x", rng.standard_normal(n)),
        },
    )
    assert "x" not in ds.coords
    out = rotary_spectrum(ds["u"], ds["v"], dim="x")
    assert out["psd_ccw"].dims == ("wavenumber",)


def test_rotary_spectrum_preserves_nyquist_bin_for_even_length_inputs():
    """Even-length FFTs put Nyquist on the negative-frequency side
    only; the outer-join on wavenumber must keep that bin so total
    rotary power is variance-consistent."""
    ds = _rotary_fixture(rotation_direction=1.0)
    out = rotary_spectrum(ds["u"], ds["v"], dim="x")
    nyquist = 0.5 / 1.0  # spacing = 1
    assert nyquist in out["wavenumber"].values


def test_rotary_spectrum_operator_matches_primitive():
    """The new ``RotarySpectrum`` operator must match the primitive on
    the same data, with the operator doing the Dataset selection."""
    from xrtoolz.transforms.operators import RotarySpectrum

    ds = _rotary_fixture(rotation_direction=1.0)
    op_out = RotarySpectrum("u", "v", "x")(ds)
    fn_out = rotary_spectrum(ds["u"], ds["v"], dim="x")
    np.testing.assert_allclose(op_out["psd_ccw"].values, fn_out["psd_ccw"].values)
    np.testing.assert_allclose(op_out["psd_cw"].values, fn_out["psd_cw"].values)


def test_ke_spectral_flux_conserves_transfer(taylor_green_vortex_uv):
    u, v = taylor_green_vortex_uv
    out = ke_spectral_flux(
        u, v, dim=("x", "y"), window=None, detrend=None, return_2d=True
    )
    assert abs(float(out["transfer"].sum())) < 1e-16
    assert abs(float(out["flux"].isel(freq_r=0))) < 1e-16
    assert abs(float(out["flux"].isel(freq_r=-1))) < 1e-16


def test_ke_spectral_flux_returns_2d(taylor_green_vortex_uv):
    u, v = taylor_green_vortex_uv
    out = ke_spectral_flux(
        u, v, dim=("x", "y"), window=None, detrend=None, return_2d=True
    )
    assert set(out.data_vars) == {"transfer", "flux", "transfer_2d"}
    assert out["transfer_2d"].sizes["freq_x"] == u.sizes["x"]
    assert out["transfer_2d"].sizes["freq_y"] == u.sizes["y"]


def test_ke_spectral_flux_avg_dims_matches_manual_average(taylor_green_vortex_uv):
    u, v = taylor_green_vortex_uv
    u_time = xr.concat([u, 2.0 * u], dim="time").assign_coords(time=[0, 1])
    v_time = xr.concat([v, 2.0 * v], dim="time").assign_coords(time=[0, 1])
    full = ke_spectral_flux(u_time, v_time, dim=("x", "y"), window=None, detrend=None)
    averaged = ke_spectral_flux(
        u_time,
        v_time,
        dim=("x", "y"),
        window=None,
        detrend=None,
        avg_dims=("time",),
    )
    xr.testing.assert_allclose(averaged["transfer"], full["transfer"].mean("time"))


def test_enstrophy_spectral_flux_budget_closes(taylor_green_vortex_uv):
    u, v = taylor_green_vortex_uv
    out = enstrophy_spectral_flux(u, v, dim=("x", "y"), window=None, detrend=None)
    assert set(out.data_vars) == {"transfer", "flux"}
    assert abs(float(out["transfer"].sum())) < 1e-16
    assert abs(float(out["flux"].isel(freq_r=0))) < 1e-16


def test_ke_spectral_flux_default_preprocessing_runs_and_closes_budget(
    taylor_green_vortex_uv,
):
    """The defaults are window='tukey' / detrend='linear'. With the
    advection product now consistently using the windowed/detrended
    fields (not raw u/v), the transfer must still integrate to zero
    even when preprocessing is enabled."""
    u, v = taylor_green_vortex_uv
    out = ke_spectral_flux(u, v, dim=("x", "y"))
    assert np.isfinite(out["transfer"].values).all()
    assert np.isfinite(out["flux"].values).all()
    assert abs(float(out["transfer"].sum())) < 1e-10
    # Flux endpoints bracket the cumulative integral; with a closed
    # budget, both ends must be (numerically) zero.
    assert abs(float(out["flux"].isel(freq_r=0))) < 1e-10
    assert abs(float(out["flux"].isel(freq_r=-1))) < 1e-10


def test_enstrophy_spectral_flux_default_preprocessing_runs_and_closes_budget(
    taylor_green_vortex_uv,
):
    u, v = taylor_green_vortex_uv
    out = enstrophy_spectral_flux(u, v, dim=("x", "y"))
    assert np.isfinite(out["transfer"].values).all()
    assert abs(float(out["transfer"].sum())) < 1e-10
    assert abs(float(out["flux"].isel(freq_r=0))) < 1e-10


def test_integral_scale_matches_gaussian_and_spike():
    width = 2.0
    k = np.linspace(0.0, 20.0, 20_001)
    psd = xr.DataArray(np.exp(-((k / width) ** 2)), dims="freq_r", coords={"freq_r": k})
    np.testing.assert_allclose(
        integral_scale(psd, moment=1), np.sqrt(np.pi) / width, rtol=5e-3
    )
    np.testing.assert_allclose(
        integral_scale(psd, moment=2), np.sqrt(2.0) / width, rtol=5e-3
    )

    spike_k = np.array([1.0, 2.0, 4.0])
    spike = xr.DataArray([0.0, 5.0, 0.0], dims="freq_r", coords={"freq_r": spike_k})
    np.testing.assert_allclose(integral_scale(spike, moment=1), 1.0 / 2.0)


def test_fit_spectral_slope_honours_window():
    k = np.linspace(1.0, 100.0, 200)
    psd = xr.DataArray(k ** (-5.0 / 3.0), dims="freq_r", coords={"freq_r": k})
    slope, intercept = fit_spectral_slope(psd, k_min=10.0, k_max=50.0)
    np.testing.assert_allclose(slope, -5.0 / 3.0, rtol=1e-2)
    np.testing.assert_allclose(intercept, 0.0, atol=1e-12)


def test_compensated_spectrum_is_flat_for_power_law():
    k = np.linspace(1.0, 10.0, 20)
    psd = xr.DataArray(k**-3.0, dims="freq_r", coords={"freq_r": k}, name="energy")
    compensated = compensated_spectrum(psd, exponent=3.0)
    assert compensated.name == "energy_compensated"
    np.testing.assert_allclose(compensated, 1.0)


def test_fit_spectral_slope_raises_when_window_has_too_few_samples():
    k = np.linspace(1.0, 100.0, 200)
    psd = xr.DataArray(k ** (-5.0 / 3.0), dims="freq_r", coords={"freq_r": k})
    # An empty/sparse window leaves <2 samples, so the linear fit is
    # underdetermined and the function must surface that explicitly.
    with pytest.raises(ValueError, match="At least two"):
        fit_spectral_slope(psd, k_min=200.0, k_max=300.0)


def test_integral_scale_rejects_unsupported_moment():
    k = np.linspace(0.0, 10.0, 11)
    psd = xr.DataArray(np.ones_like(k), dims="freq_r", coords={"freq_r": k})
    with pytest.raises(ValueError, match="moment must be 1"):
        integral_scale(psd, moment=3)
