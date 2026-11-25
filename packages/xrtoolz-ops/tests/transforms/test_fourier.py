"""Tests for :mod:`xrops.transforms._src.fourier`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrops.transforms import (
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
    """Coherence of a signal with itself is identically 1."""
    a = da_grid_daily.isel(lat=4, lon=6)
    out = coherence(a, a, dim="time", segment=64)
    assert out.name == "ssh_ssh_coh"
    np.testing.assert_allclose(out, 1.0, atol=1e-10)


def test_coherence_without_an_average_raises(da_grid_daily):
    a = da_grid_daily.isel(lat=4, lon=6)
    with pytest.raises(ValueError, match="1 without an average"):
        coherence(a, a, dim="time")


def test_coherence_of_independent_noise_is_one_over_the_realisations():
    """It is 1 for any two signals when nothing is averaged (#772)."""
    rng = np.random.default_rng(0)
    a, b = (
        xr.DataArray(
            rng.standard_normal((40, 4, 256)),
            dims=("member", "x", "t"),
            coords={"t": np.arange(256.0)},
            name=name,
        )
        for name in "ab"
    )
    over = coherence(a, b, dim="t", average="member")
    assert over.dims == ("x", "freq_t")
    assert float(over.mean()) == pytest.approx(1.0 / 40, rel=0.2)
    both = coherence(a, b, dim="t", average=("member", "x"), detrend=None)
    assert both.dims == ("freq_t",)
    assert float(both.mean()) == pytest.approx(1.0 / 160, rel=0.2)


def test_coherence_takes_the_averaged_dims_wherever_they_are():
    rng = np.random.default_rng(0)
    a = xr.DataArray(
        rng.standard_normal((5, 3, 64)),
        dims=("m", "x", "t"),
        coords={"t": np.arange(64.0)},
        name="a",
    )
    b = (a + rng.standard_normal((5, 3, 64))).rename("b")
    expected = coherence(a, b, dim="t", average="m")
    turned = coherence(a.transpose("t", "x", "m"), b, dim="t", average="m")
    xr.testing.assert_allclose(turned.transpose(*expected.dims), expected)


def test_stft_creates_segment_axis(da_grid_daily):
    da_1d = da_grid_daily.isel(lat=4, lon=6)
    out = stft(da_1d, dim="time", window_size=64, hop=32)
    assert "segment" in out.dims
    assert out.name == "ssh_stft"
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
    from xrops.transforms.operators import RotarySpectrum

    ds = _rotary_fixture(rotation_direction=1.0)
    op_out = RotarySpectrum("u", "v", "x")(ds)
    fn_out = rotary_spectrum(ds["u"], ds["v"], dim="x")
    np.testing.assert_allclose(op_out["psd_ccw"].values, fn_out["psd_ccw"].values)
    np.testing.assert_allclose(op_out["psd_cw"].values, fn_out["psd_cw"].values)


def test_ke_spectral_flux_conserves_transfer(taylor_green_vortex_uv):
    u, v = taylor_green_vortex_uv
    out = ke_spectral_flux(u, v, dim=("x", "y"))
    assert set(out.data_vars) == {"transfer", "flux"}
    assert abs(float(out["transfer"].sum())) < 1e-16
    assert abs(float(out["flux"].isel(freq_r=0))) < 1e-16
    assert abs(float(out["flux"].isel(freq_r=-1))) < 1e-16


def _turbulence(n: int = 32) -> xr.Dataset:
    """A non-divergent flow of random modes, odd kx + ky among them."""
    rng = np.random.default_rng(1)
    x = 1e3 * np.arange(n)
    k = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(k, k)
    keep = np.hypot(kx, ky) < 0.25
    psi_hat = np.fft.fft2(rng.standard_normal((n, n))) * keep
    derivative = [np.fft.ifft2(2j * np.pi * q * psi_hat).real for q in (kx, ky)]
    return xr.Dataset(
        {"u": (("y", "x"), -derivative[1]), "v": (("y", "x"), derivative[0])},
        coords={"x": x, "y": x},
    )


@pytest.mark.parametrize("flux", [ke_spectral_flux, enstrophy_spectral_flux])
def test_the_transfer_of_a_non_divergent_flow_sums_to_zero(flux):
    """Every mode with an odd kx + ky had the wrong sign (#780)."""
    ds = _turbulence()
    transfer = flux(ds["u"], ds["v"], dim=("x", "y"))["transfer"]
    assert abs(float(transfer.sum())) < 1e-12 * float(abs(transfer).sum())


@pytest.mark.parametrize("flux", [ke_spectral_flux, enstrophy_spectral_flux])
def test_a_flux_does_not_depend_on_how_the_rows_are_stored(flux):
    ds = _turbulence()
    expected = flux(ds["u"], ds["v"], dim=("x", "y"))
    flipped = ds.isel(y=slice(None, None, -1)).transpose("x", "y")
    xr.testing.assert_allclose(
        flux(flipped["u"], flipped["v"], dim=("x", "y")), expected
    )


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

    spike_k = np.array([1.0, 2.0, 3.0])
    spike = xr.DataArray([0.0, 5.0, 0.0], dims="freq_r", coords={"freq_r": spike_k})
    np.testing.assert_allclose(integral_scale(spike, moment=1), 1.0 / 2.0)
    with pytest.raises(ValueError, match="evenly spaced"):
        integral_scale(spike.assign_coords(freq_r=[1.0, 2.0, 4.0]))


def test_fit_spectral_slope_honours_window():
    k = np.linspace(1.0, 100.0, 200)
    psd = xr.DataArray(k ** (-5.0 / 3.0), dims="freq_r", coords={"freq_r": k})
    slope = fit_spectral_slope(psd, k_min=10.0, k_max=50.0)
    np.testing.assert_allclose(slope, -5.0 / 3.0, rtol=1e-12)
    many = fit_spectral_slope(
        psd.expand_dims(time=3) * xr.DataArray([1.0, 2.0, 3.0], dims="time"),
        k_min=10.0,
        k_max=50.0,
    )
    assert many.dims == ("time",)
    np.testing.assert_allclose(many, -5.0 / 3.0, rtol=1e-12)


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


# ---------- the sum of a spectrum is the variance (#797) --------------------------


def test_a_spectrum_is_one_sided_and_sums_to_the_variance(da_grid_daily):
    da = da_grid_daily
    out = power_spectrum(da, dim="time", window=None, detrend="constant")
    assert out.dims == ("freq_time", "lat", "lon")
    np.testing.assert_allclose(
        out["freq_time"], np.fft.rfftfreq(da.sizes["time"], 86400.0)
    )
    step = float(out["freq_time"][1])
    np.testing.assert_allclose(out.sum("freq_time") * step, da.var("time"))


def test_a_two_dim_spectrum_is_two_sided_and_sums_to_the_variance(da_grid_daily):
    da = da_grid_daily.isel(time=slice(0, 64))
    out = power_spectrum(da, dim=("time", "lon"), window=None, detrend="constant")
    assert out.dims == ("freq_time", "lat", "freq_lon")
    steps = [float(out[d].diff(d)[0]) for d in ("freq_time", "freq_lon")]
    total = out.sum(("freq_time", "freq_lon")) * steps[0] * steps[1]
    np.testing.assert_allclose(total, da.var(("time", "lon")))


def test_an_isotropic_spectrum_sums_to_the_variance(da_grid_daily):
    da = da_grid_daily.isel(time=slice(0, 3))
    out = isotropic_power_spectrum(
        da, dim=("lat", "lon"), window=None, detrend="constant"
    )
    assert out.dims == ("time", "freq_r")
    step = float(out["freq_r"][1])
    np.testing.assert_allclose(out.sum("freq_r") * step, da.var(("lat", "lon")))


def test_the_dims_of_a_spectrum_may_be_given_either_way_round(da_grid_daily):
    da = da_grid_daily.isel(time=slice(0, 64), lat=0)
    a = power_spectrum(da, dim=("time", "lon"))
    b = power_spectrum(da, dim=("lon", "time"))
    xr.testing.assert_allclose(a, b.transpose(*a.dims))


def test_a_plane_wave_is_in_one_bin():
    t = 0.5 * np.arange(64)
    wave = xr.DataArray(3.0 * np.cos(2 * np.pi * 0.25 * t), dims="t", coords={"t": t})
    out = power_spectrum(wave, "t", window=None, detrend=None)
    assert float(out.idxmax("freq_t")) == 0.25
    assert float(out.max() * out["freq_t"][1]) == pytest.approx(4.5)


def test_white_noise_has_the_same_spectrum_with_and_without_a_taper():
    rng = np.random.default_rng(0)
    noise = xr.DataArray(
        rng.standard_normal((400, 64)), dims=("x", "t"), coords={"t": np.arange(64.0)}
    )
    for window in ("hann", np.hamming(64), None):
        out = power_spectrum(noise, "t", window=window, detrend=None).mean("x")
        np.testing.assert_allclose(out.isel(freq_t=slice(3, -2)).mean(), 2.0, rtol=0.02)


def test_the_cross_spectrum_sums_to_the_covariance_and_has_the_phase():
    t = np.arange(128.0)
    a = xr.DataArray(np.cos(2 * np.pi * t / 16), dims="t", coords={"t": t}, name="a")
    b = xr.DataArray(
        np.cos(2 * np.pi * (t - 2.0) / 16), dims="t", coords={"t": t}, name="b"
    )
    out = cross_spectrum(a, b, "t", window=None)
    assert out.name == "a_b_csd"
    step = float(out["freq_t"][1])
    assert float(out.real.sum()) * step == pytest.approx(float((a * b).mean()))
    # a leads b by an eighth of a period
    assert float(np.angle(out.sel(freq_t=1 / 16))) == pytest.approx(np.pi / 4)


def test_spectra_on_dask_are_lazy_and_equal(da_grid_daily):
    pytest.importorskip("dask.array")
    da = da_grid_daily.isel(time=slice(0, 64))
    chunked = da.chunk({"time": 10, "lat": 3, "lon": 5})
    for call in (
        lambda x: power_spectrum(x, "time"),
        lambda x: power_spectrum(x, ("time", "lon")),
        lambda x: isotropic_power_spectrum(x, ("lat", "lon")),
        lambda x: cross_spectrum(x, x.rename("b"), "time", segment=16),
        lambda x: coherence(x, x.rename("b"), "time", average="lat"),
        lambda x: rotary_spectrum(x, x, dim="time")["psd_cw"],
        lambda x: ke_spectral_flux(x, x, dim=("lon", "lat"))["flux"],
        lambda x: stft(x, "time", window_size=16),
        lambda x: integral_scale(power_spectrum(x, "time"), wavenumber_dim="freq_time"),
    ):
        lazy = call(chunked)
        assert lazy.chunks is not None
        xr.testing.assert_allclose(lazy.compute(), call(da))


def test_an_uneven_or_missing_coordinate_raises(da_grid_daily):
    da = da_grid_daily.isel(time=slice(0, 16), lat=0, lon=0)
    uneven = da.assign_coords(time=np.arange(16.0) ** 1.1)
    with pytest.raises(ValueError, match="evenly spaced"):
        power_spectrum(uneven, "time")
    with pytest.raises(ValueError, match="no coordinate"):
        power_spectrum(da.drop_vars("time"), "time")


def test_cftime_dates_are_seconds():
    pytest.importorskip("cftime")
    time = xr.date_range("2001-01-01", periods=32, calendar="noleap", use_cftime=True)
    da = xr.DataArray(np.arange(32.0) % 5, dims="time", coords={"time": time})
    out = power_spectrum(da, "time")
    np.testing.assert_allclose(out["freq_time"], np.fft.rfftfreq(32, 86400.0))


def test_the_phase_of_an_stft_is_that_of_the_coordinate():
    t = 3.0 + 0.5 * np.arange(64)
    da = xr.DataArray(
        np.random.default_rng(0).standard_normal(64),
        dims="t",
        coords={"t": t},
        name="a",
    )
    out = stft(da, "t", window_size=16, hop=8, window=None, detrend=None)
    f = float(out["freq_t"][11])
    expected = [
        np.sum(da.values[s : s + 16] * np.exp(-2j * np.pi * f * t[s : s + 16]))
        for s in range(0, 49, 8)
    ]
    np.testing.assert_allclose(out.isel(freq_t=11), expected)
