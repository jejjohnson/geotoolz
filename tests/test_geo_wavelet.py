"""Tests for 2-D Morlet wavelet spectra."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xr_toolz.core import Sequential
from xr_toolz.geo import (
    WaveletPowerSpectrum,
    build_coi_mask,
    cwt2,
    geometric_scales,
    scale_to_wavenumber,
    wavenumber_to_scale,
    wvlt_power_spectrum,
)


def _plane_wave(nx: int = 64, ny: int = 64, wavelength: float = 4.0) -> xr.DataArray:
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    xx, _ = np.meshgrid(x, y)
    field = np.cos(2.0 * np.pi * xx / wavelength)
    return xr.DataArray(field, dims=("y", "x"), coords={"y": y, "x": x}, name="ssh")


def test_scale_wavenumber_round_trip() -> None:
    scales = geometric_scales(1.0, octaves=2, voices_per_octave=2)
    k = scale_to_wavenumber(scales, x0=2.0, k0=3.0)
    xr.testing.assert_allclose(wavenumber_to_scale(k, x0=2.0, k0=3.0), scales)


def test_cwt2_outputs_directional_coefficients_and_coi() -> None:
    da = _plane_wave()
    scales = xr.DataArray([2.0, 4.0, 8.0], dims="scale")
    out = cwt2(da, scales, x0=1.0, ntheta=8)
    assert out.dims == ("scale", "angle", "y", "x")
    assert out.sizes["angle"] == 8
    assert out["coi_mask"].dims == ("scale", "y", "x")
    assert np.iscomplexobj(out.values)


def test_plane_wave_power_peaks_at_matching_scale() -> None:
    da = _plane_wave(wavelength=4.0)
    scales = xr.DataArray([2.0, 4.0, 8.0], dims="scale")
    power = wvlt_power_spectrum(da, scales, x0=1.0, ntheta=8, isotropic=False)
    trusted = power.where(power["coi_mask"]).mean(("y", "x"), skipna=True)
    peak = trusted.max("angle").idxmax("scale")
    assert float(peak) == pytest.approx(4.0)


def test_power_normalization_recovers_variance_on_trusted_pixels() -> None:
    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.standard_normal((48, 48)),
        dims=("y", "x"),
        coords={"y": np.arange(48.0), "x": np.arange(48.0)},
        name="ssh",
    )
    scales = xr.DataArray([1.0, 2.0, 4.0, 8.0], dims="scale")
    power = wvlt_power_spectrum(da, scales, x0=0.5, ntheta=8, isotropic=False)
    dlog = xr.DataArray(
        np.gradient(np.log(scales.values)),
        dims=("scale",),
        coords={"scale": scales},
    )
    integral = (power * dlog).sum("scale") * (2.0 * np.pi / power.sizes["angle"])
    recovered = float(integral.sum("angle").where(power["coi_mask"]).mean())
    assert recovered == pytest.approx(float(da.var()), rel=0.01)


def test_coi_mask_shrinks_with_scale_and_nan_cells() -> None:
    da = _plane_wave(nx=16, ny=16)
    da = da.where(~((da["x"] == 8) & (da["y"] == 8)))
    scales = xr.DataArray([1.0, 4.0], dims="scale")
    mask = build_coi_mask(da, scales, x0=1.0)
    assert int(mask.sel(scale=4.0).sum()) < int(mask.sel(scale=1.0).sum())
    assert not bool(mask.sel(scale=1.0, y=8, x=8))


def test_wavelet_power_spectrum_operator_composes() -> None:
    da = _plane_wave()
    ds = da.to_dataset()
    scales = xr.DataArray([2.0, 4.0], dims="scale")
    op = WaveletPowerSpectrum("ssh", scales=scales, x0=1.0, ntheta=4)
    out = Sequential([op])(ds)
    assert "ssh_wpsd" in out
    assert out["ssh_wpsd"].dims == ("scale", "y", "x")
    assert json.loads(json.dumps(op.get_config())) == op.get_config()


def test_wavelet_plot_helpers_return_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from xr_toolz.geo.plot import (
        plot_resolved_scale_map,
        plot_wavelet_anisotropy,
        plot_wavelet_spectrum_1d,
    )

    da = _plane_wave()
    scales = xr.DataArray([2.0, 4.0], dims="scale")
    spectrum = wvlt_power_spectrum(da, scales, x0=1.0, ntheta=4, isotropic=False)
    assert (
        plot_resolved_scale_map(spectrum.isel(scale=0, angle=0)).name == "rectilinear"
    )
    assert (
        plot_wavelet_spectrum_1d(spectrum.isel(angle=0, y=24, x=24)).name
        == "rectilinear"
    )
    assert plot_wavelet_anisotropy(spectrum.isel(y=24, x=24)).name == "polar"


def test_morlet2_ft_peaks_at_expected_wavenumber() -> None:
    """The Morlet kernel peaks at the central wavenumber kc = k0/(s*x0)
    along its rotation axis. Catches sign/normalization regressions in
    the analytic Fourier transform."""
    from xr_toolz.geo._src.wavelet import morlet2_ft

    x0 = 1.0
    k0 = 3.0
    s = 2.0
    theta = 0.0
    sigma = s * x0
    kc = k0 / sigma
    kx = np.linspace(-2.0, 2.0, 41)
    ky = np.linspace(-2.0, 2.0, 41)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    psi_hat = morlet2_ft(KX, KY, s=s, theta=theta, x0=x0, k0=k0)
    assert psi_hat.shape == KX.shape
    assert np.iscomplexobj(psi_hat)
    peak_idx = np.unravel_index(np.abs(psi_hat).argmax(), psi_hat.shape)
    np.testing.assert_allclose(KX[peak_idx], kc, atol=2 * (kx[1] - kx[0]))
    np.testing.assert_allclose(KY[peak_idx], 0.0, atol=2 * (ky[1] - ky[0]))


def test_wvlt_cross_spectrum_self_matches_power_spectrum_up_to_constant() -> None:
    """Cross-spectrum of a field with itself is the *raw* auto-power
    (no variance normalization), so it differs from
    ``wvlt_power_spectrum`` by a single overall scalar — verifying the
    ratio is spatially constant locks in the cross-spectrum's
    normalization convention."""
    from xr_toolz.geo._src.wavelet import wvlt_cross_spectrum

    da = _plane_wave(nx=32, ny=32, wavelength=4.0)
    scales = xr.DataArray([2.0, 4.0], dims="scale")
    csd = wvlt_cross_spectrum(da, da, scales, x0=1.0, ntheta=4, isotropic=True)
    psd = wvlt_power_spectrum(da, scales, x0=1.0, ntheta=4, isotropic=True)
    assert csd.dims == psd.dims
    np.testing.assert_allclose(np.asarray(csd.imag), 0.0, atol=1e-12)
    psd_arr = np.asarray(psd)
    ratio = np.asarray(csd.real)[psd_arr > 1e-10] / psd_arr[psd_arr > 1e-10]
    if ratio.size:
        np.testing.assert_allclose(ratio, ratio.mean(), rtol=1e-6)


def test_plot_wavelet_spectrum_1d_rejects_higher_dimensional_input() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from xr_toolz.geo.plot import plot_wavelet_spectrum_1d

    da = _plane_wave()
    scales = xr.DataArray([2.0, 4.0], dims="scale")
    spectrum = wvlt_power_spectrum(da, scales, x0=1.0, ntheta=4, isotropic=False)
    with pytest.raises(ValueError, match="1-D"):
        plot_wavelet_spectrum_1d(spectrum)


def test_scales_validation_rejects_non_finite_and_non_monotone() -> None:
    da = _plane_wave()
    bad_nan = xr.DataArray([2.0, np.nan, 4.0], dims="scale")
    bad_unsorted = xr.DataArray([4.0, 2.0, 8.0], dims="scale")
    with pytest.raises(ValueError, match="finite"):
        cwt2(da, bad_nan, x0=1.0, ntheta=4)
    with pytest.raises(ValueError, match="increasing"):
        cwt2(da, bad_unsorted, x0=1.0, ntheta=4)
