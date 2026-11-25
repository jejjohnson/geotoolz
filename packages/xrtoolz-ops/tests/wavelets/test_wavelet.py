"""Tests for 2-D Morlet wavelet spectra."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr
from pipekit_ops import Sequential

from xrops.wavelets import (
    WaveletPowerSpectrum,
    WaveletScalogram,
    WaveletSignificance,
    build_coi_mask,
    cwt1d,
    cwt2,
    dominant_period_map,
    geometric_scales,
    icwt1d,
    scale_to_wavenumber,
    wavelet_significance,
    wavenumber_to_scale,
    wvlt_power_spectrum,
)


def _plane_wave(nx: int = 64, ny: int = 64, wavelength: float = 4.0) -> xr.DataArray:
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    xx, _ = np.meshgrid(x, y)
    field = np.cos(2.0 * np.pi * xx / wavelength)
    return xr.DataArray(field, dims=("y", "x"), coords={"y": y, "x": x}, name="ssh")


def _sine_wave(n: int = 128, period: float = 8.0) -> xr.DataArray:
    time = np.arange(n, dtype=float)
    data = np.sin(2.0 * np.pi * time / period)
    return xr.DataArray(data, dims="time", coords={"time": time}, name="signal")


def test_scale_wavenumber_round_trip() -> None:
    scales = geometric_scales(1.0, octaves=2, voices_per_octave=2)
    k = scale_to_wavenumber(scales, x0=2.0, k0=3.0)
    xr.testing.assert_allclose(wavenumber_to_scale(k, x0=2.0, k0=3.0), scales)


def test_cwt1d_outputs_power_rectification_and_coi() -> None:
    da = _sine_wave()
    out = cwt1d(da)
    assert set(out.data_vars) == {"wave", "power", "power_rect", "coi", "coi_mask"}
    assert out["wave"].dims == ("scale", "time")
    assert out["coi_mask"].dims == ("scale", "time")
    xr.testing.assert_allclose(out["power_rect"], out["power"] / out["scale"])
    assert np.iscomplexobj(out["wave"].values)


@pytest.mark.parametrize("period", [4.0, 8.0, 16.0])
def test_cwt1d_recovers_synthetic_sine_period(period: float) -> None:
    da = _sine_wave(period=period)
    out = cwt1d(da, dj=0.125)
    spectrum = out["power_rect"].where(out["coi_mask"]).mean("time", skipna=True)
    peak_scale = spectrum.idxmax("scale")
    peak_period = out["period"].sel(scale=peak_scale)
    sig = wavelet_significance(out["power_rect"], null="white")
    assert bool(sig.sel(scale=peak_scale).any())
    assert float(peak_period) == pytest.approx(period, rel=0.07)


def test_cwt1d_supports_paul_and_dog_mothers() -> None:
    da = _sine_wave(n=64)
    for mother in ("paul", "dog"):
        out = cwt1d(da, mother=mother, dj=0.5)
        assert out["wave"].dims == ("scale", "time")
        assert np.isfinite(out["power_rect"]).all()


def test_icwt1d_reconstructs_morlet_signal() -> None:
    da = _sine_wave(n=128, period=8.0)
    out = cwt1d(da, dj=0.125, pad=False)
    rec = icwt1d(out["wave"])
    corr = np.corrcoef(da.values, rec.values)[0, 1]
    assert corr == pytest.approx(1.0, abs=1e-3)
    np.testing.assert_allclose(rec, da, atol=0.02)


def test_wavelet_significance_and_dominant_period_map() -> None:
    da = _sine_wave(period=8.0)
    out = cwt1d(da)
    sig = wavelet_significance(out["power_rect"], null="white")
    assert sig.dims == ("scale", "time")
    assert sig.dtype == bool
    red = wavelet_significance(out["power_rect"], null="red", alpha=0.5)
    assert red.attrs["null"] == "red"
    assert red.attrs["alpha"] == 0.5
    pmap = dominant_period_map(
        out["power_rect"], coi_mask=out["coi_mask"], signif_mask=sig
    )
    assert float(pmap) == pytest.approx(8.0, rel=0.07)


def test_red_noise_significance_is_stricter_for_ar1_signal() -> None:
    rng = np.random.default_rng(0)
    values = np.empty(128, dtype=float)
    values[0] = 0.0
    for i in range(1, values.size):
        values[i] = 0.8 * values[i - 1] + rng.normal()
    da = xr.DataArray(
        values,
        dims="time",
        coords={"time": np.arange(values.size, dtype=float)},
        name="signal",
    )
    out = cwt1d(da)
    white = wavelet_significance(out["power_rect"], null="white")
    red = wavelet_significance(out["power_rect"], null="red", alpha=0.8)
    assert int(red.sum()) < int(white.sum())


def test_cwt1d_pixelwise_over_outer_dimensions() -> None:
    da = xr.concat([_sine_wave(period=8.0), _sine_wave(period=16.0)], dim="lat")
    da = da.assign_coords(lat=[0.0, 1.0])
    out = cwt1d(da)
    assert out["wave"].dims == ("scale", "time", "lat")
    pmap = dominant_period_map(out["power_rect"], coi_mask=out["coi_mask"])
    assert pmap.dims == ("lat",)
    assert float(pmap.sel(lat=0.0)) < float(pmap.sel(lat=1.0))


def test_cwt1d_rejects_irregular_coordinates() -> None:
    da = xr.DataArray([1.0, 2.0, 3.0], dims="time", coords={"time": [0.0, 1.0, 3.0]})
    with pytest.raises(ValueError, match="uniformly spaced"):
        cwt1d(da)


def test_per_series_source_variance_drives_per_index_threshold() -> None:
    """High-variance series should clear a stricter absolute threshold than
    a low-variance series even though both share the same null spectrum."""
    big = _sine_wave(period=8.0) * 5.0
    small = _sine_wave(period=8.0) * 0.1
    da = xr.concat([big, small], dim="series").assign_coords(series=[0, 1])
    out = cwt1d(da)
    assert "source_variance" in out["wave"].coords
    sig = wavelet_significance(out["power_rect"], null="white")
    assert int(sig.sel(series=0).sum()) > 0
    assert int(sig.sel(series=1).sum()) == int(sig.sel(series=0).sum())


@pytest.mark.parametrize("mother", ["morlet", "paul", "dog"])
def test_icwt1d_gives_the_signal_back(mother: str) -> None:
    """The hat came back with its sign flipped (#786)."""
    t = 0.25 * np.arange(300)
    x = np.sin(2 * np.pi * t / 8) * np.exp(-(((t - 37) / 12) ** 2))
    da = xr.DataArray(x - x.mean(), dims="time", coords={"time": t})
    wave = cwt1d(da, mother=mother, s0=0.1, dj=0.125)["wave"]
    back = icwt1d(wave)
    assert back.dims == ("time",)
    assert float((back * da).sum() / (da**2).sum()) == pytest.approx(1.0, abs=0.01)
    xr.testing.assert_allclose(icwt1d(wave, mother=mother), back)


def test_icwt1d_of_a_band_keeps_the_waves_in_it() -> None:
    t = np.arange(512.0)
    slow, fast = np.cos(2 * np.pi * t / 64), np.cos(2 * np.pi * t / 8)
    da = xr.DataArray(slow + fast, dims="time", coords={"time": t})
    out = cwt1d(da, dj=0.125, pad=False)
    back = icwt1d(out["wave"], band=(16.0, 4.0))
    np.testing.assert_allclose(back, fast, atol=0.05)


def test_icwt1d_of_a_stack_and_of_an_odd_derivative() -> None:
    da = xr.concat([_sine_wave(), 2.0 * _sine_wave()], dim="lat")
    wave = cwt1d(da, dj=0.125)["wave"]
    back = icwt1d(wave)
    assert back.dims == ("time", "lat")
    np.testing.assert_allclose(back.isel(lat=1), 2.0 * back.isel(lat=0))
    with pytest.raises(ValueError, match="has no inverse"):
        icwt1d(cwt1d(da, mother="dog", param=3)["wave"])


@pytest.mark.parametrize("mother", ["morlet", "paul", "dog"])
def test_white_noise_passes_at_its_level(mother: str) -> None:
    """The hat, which has one degree of freedom, passed 8.3 % at 5 % (#787)."""
    x = np.random.default_rng(0).standard_normal((20, 2048))
    da = xr.DataArray(x, dims=("series", "time"), coords={"time": np.arange(2048.0)})
    out = cwt1d(da, mother=mother, j_max=16)
    passed = wavelet_significance(out["power"], null="white").where(out["coi_mask"])
    assert float(passed.mean()) == pytest.approx(0.05, abs=0.006)


@pytest.mark.parametrize("mother", ["morlet", "paul", "dog"])
def test_the_cone_is_where_a_spike_at_the_end_has_fallen_by_e2(mother: str) -> None:
    """The cone of Paul's was twice as wide (#789)."""
    t = 0.0125 * np.arange(4000)
    spike = xr.DataArray(np.zeros(4000), dims="time", coords={"time": t})
    spike[0] = 1.0
    out = cwt1d(spike, mother=mother, s0=1.0, j_max=0)
    power, inside = out["power"].isel(scale=0), out["coi_mask"].isel(scale=0)
    start = int(inside.argmax("time"))
    ratio = float(power[start] / power[0])
    assert ratio == pytest.approx(np.exp(-2.0), rel=0.06)
    # the line of the cone crosses the period at the same sample
    crossing = int((out["coi"] >= float(out["period"][0])).argmax("time"))
    assert crossing == start


def test_a_periodic_series_has_no_cone() -> None:
    out = cwt1d(_sine_wave(), pad=False)
    assert bool(out["coi_mask"].all())
    assert np.isinf(out["coi"]).all()


def test_the_significance_takes_the_mother_that_was_saved() -> None:
    out = cwt1d(_sine_wave(), mother="dog")
    saved = wavelet_significance(out["power"], null="white")
    given = wavelet_significance(out["power"], null="white", mother="dog")
    xr.testing.assert_identical(saved, given)
    other = wavelet_significance(out["power"], null="white", mother="morlet")
    assert int(other.sum()) != int(saved.sum())


def test_cwt1d_fills_gaps_and_takes_the_mean_away_when_asked() -> None:
    da = _sine_wave() + 10.0
    plain = cwt1d(_sine_wave(), detrend="constant")["wave"]
    xr.testing.assert_allclose(cwt1d(da, detrend="constant")["wave"], plain)
    assert float(abs(cwt1d(da)["wave"] - plain).max()) > 1.0
    gappy = da.where(da["time"] != 30.0)
    assert np.isfinite(cwt1d(gappy)["wave"]).all()


def test_cwt1d_on_dask_is_lazy_and_equal() -> None:
    pytest.importorskip("dask.array")
    da = xr.concat([_sine_wave(), _sine_wave(period=16.0)], dim="lat")
    lazy = cwt1d(da.chunk({"lat": 1}))["wave"]
    assert lazy.chunks is not None
    xr.testing.assert_allclose(lazy.compute(), cwt1d(da)["wave"])
    back = icwt1d(lazy)
    assert back.chunks is not None
    xr.testing.assert_allclose(back.compute(), icwt1d(cwt1d(da)["wave"]))


def test_wavelet_scalogram_and_significance_operators_compose() -> None:
    ds = _sine_wave().rename("nino3").to_dataset()
    pipe = Sequential(
        [
            WaveletScalogram("nino3", dj=0.5),
            WaveletSignificance("nino3_power_rect", null="white"),
        ]
    )
    out = pipe(ds)
    assert "nino3_wave" in out
    assert "nino3_power_rect_signif_mask" in out
    cfg = WaveletScalogram("nino3").get_config()
    assert json.loads(json.dumps(cfg)) == cfg


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
    power = wvlt_power_spectrum(da, scales, x0=1.0)
    trusted = power.where(power["coi_mask"]).mean(("y", "x"), skipna=True)
    # per unit of log wavenumber, a density times k has its peak at the wave
    assert float((trusted * trusted["wavenumber"]).idxmax("scale")) == 4.0


def test_the_integral_over_the_scales_is_the_variance() -> None:
    n = 128
    y, x = np.mgrid[:n, :n]
    da = xr.DataArray(
        3.0 * np.cos(2 * np.pi * (5 * x + 12 * y) / n),
        dims=("y", "x"),
        coords={"y": np.arange(float(n)), "x": np.arange(float(n))},
        name="ssh",
    )
    scales = geometric_scales(0.4, octaves=2.6, voices_per_octave=12)
    power = wvlt_power_spectrum(da, scales, x0=10.0)
    centre = power.isel(y=64, x=64)
    area = -np.trapezoid(centre, centre["wavenumber"])
    assert area == pytest.approx(4.5, rel=0.02)


def test_the_power_of_a_plane_wave_does_not_depend_on_its_direction() -> None:
    """It changed by 55 % with eight distinct directions (#788)."""
    n, wavelength = 64, 8.0
    y, x = np.mgrid[:n, :n] - n // 2
    scales = xr.DataArray([1.0], dims="scale", coords={"scale": [1.0]})
    values = []
    for angle in np.deg2rad(np.arange(0.0, 90.0, 5.0)):
        wave = np.cos(2 * np.pi * (x * np.cos(angle) + y * np.sin(angle)) / wavelength)
        da = xr.DataArray(
            wave, dims=("y", "x"), coords={"y": np.arange(n), "x": np.arange(n)}
        )
        power = wvlt_power_spectrum(da, scales, x0=wavelength)
        values.append(float(power.isel(scale=0, y=32, x=32)))
    assert np.ptp(values) < 1e-3 * np.mean(values)


def test_a_missing_cell_is_filled_and_masked() -> None:
    da = _plane_wave(nx=32, ny=32)
    da[10, 12] = np.nan
    power = wvlt_power_spectrum(da, xr.DataArray([2.0], dims="scale"), x0=1.0)
    assert np.isfinite(power).all()
    assert not bool(power["coi_mask"].isel(scale=0, y=10, x=12))


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
    from xrops.viz import (
        plot_dominant_period_map,
        plot_global_wavelet_spectrum,
        plot_resolved_scale_map,
        plot_scalogram,
        plot_wavelet_anisotropy,
        plot_wavelet_spectrum_1d,
    )

    da = _plane_wave()
    scales = xr.DataArray([2.0, 4.0], dims="scale")
    spectrum = abs(cwt2(da, scales, x0=1.0, ntheta=4)) ** 2
    assert (
        plot_resolved_scale_map(spectrum.isel(scale=0, angle=0)).name == "rectilinear"
    )
    assert (
        plot_wavelet_spectrum_1d(spectrum.isel(angle=0, y=24, x=24)).name
        == "rectilinear"
    )
    assert plot_wavelet_anisotropy(spectrum.isel(y=24, x=24)).name == "polar"
    out = cwt1d(_sine_wave())
    assert plot_scalogram(out["power_rect"], coi=out["coi"]).name == "rectilinear"
    assert plot_global_wavelet_spectrum(out["power_rect"]).name == "rectilinear"
    assert (
        plot_dominant_period_map(xr.DataArray([[2.0]], dims=("y", "x"))).name
        == "rectilinear"
    )


def test_morlet2_ft_peaks_at_expected_wavenumber() -> None:
    """The Morlet kernel peaks at the central wavenumber kc = k0/(s*x0)
    along its rotation axis. Catches sign/normalization regressions in
    the analytic Fourier transform."""
    from xrops.wavelets._src.wavelet import morlet2_ft

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


def test_wvlt_cross_spectrum_of_a_field_with_itself_is_its_power() -> None:
    from xrops.wavelets._src.wavelet import wvlt_cross_spectrum

    da = _plane_wave(nx=32, ny=32, wavelength=4.0)
    scales = xr.DataArray([2.0, 4.0], dims="scale")
    csd = wvlt_cross_spectrum(da, da, scales, x0=1.0, ntheta=4)
    power = (abs(cwt2(da, scales, x0=1.0, ntheta=4)) ** 2).mean("angle")
    np.testing.assert_allclose(np.asarray(csd.imag), 0.0, atol=1e-12)
    np.testing.assert_allclose(csd.real, power)


def test_plot_wavelet_spectrum_1d_rejects_higher_dimensional_input() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from xrops.viz import plot_wavelet_spectrum_1d

    da = _plane_wave()
    scales = xr.DataArray([2.0, 4.0], dims="scale")
    spectrum = wvlt_power_spectrum(da, scales, x0=1.0, ntheta=4)
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
