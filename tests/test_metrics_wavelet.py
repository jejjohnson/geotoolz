"""Tests for wavelet spectral metrics."""

from __future__ import annotations

import json

import numpy as np
import xarray as xr

from xr_toolz.metrics import (
    WaveletPSDScore,
    wavelet_psd_score,
    wavelet_resolved_scale_map,
)


def _dataset(nx: int = 48, ny: int = 48) -> xr.Dataset:
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    xx, yy = np.meshgrid(x, y)
    field = np.cos(2.0 * np.pi * xx / 6.0) + 0.2 * np.sin(2.0 * np.pi * yy / 9.0)
    return xr.Dataset({"ssh": (("y", "x"), field)}, coords={"y": y, "x": x})


def test_wavelet_psd_score_perfect_prediction_is_one_on_trusted_pixels() -> None:
    ds = _dataset()
    scales = xr.DataArray([2.0, 4.0, 8.0], dims="scale")
    out = wavelet_psd_score(ds, ds, "ssh", scales, x0=1.0, ntheta=4)
    trusted = out["score"].where(out["score"]["coi_mask"])
    values = trusted.values[np.isfinite(trusted.values)]
    np.testing.assert_allclose(values, 1.0, rtol=1e-12, atol=1e-12)


def test_wavelet_psd_score_operator_config_and_call() -> None:
    ds = _dataset()
    scales = xr.DataArray([2.0, 4.0], dims="scale")
    op = WaveletPSDScore("ssh", scales=scales, x0=1.0, ntheta=4)
    out = op(ds, ds)
    assert "score" in out
    assert json.loads(json.dumps(op.get_config())) == op.get_config()


def test_wavelet_resolved_scale_map_returns_spatial_field_with_coi_nans() -> None:
    ds = _dataset()
    # Use a degraded prediction so the score column actually crosses
    # the 0.5 threshold somewhere along the scale axis — perfect
    # skill (truth==pred) is a NaN-everywhere case by design now.
    rng = np.random.default_rng(0)
    pred = ds["ssh"] + 0.5 * xr.DataArray(
        rng.standard_normal(ds["ssh"].shape),
        dims=ds["ssh"].dims,
        coords=ds["ssh"].coords,
    )
    scales = xr.DataArray([2.0, 4.0, 8.0], dims="scale")
    out = wavelet_resolved_scale_map(
        truth=ds["ssh"],
        pred=pred,
        scales=scales,
        x0=1.0,
        ntheta=4,
    )
    assert out.dims == ("y", "x")
    assert out.attrs["units"] == "km"
    # Interior points have enough scale samples + a real crossing.
    assert bool(np.isfinite(out.sel(y=24, x=24)))
    # COI mask invalidates the corners.
    assert bool(np.isnan(out.sel(y=0, x=0)))


def test_wavelet_resolved_scale_map_returns_nan_for_perfect_or_hopeless_skill() -> None:
    """Without a threshold-crossing guard, identical truth/pred would
    extrapolate to an arbitrary edge wavelength. With the guard it
    must return NaN for these unsolvable columns."""
    ds = _dataset()
    scales = xr.DataArray([2.0, 4.0, 8.0], dims="scale")
    out = wavelet_resolved_scale_map(
        truth=ds["ssh"],
        pred=ds["ssh"],
        scales=scales,
        x0=1.0,
        ntheta=4,
    )
    assert bool(np.isnan(out).all())


def test_wavelet_resolved_scale_map_honours_wavelength_scale_kwarg() -> None:
    """``wavelength_scale=1.0`` keeps the result in the coord units;
    the default 1e-3 collapses metres to km."""
    ds = _dataset()
    rng = np.random.default_rng(0)
    pred = ds["ssh"] + 0.5 * xr.DataArray(
        rng.standard_normal(ds["ssh"].shape),
        dims=ds["ssh"].dims,
        coords=ds["ssh"].coords,
    )
    scales = xr.DataArray([2.0, 4.0, 8.0], dims="scale")
    km = wavelet_resolved_scale_map(
        truth=ds["ssh"],
        pred=pred,
        scales=scales,
        x0=1.0,
        ntheta=4,
    )
    raw = wavelet_resolved_scale_map(
        truth=ds["ssh"],
        pred=pred,
        scales=scales,
        x0=1.0,
        ntheta=4,
        wavelength_scale=1.0,
        wavelength_units="cycles",
    )
    finite = np.isfinite(km) & np.isfinite(raw)
    if bool(finite.any()):
        ratio = (raw.where(finite) / km.where(finite)).values
        ratio = ratio[np.isfinite(ratio)]
        np.testing.assert_allclose(ratio, 1000.0, rtol=1e-9)
    assert raw.attrs["units"] == "cycles"
