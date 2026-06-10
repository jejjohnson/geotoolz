"""2-D Morlet continuous wavelet spectra for xarray fields."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.fft import fft2, fftfreq, ifft2

from xrtoolz.geo._src.wavelet_utils import (
    _require_dims,
    _scale_values,
    build_coi_mask,
    scale_to_wavenumber,
)
from xrtoolz.utils._src.spacing import coord_spacing


def morlet2_ft(
    kx: np.ndarray,
    ky: np.ndarray,
    s: float,
    theta: float,
    *,
    x0: float,
    k0: float = 1.0,
    a: float = 1.0,
) -> np.ndarray:
    """Analytic Fourier transform of a rotated 2-D Morlet wavelet.

    Args:
        kx: Zonal wavenumber grid in cycles per coordinate unit.
        ky: Meridional wavenumber grid in cycles per coordinate unit.
        s: Positive dimensionless scale.
        theta: Wavelet orientation in radians.
        x0: Reference length scale in coordinate units.
        k0: Dimensionless central wavenumber.
        a: Amplitude multiplier.

    Returns:
        Complex Fourier-domain wavelet sampled on ``kx``/``ky``.
    """
    if s <= 0:
        raise ValueError("s must be strictly positive")
    if x0 <= 0:
        raise ValueError("x0 must be strictly positive")
    if k0 <= 0:
        raise ValueError("k0 must be strictly positive")
    sigma = s * x0
    kc = k0 / sigma
    kcx = kc * np.cos(theta)
    kcy = kc * np.sin(theta)
    envelope = np.exp(-2.0 * np.pi**2 * sigma**2 * ((kx - kcx) ** 2 + (ky - kcy) ** 2))
    return (a * 2.0 * np.pi * sigma**2 * envelope).astype(np.complex128)


def cwt2(
    da: xr.DataArray,
    s: xr.DataArray,
    dim: tuple[str, str] = ("y", "x"),
    *,
    x0: float = 50e3,
    ntheta: int = 16,
    k0: float = 1.0,
    a: float = 1.0,
) -> xr.DataArray:
    """Compute a 2-D directional Morlet CWT on a Cartesian grid.

    Args:
        da: Two-dimensional field.
        s: One-dimensional positive scale coordinate.
        dim: Spatial dimensions as ``(y, x)``.
        x0: Reference length scale in the same units as ``dim`` coords.
        ntheta: Number of evenly spaced angles in ``[0, 2π)``.
        k0: Dimensionless Morlet central wavenumber.
        a: Fourier-domain amplitude multiplier.

    Returns:
        Complex coefficients with dims ``("scale", "angle", y, x)`` and a
        boolean ``coi_mask`` coordinate with dims ``("scale", y, x)``.
    """
    ydim, xdim = dim
    s = _as_scale_dataarray(s)
    _validate_cwt_input(da, s, dim=dim, ntheta=ntheta)
    dy = coord_spacing(da, ydim)
    dx = coord_spacing(da, xdim)
    arr = np.asarray(da.transpose(ydim, xdim).values)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        fill = float(np.nanmean(arr)) if not np.all(nan_mask) else 0.0
        arr = np.where(nan_mask, fill, arr)

    ky = fftfreq(arr.shape[0], d=dy)
    kx = fftfreq(arr.shape[1], d=dx)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    field_ft = fft2(arr)

    scale_values = _scale_values(s)
    angles = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    coeffs = np.empty((scale_values.size, angles.size, *arr.shape), dtype=np.complex128)
    for i, scale in enumerate(scale_values):
        for j, angle in enumerate(angles):
            kernel = morlet2_ft(
                kx_grid,
                ky_grid,
                float(scale),
                float(angle),
                x0=x0,
                k0=k0,
                a=a,
            )
            coeffs[i, j] = ifft2(field_ft * np.conj(kernel))

    scale_dim = s.dims[0]
    out = xr.DataArray(
        coeffs,
        dims=(scale_dim, "angle", ydim, xdim),
        coords={
            scale_dim: s,
            "angle": angles,
            ydim: da[ydim],
            xdim: da[xdim],
        },
        name=f"{da.name or 'field'}_cwt2",
        attrs=dict(da.attrs),
    )
    out = out.assign_coords(
        coi_mask=build_coi_mask(da, s, dim=dim, x0=x0),
        wavenumber=scale_to_wavenumber(s, x0=x0, k0=k0),
    )
    out["wavenumber"].attrs["units"] = f"cycles per {xdim}/{ydim} coordinate unit"
    return out


def wvlt_power_spectrum(
    da: xr.DataArray,
    s: xr.DataArray,
    *,
    dim: tuple[str, str] = ("y", "x"),
    x0: float = 50e3,
    ntheta: int = 16,
    k0: float = 1.0,
    isotropic: bool = True,
) -> xr.DataArray:
    """Return the 2-D Morlet wavelet power spectrum of ``da``."""
    s = _as_scale_dataarray(s)
    coeffs = cwt2(da, s, dim=dim, x0=x0, ntheta=ntheta, k0=k0)
    power = xr.apply_ufunc(np.abs, coeffs) ** 2
    power = _normalize_power(power, da)
    power.name = f"{da.name or 'field'}_wpsd"
    power.attrs.update({"long_name": "2-D Morlet wavelet power spectrum"})
    if isotropic:
        power = power.mean("angle", keep_attrs=True)
    return power


def wvlt_cross_spectrum(
    da1: xr.DataArray,
    da2: xr.DataArray,
    s: xr.DataArray,
    *,
    dim: tuple[str, str] = ("y", "x"),
    x0: float = 50e3,
    ntheta: int = 16,
    k0: float = 1.0,
    isotropic: bool = True,
) -> xr.DataArray:
    """Return the complex 2-D Morlet cross-spectrum of two fields."""
    s = _as_scale_dataarray(s)
    w1 = cwt2(da1, s, dim=dim, x0=x0, ntheta=ntheta, k0=k0)
    w2 = cwt2(da2, s, dim=dim, x0=x0, ntheta=ntheta, k0=k0)
    cross = w1 * np.conj(w2)
    if isotropic:
        cross = cross.mean("angle", keep_attrs=True)
    cross.name = f"{da1.name or 'field'}_{da2.name or 'field'}_wcsd"
    return cross


def _normalize_power(power: xr.DataArray, source: xr.DataArray) -> xr.DataArray:
    """Normalize trusted wavelet power to match source-field variance."""
    scale_dim = power.dims[0]
    scales = np.asarray(power[scale_dim].values, dtype=float)
    if scales.size < 2:
        return power
    log_scales = np.log(scales)
    dlog = np.gradient(log_scales)
    dtheta = 2.0 * np.pi / power.sizes["angle"]
    weights = xr.DataArray(
        dlog,
        dims=(scale_dim,),
        coords={scale_dim: power[scale_dim]},
    )
    integral = (power * weights).sum(dim=[scale_dim])
    integral = (integral * dtheta).sum("angle")
    denom = float(integral.where(power["coi_mask"]).mean(skipna=True).real)
    variance = float(source.var(skipna=True))
    if denom <= 0 or not np.isfinite(denom) or not np.isfinite(variance):
        return power
    return power * (variance / denom)


def _as_scale_dataarray(scales) -> xr.DataArray:
    """Coerce array-like scales to a one-dimensional scale DataArray."""
    if isinstance(scales, xr.DataArray):
        return scales
    values = np.asarray(scales, dtype=float)
    return xr.DataArray(values, dims=("scale",), coords={"scale": values}, name="scale")


def _validate_cwt_input(
    da: xr.DataArray,
    s: xr.DataArray,
    *,
    dim: tuple[str, str],
    ntheta: int,
) -> None:
    """Validate CWT dimensions, scales, angles, and spatial chunking."""
    _require_dims(da, dim)
    if da.ndim != 2:
        raise ValueError(f"cwt2 expects a 2-D DataArray; got dims {da.dims}.")
    if ntheta <= 0:
        raise ValueError("ntheta must be positive")
    _scale_values(s)
    _raise_if_spatially_chunked(da, dim)


def _raise_if_spatially_chunked(da: xr.DataArray, dim: tuple[str, str]) -> None:
    """Raise when spatial dimensions have multiple chunks."""
    chunksizes = getattr(da, "chunksizes", {})
    for d in dim:
        chunks = chunksizes.get(d)
        if chunks is not None and len(chunks) > 1:
            raise ValueError(
                "cwt2 requires each spatial dimension to be a single chunk; "
                f"dimension {d!r} has chunks {tuple(chunks)!r}."
            )
