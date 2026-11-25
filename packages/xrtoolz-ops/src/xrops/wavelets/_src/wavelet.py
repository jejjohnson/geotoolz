"""2-D Morlet continuous wavelet spectra for xarray fields."""

from __future__ import annotations

import numpy as np
import xarray as xr
from diagnostix import spectra
from jaxtyping import Complex, Float
from scipy.fft import fft2, fftfreq, ifft2

from xrops.utils._src.spacing import coord_spacing
from xrops.utils._src.validation import _require_dims
from xrops.wavelets._src.wavelet_utils import (
    _scale_values,
    build_coi_mask,
    scale_to_wavenumber,
)


def morlet2_ft(
    kx: Float[np.ndarray, "y x"],
    ky: Float[np.ndarray, "y x"],
    s: float,
    theta: float,
    *,
    x0: float,
    k0: float = 1.0,
    a: float = 1.0,
) -> Complex[np.ndarray, "y x"]:
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
    ntheta: int = 24,
    k0: float = 1.0,
    a: float = 1.0,
) -> xr.DataArray:
    """Compute a 2-D directional Morlet CWT on a Cartesian grid.

    Args:
        da: Two-dimensional field.
        s: One-dimensional positive scale coordinate.
        dim: Spatial dimensions as ``(y, x)``.
        x0: Reference length scale in the same units as ``dim`` coords.
        ntheta: Number of evenly spaced angles in ``[0, π)``: for a real field
            the coefficient at ``θ + π`` is the conjugate of that at ``θ``. With
            fewer than 16 the power depends on the direction of a wave.
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
    arr = _filled(arr)

    ky = fftfreq(arr.shape[0], d=dy)
    kx = fftfreq(arr.shape[1], d=dx)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    field_ft = fft2(arr)

    scale_values = _scale_values(s)
    angles = np.linspace(0.0, np.pi, int(ntheta), endpoint=False)
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
    ntheta: int = 24,
) -> xr.DataArray:
    """The isotropic spectrum of ``da`` about each point, from Morlet wavelets.

    The density of variance over wavenumber, whose integral over every scale is
    the variance about a point (``diagnostix.spectra.wavelet_power_2d``): its
    mean over the domain is the isotropic spectrum, smoothed. The squared
    magnitude of :func:`cwt2` has the direction too.

    Args:
        da: Two-dimensional field on a Cartesian grid. A missing cell is filled
            with the mean, and left out of the ``coi_mask``.
        s: One-dimensional positive scale coordinate, increasing.
        dim: Spatial dimensions as ``(y, x)``.
        x0: Reference length scale in the same units as ``dim`` coords: a scale
            is a wavelength of ``s x0``.
        ntheta: Number of directions in ``[0, π)``. With 24 the power of a plane
            wave does not depend on its direction.

    Returns:
        DataArray with dims ``("scale", y, x)``, a boolean ``coi_mask`` and a
        ``wavenumber`` coordinate.
    """
    ydim, xdim = dim
    s = _as_scale_dataarray(s)
    _validate_cwt_input(da, s, dim=dim, ntheta=ntheta)
    field = da.transpose(ydim, xdim)
    arr = _filled(np.asarray(field.values))
    steps = (coord_spacing(da, ydim), coord_spacing(da, xdim))
    local = spectra.wavelet_power_2d(
        arr, steps, _scale_values(s) * x0, n_angles=int(ntheta)
    )
    scale_dim = s.dims[0]
    power = xr.DataArray(
        local.power,
        dims=(scale_dim, ydim, xdim),
        coords={scale_dim: s, ydim: da[ydim], xdim: da[xdim]},
        name=f"{da.name or 'field'}_wpsd",
        attrs={**da.attrs, "long_name": "2-D Morlet wavelet power spectrum"},
    )
    return power.assign_coords(
        coi_mask=build_coi_mask(da, s, dim=dim, x0=x0),
        wavenumber=scale_to_wavenumber(s, x0=x0),
    )


def wvlt_cross_spectrum(
    da1: xr.DataArray,
    da2: xr.DataArray,
    s: xr.DataArray,
    *,
    dim: tuple[str, str] = ("y", "x"),
    x0: float = 50e3,
    ntheta: int = 24,
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


def _filled(arr: np.ndarray) -> np.ndarray:
    """A field with its missing cells at the mean, or 0 if it has none."""
    missing = np.isnan(arr)
    if not missing.any():
        return arr
    fill = 0.0 if missing.all() else float(np.nanmean(arr))
    return np.where(missing, fill, arr)


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
    _require_dims(da, *dim)
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
