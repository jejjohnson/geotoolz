"""Helpers for 2-D Morlet wavelet spectra."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt


def geometric_scales(
    s0: float,
    *,
    octaves: float,
    voices_per_octave: int = 8,
    name: str = "scale",
) -> xr.DataArray:
    """Return a geometric wavelet scale grid.

    Args:
        s0: Smallest positive scale.
        octaves: Number of octaves to span from ``s0``.
        voices_per_octave: Number of samples per octave.
        name: Scale dimension and DataArray name.

    Returns:
        DataArray with one dimension named ``name``.
    """
    if s0 <= 0:
        raise ValueError("s0 must be strictly positive")
    if octaves < 0:
        raise ValueError("octaves must be non-negative")
    if voices_per_octave <= 0:
        raise ValueError("voices_per_octave must be positive")
    n = int(np.floor(octaves * voices_per_octave)) + 1
    values = s0 * 2.0 ** (np.arange(n, dtype=float) / voices_per_octave)
    return xr.DataArray(values, dims=(name,), coords={name: values}, name=name)


def scale_to_wavenumber(
    scales: xr.DataArray | np.ndarray,
    *,
    x0: float,
    k0: float = 1.0,
) -> xr.DataArray | np.ndarray:
    """Convert Morlet scales to central wavenumber ``k = k0 / (s x0)``."""
    if x0 <= 0:
        raise ValueError("x0 must be strictly positive")
    if k0 <= 0:
        raise ValueError("k0 must be strictly positive")
    return k0 / (scales * x0)


def wavenumber_to_scale(
    wavenumber: xr.DataArray | np.ndarray,
    *,
    x0: float,
    k0: float = 1.0,
) -> xr.DataArray | np.ndarray:
    """Convert central wavenumber to Morlet scale."""
    if x0 <= 0:
        raise ValueError("x0 must be strictly positive")
    if k0 <= 0:
        raise ValueError("k0 must be strictly positive")
    return k0 / (wavenumber * x0)


def build_coi_mask(
    da: xr.DataArray,
    scales: xr.DataArray,
    *,
    dim: tuple[str, str] = ("y", "x"),
    x0: float,
) -> xr.DataArray:
    """Build a cone-of-influence mask for boundaries and NaN land cells.

    A sample is trustworthy when its distance to the nearest boundary or
    invalid cell exceeds ``scale * x0`` in coordinate units.
    """
    ydim, xdim = dim
    _require_dims(da, dim)
    dy = _coord_spacing(da, ydim)
    dx = _coord_spacing(da, xdim)
    arr = np.asarray(da.transpose(ydim, xdim).values, dtype=float)
    valid = np.isfinite(arr)

    yy, xx = np.indices(valid.shape, dtype=float)
    boundary_distance = np.minimum.reduce(
        [
            yy * dy,
            (valid.shape[0] - 1 - yy) * dy,
            xx * dx,
            (valid.shape[1] - 1 - xx) * dx,
        ]
    )

    if valid.all():
        invalid_distance = np.full(valid.shape, np.inf, dtype=float)
    else:
        invalid_distance = distance_transform_edt(valid, sampling=(dy, dx))
    distance = np.minimum(boundary_distance, invalid_distance)

    scale_values = _scale_values(scales)
    mask = valid[None, :, :] & (distance[None, :, :] > scale_values[:, None, None] * x0)
    scale_dim = scales.dims[0]
    return xr.DataArray(
        mask,
        dims=(scale_dim, ydim, xdim),
        coords={
            scale_dim: scales,
            ydim: da[ydim],
            xdim: da[xdim],
        },
        name="coi_mask",
    )


def _coord_spacing(da: xr.DataArray, dim: str) -> float:
    """Validate a coordinate is uniform and return its absolute spacing."""
    if dim not in da.coords:
        raise ValueError(
            f"dim {dim!r} has no coordinate; wavelet spectra require "
            "uniform locally-Cartesian coordinates."
        )
    values = np.asarray(da[dim].values, dtype=float)
    if values.size < 2:
        raise ValueError(f"dim {dim!r} must contain at least two samples")
    diffs = np.diff(values)
    if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-9):
        raise ValueError(
            f"coord {dim!r} is not uniformly spaced; reproject/resample onto "
            "a regular Cartesian grid before calling cwt2."
        )
    return float(abs(diffs[0]))


def _scale_values(scales: xr.DataArray) -> np.ndarray:
    """Return scale values as a one-dimensional positive float array.

    Downstream code uses ``np.log(scales)`` and ``np.gradient`` over
    the scale axis, so non-finite or non-monotone scales would
    silently produce invalid weights / interpolations — refuse them
    here with a clear error.
    """
    values = np.asarray(scales.values, dtype=float)
    if values.ndim != 1:
        raise ValueError("scales must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("scales must be finite (no NaN or inf).")
    if np.any(values <= 0):
        raise ValueError("scales must be strictly positive")
    if values.size > 1 and not np.all(np.diff(values) > 0):
        raise ValueError("scales must be strictly increasing.")
    return values


def _require_dims(da: xr.DataArray, dim: tuple[str, str]) -> None:
    """Validate that all required dimensions are present."""
    missing = [d for d in dim if d not in da.dims]
    if missing:
        raise ValueError(f"DataArray is missing wavelet dims {missing!r}")
