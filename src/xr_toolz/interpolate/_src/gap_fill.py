"""Gap-filling primitives.

Spatial NaN filling uses :func:`scipy.interpolate.griddata` (linear,
nearest, or cubic). Temporal NaN filling delegates to xarray's native
``interpolate_na``. ``fillnan_rbf`` uses
:class:`scipy.interpolate.RBFInterpolator` for smooth, globally-aware
infilling.

These deliberately avoid heavy C++ dependencies (``pyinterp``,
``xesmf``); for Gauss-Seidel or ESMF-conservative regridding, use those
libraries directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from scipy.interpolate import RBFInterpolator, griddata


def fillnan_spatial(
    da: xr.DataArray,
    method: str = "linear",
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """Fill NaNs in a 2-D lon/lat field by scattered interpolation.

    Operates slice-by-slice along any leading dimensions. Uses
    :func:`scipy.interpolate.griddata` over the non-NaN support.

    Args:
        da: Input DataArray with at least ``lon`` and ``lat`` dims.
        method: ``"linear"``, ``"nearest"``, or ``"cubic"``.
        lon: Name of the longitude coordinate.
        lat: Name of the latitude coordinate.

    Returns:
        Same-shaped DataArray with NaNs filled where interpolation is
        possible; points outside the convex hull of non-NaN samples stay
        NaN (except for ``method="nearest"``, which extrapolates).
    """
    lon_vals = da[lon].values
    lat_vals = da[lat].values
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals, indexing="xy")
    targets = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    def _fill_slice(arr: np.ndarray) -> np.ndarray:
        finite = np.isfinite(arr)
        if finite.all() or not finite.any():
            return arr
        samples = np.column_stack([lon_grid[finite].ravel(), lat_grid[finite].ravel()])
        values = arr[finite].ravel()
        filled = griddata(samples, values, targets, method=method)
        out = arr.copy()
        flat = filled.reshape(arr.shape)
        nan_positions = np.isnan(arr)
        out[nan_positions] = np.where(
            np.isnan(flat[nan_positions]),
            arr[nan_positions],
            flat[nan_positions],
        )
        return out

    return xr.apply_ufunc(
        _fill_slice,
        da,
        input_core_dims=[[lat, lon]],
        output_core_dims=[[lat, lon]],
        vectorize=True,
    )


def fillnan_temporal(
    ds: xr.Dataset | xr.DataArray,
    method: str = "linear",
    time: str = "time",
    max_gap: Any = None,
) -> xr.Dataset | xr.DataArray:
    """Interpolate NaNs along the time dimension.

    Args:
        ds: Input Dataset or DataArray.
        method: Any method accepted by xarray's ``interpolate_na``
            (``"linear"``, ``"nearest"``, ``"quadratic"``, ``"cubic"``,
            ``"spline"``, etc.).
        time: Name of the time dimension.
        max_gap: Maximum time delta to interpolate across; gaps wider
            than this stay NaN.

    Returns:
        Same-shaped container with temporal NaNs interpolated.
    """
    return ds.interpolate_na(dim=time, method=method, max_gap=max_gap)


def fillnan_rbf(
    da: xr.DataArray,
    kernel: str = "thin_plate_spline",
    neighbors: int | None = 32,
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """Fill NaNs using a radial-basis-function interpolator.

    Uses :class:`scipy.interpolate.RBFInterpolator`. More expensive than
    ``fillnan_spatial`` but extrapolates smoothly and respects the
    global shape of the signal.
    """
    lon_vals = da[lon].values
    lat_vals = da[lat].values
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals, indexing="xy")

    def _fill_slice(arr: np.ndarray) -> np.ndarray:
        finite = np.isfinite(arr)
        if finite.all() or not finite.any():
            return arr
        samples = np.column_stack([lon_grid[finite].ravel(), lat_grid[finite].ravel()])
        values = arr[finite].ravel()
        rbf = RBFInterpolator(samples, values, kernel=kernel, neighbors=neighbors)
        # Only patch missing positions; leave observed values untouched.
        missing = ~finite
        missing_points = np.column_stack(
            [lon_grid[missing].ravel(), lat_grid[missing].ravel()]
        )
        out = arr.copy()
        out[missing] = rbf(missing_points)
        return out

    return xr.apply_ufunc(
        _fill_slice,
        da,
        input_core_dims=[[lat, lon]],
        output_core_dims=[[lat, lon]],
        vectorize=True,
    )
