"""Gap-filling primitives.

Spatial NaN filling uses :func:`scipy.interpolate.griddata` (linear,
nearest, or cubic). Temporal NaN filling delegates to xarray's native
``interpolate_na``. ``fillnan_laplacian`` performs iterative harmonic
relaxation. ``fillnan_rbf`` uses
:class:`scipy.interpolate.RBFInterpolator` for smooth, globally-aware
infilling.

These deliberately avoid heavy C++ dependencies (``pyinterp``,
``xesmf``); for ESMF-conservative regridding, use those libraries
directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from scipy.interpolate import RBFInterpolator, griddata

from xr_toolz.interpolate._src.knn import fillnan_idw


__all__ = [
    "fillnan_idw",
    "fillnan_laplacian",
    "fillnan_rbf",
    "fillnan_spatial",
    "fillnan_temporal",
]


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


def _validate_laplacian_args(
    max_iter: int,
    tol: float,
    relaxation: float,
    boundary: str,
) -> None:
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    if not 0.0 < relaxation < 2.0:
        raise ValueError(
            f"relaxation must satisfy 0 < relaxation < 2, got {relaxation}"
        )
    if boundary not in {"reflect", "wrap"}:
        raise ValueError(f"boundary must be 'reflect' or 'wrap', got {boundary!r}")


def _laplacian_neighbor_average(u: np.ndarray, boundary: str) -> np.ndarray:
    up = np.empty_like(u)
    down = np.empty_like(u)
    up[1:, :] = u[:-1, :]
    up[0, :] = u[0, :]
    down[:-1, :] = u[1:, :]
    down[-1, :] = u[-1, :]

    if boundary == "wrap":
        left = np.roll(u, 1, axis=1)
        right = np.roll(u, -1, axis=1)
    else:
        left = np.empty_like(u)
        right = np.empty_like(u)
        left[:, 1:] = u[:, :-1]
        left[:, 0] = u[:, 0]
        right[:, :-1] = u[:, 1:]
        right[:, -1] = u[:, -1]

    return 0.25 * (up + down + left + right)


def _fillnan_laplacian_slice(
    arr: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    relaxation: float,
    boundary: str,
) -> tuple[np.ndarray, int]:
    finite = np.isfinite(arr)
    if finite.all() or not finite.any():
        return arr, 0

    u = arr.astype(float, copy=True)
    missing = ~finite
    # `np.nanmean` skips NaN but not ±inf, so seed the initial guess from
    # the finite mask explicitly to avoid inf propagating into the fill.
    u[missing] = float(np.mean(u[finite]))

    rows, cols = np.indices(u.shape)
    red = missing & ((rows + cols) % 2 == 0)
    black = missing & ~red

    iterations = 0
    for step in range(1, max_iter + 1):
        iterations = step
        old = u[missing].copy() if tol > 0 else None
        for color in (red, black):
            if color.any():
                avg = _laplacian_neighbor_average(u, boundary)
                u[color] += relaxation * (avg[color] - u[color])
        if old is not None and float(np.max(np.abs(u[missing] - old))) < tol:
            break

    return u, iterations


def fillnan_laplacian(
    da: xr.DataArray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4,
    relaxation: float = 1.0,
    boundary: str = "reflect",
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """Fill NaN cells via iterative harmonic relaxation.

    Solves ∇²u = 0 on the masked region with Dirichlet boundary values
    from finite cells. Iteration stops when the maximum absolute update
    over missing cells is below ``tol`` or after ``max_iter`` iterations.
    All-NaN and all-finite slices pass through unchanged.

    Args:
        da: Input DataArray with at least ``lat`` and ``lon`` dims.
        max_iter: Maximum red/black Gauss-Seidel sweeps.
        tol: Absolute convergence tolerance for missing-cell updates.
        relaxation: SOR relaxation factor. ``1.0`` is Gauss-Seidel.
        boundary: ``"reflect"`` for Neumann edges, or ``"wrap"`` to wrap
            longitude edges while reflecting latitude edges.
        lon: Name of the longitude dimension.
        lat: Name of the latitude dimension.

    Returns:
        Same-shaped DataArray with NaNs filled slice-by-slice along any
        leading dimensions.
    """
    _validate_laplacian_args(max_iter, tol, relaxation, boundary)

    def _fill_slice(arr: np.ndarray) -> np.ndarray:
        filled, _ = _fillnan_laplacian_slice(
            arr,
            max_iter=max_iter,
            tol=tol,
            relaxation=relaxation,
            boundary=boundary,
        )
        return filled

    return xr.apply_ufunc(
        _fill_slice,
        da,
        input_core_dims=[[lat, lon]],
        output_core_dims=[[lat, lon]],
        vectorize=True,
    )


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
