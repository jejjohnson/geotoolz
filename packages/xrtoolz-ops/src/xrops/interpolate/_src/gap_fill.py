"""Gap-filling primitives.

Spatial NaN filling uses :func:`scipy.interpolate.griddata` (linear,
nearest, or cubic). Temporal NaN filling delegates to xarray's native
``interpolate_na``. ``fillnan_laplacian`` and ``fillnan_biharmonic`` solve
Δu = 0 and Δ²u = 0 in the holes with ``functax.project.fill_gaps`` (the
``[functax]`` extra); ``boundary="wrap"`` keeps the NumPy red/black SOR
until functax has a periodic fill. ``fillnan_rbf`` uses functax's ``RBF``
plan for a global polyharmonic interpolant and
:class:`scipy.interpolate.RBFInterpolator` for a local one (``neighbors``)
or any other kernel.

These deliberately avoid heavy C++ dependencies (``pyinterp``,
``xesmf``); for ESMF-conservative regridding, use those libraries
directly.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr
from jaxtyping import Float
from scipy.interpolate import RBFInterpolator, griddata

from xrops.interpolate._src._functax import (
    POLYHARMONIC_ORDERS,
    fill_gaps_slice,
    polyharmonic_rbf,
)
from xrops.interpolate._src.knn import fillnan_idw
from xrops.utils._src.finite import _finite_mask, _floating_output_dtype
from xrops.utils._src.validation import _require_dims


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
        finite = _finite_mask(arr)
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
        dask="parallelized",
        output_dtypes=[_floating_output_dtype(da)],
        dask_gufunc_kwargs={"allow_rechunk": False},
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


def fillnan_climatology(
    da: xr.DataArray,
    *,
    time: str = "time",
    group: Literal["month", "dayofyear", "season"] = "month",
    residual: Literal["zero", "linear"] = "linear",
    min_count: int = 1,
) -> xr.DataArray:
    """Fill temporal NaNs from a climatological mean.

    Args:
        da: Input DataArray with a datetime-like time dimension.
        time: Name of the time dimension.
        group: Calendar grouping used to compute the climatology.
        residual: ``"zero"`` fills missing values with climatology only;
            ``"linear"`` linearly interpolates the anomaly and adds it back.
        min_count: Minimum finite observations required for each climatology
            group.

    Returns:
        Same-shaped DataArray with fillable temporal NaNs replaced.
    """
    if time not in da.dims:
        raise ValueError(f"da must have dim {time!r}.")
    if group not in {"month", "dayofyear", "season"}:
        raise ValueError(
            f"group must be one of {{'month', 'dayofyear', 'season'}}, got {group!r}."
        )
    if residual not in {"zero", "linear"}:
        raise ValueError(f"residual must be 'zero' or 'linear', got {residual!r}.")
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}.")

    try:
        grouper = getattr(da[time].dt, group)
    except AttributeError as exc:
        raise ValueError(
            f"{time!r} coordinate must be datetime-like for climatology grouping."
        ) from exc
    # Name of the synthetic group coordinate created by xarray (e.g. "month").
    group_name = grouper.name
    grouped = da.groupby(grouper)
    climatology = grouped.mean(time, skipna=True)
    counts = grouped.count(time)
    climatology = climatology.where(counts >= min_count)
    climatology_broadcast = climatology.sel({group_name: grouper})

    if residual == "zero":
        out = da.fillna(climatology_broadcast)
    else:
        anomaly = (da - climatology_broadcast).interpolate_na(dim=time, method="linear")
        out = da.fillna(anomaly + climatology_broadcast)

    # Drop the synthetic groupby coordinate if it was not present on input.
    if group_name not in da.coords and group_name in out.coords:
        out = out.drop_vars(group_name)
    return out


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


def _laplacian_neighbor_average(
    u: Float[np.ndarray, "h w"], boundary: str
) -> Float[np.ndarray, "h w"]:
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
    """Red/black SOR for ∇²u = 0 in the holes; the ``boundary="wrap"`` path."""
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
    """Fill non-finite cells with the harmonic field: ∇²u = 0 in the holes.

    The finite cells are Dirichlet values and the edges mirror (Neumann).
    With ``boundary="reflect"`` the five-point system is solved by
    ``functax.project.fill_gaps(method="laplace")`` (conjugate gradients to
    √ε in float64, the ``[functax]`` extra), so ``max_iter``, ``tol`` and
    ``relaxation`` are validated but do not enter the solve. With
    ``boundary="wrap"`` the longitude axis is periodic, which functax's fill
    does not offer yet, so red/black SOR on NumPy runs instead: it stops
    when the largest update over the holes is below ``tol`` or after
    ``max_iter`` sweeps. All-NaN and all-finite slices pass through
    unchanged.

    Args:
        da: Input DataArray with at least ``lat`` and ``lon`` dims.
        max_iter: Maximum red/black Gauss-Seidel sweeps (``"wrap"`` only).
        tol: Absolute convergence tolerance for missing-cell updates
            (``"wrap"`` only).
        relaxation: SOR relaxation factor. ``1.0`` is Gauss-Seidel
            (``"wrap"`` only).
        boundary: ``"reflect"`` for Neumann edges, or ``"wrap"`` to wrap
            longitude edges while reflecting latitude edges.
        lon: Name of the longitude dimension.
        lat: Name of the latitude dimension.

    Returns:
        Same-shaped DataArray with NaNs filled slice-by-slice along any
        leading dimensions.

    Raises:
        ImportError: If ``boundary="reflect"`` and the ``[functax]`` extra
            is missing.
    """
    _validate_laplacian_args(max_iter, tol, relaxation, boundary)

    def _fill_slice(arr: np.ndarray) -> np.ndarray:
        if boundary == "reflect":
            finite = np.isfinite(arr)
            if finite.all() or not finite.any():
                return arr
            return fill_gaps_slice(arr, finite, "laplace", feature="fillnan_laplacian")
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
        dask="parallelized",
        output_dtypes=[_floating_output_dtype(da)],
        dask_gufunc_kwargs={"allow_rechunk": False},
    )


def fillnan_biharmonic(
    da: xr.DataArray,
    *,
    lon: str = "lon",
    lat: str = "lat",
    mask: xr.DataArray | None = None,
    split_into_regions: bool = True,
) -> xr.DataArray:
    """Fill NaNs in a 2-D lon/lat field by biharmonic inpainting: Δ²u = 0.

    Operates slice-by-slice along any leading dimensions with
    ``functax.project.fill_gaps(method="biharmonic")`` (the ``[functax]``
    extra): the masked pixels solve Δ²u = 0 with mirror edges and the other
    pixels as boundary values. The optional ``mask`` follows scikit-image
    semantics: ``True`` marks pixels to fill.

    Args:
        da: Input DataArray with at least ``lon`` and ``lat`` dims.
        lon: Name of the longitude dimension.
        lat: Name of the latitude dimension.
        mask: Optional explicit boolean mask. If ``None``, NaNs in ``da``
            are filled.
        split_into_regions: Kept for compatibility with the scikit-image
            backend; functax solves every masked region in one system, so
            it does not change the result.

    Returns:
        Same-shaped DataArray with masked pixels inpainted. Fully masked
        slices pass through unchanged.

    Raises:
        ImportError: If the ``[functax]`` extra is not installed.
        ValueError: If ``da`` or ``mask`` is missing the spatial dimensions.
    """
    _require_dims(da, lon, lat, name="da")

    if mask is None:
        mask_da = da.isnull()
    else:
        _require_dims(mask, lon, lat, name="mask")
        mask_da = mask.astype(bool)

    output_dtype = _floating_output_dtype(da)

    def _fill_slice(arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
        mask_bool = np.array(mask_arr, dtype=bool, copy=True)
        if not mask_bool.any() or mask_bool.all():
            return arr.astype(output_dtype, copy=True)

        # Extend the solver mask to cover any non-finite pixels outside the
        # caller's mask — using NaNs as zero-valued boundary conditions would
        # bias the solve. After the solve we restore those positions back to
        # their original (NaN) value so the caller's mask remains the only
        # set of cells whose values we modify.
        nonfinite_outside_mask = ~_finite_mask(arr) & ~mask_bool
        solver_mask = mask_bool | nonfinite_outside_mask
        if solver_mask.all():
            msg = "fillnan_biharmonic needs a finite pixel outside the fill mask"
            raise ValueError(msg)
        # fill_gaps never reads the values inside the solver mask.
        out = fill_gaps_slice(
            arr, ~solver_mask, "biharmonic", feature="fillnan_biharmonic"
        )
        out = np.asarray(out, dtype=output_dtype)
        # Restore everything outside the caller's mask (including unrelated
        # NaNs and inf values) to its original input value.
        out[~mask_bool] = arr[~mask_bool]
        return out

    return xr.apply_ufunc(
        _fill_slice,
        da,
        mask_da,
        input_core_dims=[[lat, lon], [lat, lon]],
        output_core_dims=[[lat, lon]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[output_dtype],
        dask_gufunc_kwargs={"allow_rechunk": False},
    )


def fillnan_rbf(
    da: xr.DataArray,
    kernel: str = "thin_plate_spline",
    neighbors: int | None = 32,
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """Fill NaNs using a radial-basis-function interpolator.

    More expensive than ``fillnan_spatial`` but extrapolates smoothly and
    respects the global shape of the signal. With ``neighbors=None`` and a
    polyharmonic ``kernel`` (``"linear"``, ``"thin_plate_spline"``,
    ``"cubic"``, ``"quintic"``) the global interpolant is functax's ``RBF``
    plan (the ``[functax]`` extra), with SciPy's default polynomial tail.
    A local interpolant (``neighbors`` set) or any other kernel uses
    :class:`scipy.interpolate.RBFInterpolator`, until functax has a local
    RBF.

    Args:
        da: Input DataArray with at least ``lon`` and ``lat`` dims.
        kernel: A :class:`scipy.interpolate.RBFInterpolator` kernel name.
        neighbors: Nearest samples per evaluation point, or ``None`` for
            the global interpolant over every finite sample.
        lon: Name of the longitude coordinate.
        lat: Name of the latitude coordinate.

    Returns:
        Same-shaped DataArray with NaNs filled; observed values untouched.

    Raises:
        ImportError: If the global polyharmonic path needs the
            ``[functax]`` extra and it is missing.
    """
    lon_vals = da[lon].values
    lat_vals = da[lat].values
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals, indexing="xy")

    def _fill_slice(arr: np.ndarray) -> np.ndarray:
        finite = _finite_mask(arr)
        if finite.all() or not finite.any():
            return arr
        samples = np.column_stack([lon_grid[finite].ravel(), lat_grid[finite].ravel()])
        values = arr[finite].ravel()
        # Only patch missing positions; leave observed values untouched.
        missing = ~finite
        missing_points = np.column_stack(
            [lon_grid[missing].ravel(), lat_grid[missing].ravel()]
        )
        out = arr.copy()
        if neighbors is None and kernel in POLYHARMONIC_ORDERS:
            out[missing] = polyharmonic_rbf(
                samples, values, missing_points, order=POLYHARMONIC_ORDERS[kernel]
            )
            return out
        rbf = RBFInterpolator(samples, values, kernel=kernel, neighbors=neighbors)
        out[missing] = rbf(missing_points)
        return out

    return xr.apply_ufunc(
        _fill_slice,
        da,
        input_core_dims=[[lat, lon]],
        output_core_dims=[[lat, lon]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[_floating_output_dtype(da)],
        dask_gufunc_kwargs={"allow_rechunk": False},
    )
