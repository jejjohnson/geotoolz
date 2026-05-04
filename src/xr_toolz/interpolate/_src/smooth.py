"""Tier B — value-preserving smoothers on xarray Datasets (D12, F3.3).

Functions take an :class:`xr.Dataset`, a dimension name, and smoother
parameters; they return a Dataset with the same shape and coords, every
numeric data variable smoothed along ``dim``. Variables that don't
carry ``dim`` (or are non-numeric) pass through untouched.

Per D12 the smoothers are deterministic and parameter-free —
``KalmanSmoother`` is out of scope here and lives under future
``assimilate.smooth``.

Tier A array kernels live at :mod:`xr_toolz.interpolate._src.array_smooth`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from xr_toolz.interpolate._src import array_smooth as _array


def _apply_along_dim(
    ds: xr.Dataset,
    dim: str,
    fn: Callable[..., Any],
) -> xr.Dataset:
    """Apply a Tier A kernel to every numeric data variable that carries ``dim``."""
    if dim not in ds.dims:
        raise ValueError(f"dim {dim!r} not in Dataset dims {tuple(ds.dims)}")

    out_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if dim not in da.dims or not np.issubdtype(da.dtype, np.number):
            out_vars[str(name)] = da
            continue
        axis = da.get_axis_num(dim)
        smoothed = fn(da.values, axis=axis)
        out_vars[str(name)] = xr.DataArray(
            smoothed,
            dims=da.dims,
            coords=da.coords,
            attrs=dict(da.attrs),
            name=da.name,
        )
    return xr.Dataset(out_vars, coords=ds.coords, attrs=dict(ds.attrs))


def moving_average(
    ds: xr.Dataset,
    *,
    dim: str,
    window: int,
    center: bool = True,
    min_periods: int | None = None,
) -> xr.Dataset:
    """Sliding-window mean along ``dim``.

    See :func:`xr_toolz.interpolate.array.moving_average` for the Tier A
    kernel and parameter semantics.
    """

    def _fn(arr: np.ndarray, *, axis: int) -> np.ndarray:
        return _array.moving_average(
            arr,
            axis=axis,
            window=window,
            center=center,
            min_periods=min_periods,
        )

    return _apply_along_dim(ds, dim, _fn)


def gaussian_smooth(
    ds: xr.Dataset,
    *,
    dim: str,
    sigma: float,
    truncate: float = 4.0,
) -> xr.Dataset:
    """Gaussian smoothing along ``dim`` with standard deviation ``sigma``."""

    def _fn(arr: np.ndarray, *, axis: int) -> np.ndarray:
        return _array.gaussian_smooth(arr, axis=axis, sigma=sigma, truncate=truncate)

    return _apply_along_dim(ds, dim, _fn)


def lowpass_filter(
    ds: xr.Dataset,
    *,
    dim: str,
    cutoff: float | tuple[float, float] | list[float] | np.ndarray,
    order: int = 4,
    btype: str = "low",
) -> xr.Dataset:
    """Zero-phase Butterworth filter along ``dim``.

    ``cutoff`` is the normalized critical frequency (fraction of the
    Nyquist rate). For ``btype`` in ``{"bandpass", "bandstop"}`` pass a
    length-2 ``(low, high)`` sequence. See
    :func:`xr_toolz.interpolate.array.lowpass_filter`.
    """

    def _fn(arr: np.ndarray, *, axis: int) -> np.ndarray:
        return _array.lowpass_filter(
            arr, axis=axis, cutoff=cutoff, order=order, btype=btype
        )

    return _apply_along_dim(ds, dim, _fn)


__all__ = [
    "gaussian_smooth",
    "lowpass_filter",
    "moving_average",
]
