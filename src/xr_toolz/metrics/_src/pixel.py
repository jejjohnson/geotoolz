"""Pixel-level (pointwise) evaluation metrics.

Layer-0 functions take ``ds_pred`` (prediction) and ``ds_ref`` (reference)
Datasets, a variable name, and a list of reduction dimensions; they
return a :class:`xr.DataArray` with the remaining dimensions.

Convention: positive ``bias`` means the prediction is larger than the
reference. Correlation is Pearson's ``r``.

Per design decision D11, the pointwise math lives in the Tier A array
kernels at :mod:`xr_toolz.metrics._src.array_pixel`; the Tier B wrappers
below delegate to those kernels via :func:`xr.apply_ufunc`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from xr_toolz.core import Operator
from xr_toolz.metrics._src import array_pixel


Dims = str | Sequence[str]


def _normalize_dims(dims: Dims) -> list[str]:
    return [dims] if isinstance(dims, str) else list(dims)


def _apply_pixel_kernel(
    fn: Any,
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Run a Tier A pixel kernel on the named variable, reducing ``dims``.

    Selects ``variable`` from each Dataset, then dispatches to the array
    kernel via :func:`xr.apply_ufunc` with ``dims`` as the input core
    dimensions. The kernel sees a flattened trailing block of axes and
    reduces with ``axis=tuple(range(-len(core), 0))``.

    For dask-backed inputs, ``allow_rechunk=True`` is set so a core
    dimension that is split across multiple chunks is rechunked into a
    single chunk before the kernel runs. Without this, ``apply_ufunc``
    raises on multi-chunk core dims, which is a common case for the
    reduce dimension (e.g. ``time`` chunked monthly).

    Output dtype is promoted to at least ``float64`` so that integer
    inputs don't truncate floating-point reductions.
    """
    da_pred = ds_pred[variable]
    da_ref = ds_ref[variable]
    core = _normalize_dims(dims)
    out_dtype = np.result_type(da_pred.dtype, np.float64)
    out: xr.DataArray = xr.apply_ufunc(
        lambda p, r: fn(p, r, axis=tuple(range(-len(core), 0))),
        da_pred,
        da_ref,
        input_core_dims=[core, core],
        dask="parallelized",
        output_dtypes=[out_dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    return out


# ---------- Layer-0 (xarray) ----------------------------------------------


def mse(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Mean squared error reduced over ``dims``."""
    return _apply_pixel_kernel(array_pixel.mse, ds_pred, ds_ref, variable, dims)


def rmse(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Root mean squared error reduced over ``dims``."""
    return _apply_pixel_kernel(array_pixel.rmse, ds_pred, ds_ref, variable, dims)


def nrmse(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Normalized RMSE: ``1 - RMSE / sqrt(<ref^2>)``.

    Returns a score in ``(-inf, 1]`` where 1 means a perfect match and 0
    means the prediction is as wrong as a zero prediction.
    """
    return _apply_pixel_kernel(array_pixel.nrmse, ds_pred, ds_ref, variable, dims)


def mae(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Mean absolute error reduced over ``dims``."""
    return _apply_pixel_kernel(array_pixel.mae, ds_pred, ds_ref, variable, dims)


def bias(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Mean bias ``<pred - ref>`` reduced over ``dims``."""
    return _apply_pixel_kernel(array_pixel.bias, ds_pred, ds_ref, variable, dims)


def correlation(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Pearson correlation between prediction and reference over ``dims``."""
    return _apply_pixel_kernel(array_pixel.correlation, ds_pred, ds_ref, variable, dims)


def r2_score(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Coefficient of determination: ``1 - SS_res / SS_tot``."""
    return _apply_pixel_kernel(array_pixel.r2_score, ds_pred, ds_ref, variable, dims)


# ---------- Layer-1 (Operator wrappers) -----------------------------------


class _PixelMetricOp(Operator):
    """Base class for two-input pixel metrics."""

    _fn: Any = None

    def __init__(self, variable: str, dims: str | Sequence[str]):
        self.variable = variable
        self.dims = dims if isinstance(dims, str) else list(dims)

    def _apply(self, ds_pred, ds_ref):
        return self.__class__._fn(ds_pred, ds_ref, self.variable, self.dims)

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "dims": self.dims if isinstance(self.dims, str) else list(self.dims),
        }


class MSE(_PixelMetricOp):
    _fn = staticmethod(mse)


class RMSE(_PixelMetricOp):
    _fn = staticmethod(rmse)


class NRMSE(_PixelMetricOp):
    _fn = staticmethod(nrmse)


class MAE(_PixelMetricOp):
    _fn = staticmethod(mae)


class Bias(_PixelMetricOp):
    _fn = staticmethod(bias)


class Correlation(_PixelMetricOp):
    _fn = staticmethod(correlation)


class R2Score(_PixelMetricOp):
    _fn = staticmethod(r2_score)


__all__ = [
    "MAE",
    "MSE",
    "NRMSE",
    "RMSE",
    "Bias",
    "Correlation",
    "R2Score",
    "bias",
    "correlation",
    "mae",
    "mse",
    "nrmse",
    "r2_score",
    "rmse",
]
