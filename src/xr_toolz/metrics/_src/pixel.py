"""Pixel-level (pointwise) evaluation metrics.

Layer-0 functions take ``ds_pred`` (prediction) and ``ds_ref`` (reference)
Datasets, a variable name, and a list of reduction dimensions; they
return a :class:`xr.DataArray` with the remaining dimensions.

Convention: positive ``bias`` means the prediction is larger than the
reference. Correlation is Pearson's ``r``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import xarray as xr

from xr_toolz.core import Operator


Dims = str | Sequence[str]


# ---------- Layer-0 (xarray) ----------------------------------------------


def mse(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Mean squared error reduced over ``dims``."""
    diff = ds_pred[variable] - ds_ref[variable]
    return (diff**2).mean(dim=dims)


def rmse(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Root mean squared error reduced over ``dims``."""
    return mse(ds_pred, ds_ref, variable, dims) ** 0.5


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
    err = rmse(ds_pred, ds_ref, variable, dims)
    scale = (ds_ref[variable] ** 2).mean(dim=dims) ** 0.5
    return 1.0 - err / scale


def mae(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Mean absolute error reduced over ``dims``."""
    return abs(ds_pred[variable] - ds_ref[variable]).mean(dim=dims)


def bias(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Mean bias ``<pred - ref>`` reduced over ``dims``."""
    return (ds_pred[variable] - ds_ref[variable]).mean(dim=dims)


def correlation(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Pearson correlation between prediction and reference over ``dims``."""
    return xr.corr(ds_pred[variable], ds_ref[variable], dim=dims)


def r2_score(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Coefficient of determination: ``1 - SS_res / SS_tot``."""
    ref = ds_ref[variable]
    ss_res = ((ref - ds_pred[variable]) ** 2).sum(dim=dims)
    ss_tot = ((ref - ref.mean(dim=dims)) ** 2).sum(dim=dims)
    return 1.0 - ss_res / ss_tot


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
