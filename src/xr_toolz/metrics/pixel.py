"""Pointwise (pixel-level) evaluation metrics — public re-export.

Layer-0 functions: :func:`mse`, :func:`rmse`, :func:`nrmse`, :func:`mae`,
:func:`bias`, :func:`correlation`, :func:`r2_score`.

Layer-1 operators: :class:`MSE`, :class:`RMSE`, :class:`NRMSE`,
:class:`MAE`, :class:`Bias`, :class:`Correlation`, :class:`R2Score`.

Implementation lives in :mod:`xr_toolz.metrics._src.pixel`.
"""

from xr_toolz.metrics._src.pixel import (
    MAE,
    MSE,
    NRMSE,
    RMSE,
    Bias,
    Correlation,
    R2Score,
    bias,
    correlation,
    mae,
    mse,
    nrmse,
    r2_score,
    rmse,
)


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
