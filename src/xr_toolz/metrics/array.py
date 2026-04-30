"""Tier A — array-tier entry points for :mod:`xr_toolz.metrics`.

Per design decision D11, every arithmetic submodule grows a duck-array
``axis=`` entry point under ``<module>/array.py``. This module re-exports
the pilot pixel metrics (``mse``, ``rmse``, ``mae``, ``bias``, ``nrmse``,
``correlation``, ``r2_score``) that take :class:`numpy.ndarray` (or any
duck array convertible via :func:`numpy.asarray`) and an ``axis``
argument, returning a numpy array.

These are the kernels Tier B (xarray, ``dim=``) and Tier C (``Operator``)
delegate to.
"""

from __future__ import annotations

from xr_toolz.metrics._src.array_pixel import (
    bias,
    correlation,
    mae,
    mse,
    nrmse,
    r2_score,
    rmse,
)


__all__ = [
    "bias",
    "correlation",
    "mae",
    "mse",
    "nrmse",
    "r2_score",
    "rmse",
]
