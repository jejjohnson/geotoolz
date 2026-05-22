"""Tier A — array-tier entry points for :mod:`xrtoolz.metrics`.

Per design decision D11, every arithmetic submodule grows a duck-array
``axis=`` entry point under ``<module>/array.py``. This module re-exports
the pilot pixel metrics (``mse``, ``rmse``, ``mae``, ``bias``, ``nrmse``,
``nrmse_score``, ``correlation``, ``r2_score``) that take
:class:`numpy.ndarray` (or any duck array convertible via
:func:`numpy.asarray`) and an ``axis`` argument, returning a numpy
array.

These are the kernels Tier B (xarray, ``dim=``) and Tier C (``Operator``)
delegate to.
"""

from __future__ import annotations

from xrtoolz.metrics._src.array_pixel import (
    bias,
    correlation,
    mae,
    mse,
    nrmse,
    nrmse_score,
    r2_score,
    rmse,
)
from xrtoolz.metrics._src.array_segmented_psd import (
    segment_signal,
    segmented_coherence,
    segmented_csd,
    segmented_psd,
)


__all__ = [
    "bias",
    "correlation",
    "mae",
    "mse",
    "nrmse",
    "nrmse_score",
    "r2_score",
    "rmse",
    "segment_signal",
    "segmented_coherence",
    "segmented_csd",
    "segmented_psd",
]
