"""Tier A — array-tier entry points for :mod:`xrtoolz.interpolate`.

Per design decision D11, every arithmetic submodule grows a duck-array
``axis=`` entry point under ``<module>/array.py``. This module re-exports
the smoother kernels (``moving_average``, ``gaussian_smooth``,
``lowpass_filter``) that take :class:`numpy.ndarray` (or any duck array
convertible via :func:`numpy.asarray`) and an ``axis`` argument,
returning an array of the same shape.

These are the kernels Tier B (xarray, ``dim=``) and Tier C
(``Operator``) delegate to.
"""

from __future__ import annotations

from xrtoolz.interpolate._src.array_coord_remap import remap_axis
from xrtoolz.interpolate._src.array_smooth import (
    fir_filter,
    gaussian_smooth,
    gaussian_smooth_nd,
    lowpass_filter,
    moving_average,
)


__all__ = [
    "fir_filter",
    "gaussian_smooth",
    "gaussian_smooth_nd",
    "lowpass_filter",
    "moving_average",
    "remap_axis",
]
