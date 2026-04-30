"""Spectral evaluation metrics — public re-export.

Layer-0 functions: :func:`psd_error`, :func:`psd_score`,
:func:`resolved_scale`, :func:`find_intercept_1D`.

Layer-1 operators: :class:`PSDScore`.

Implementation lives in :mod:`xr_toolz.metrics._src.spectral`.
"""

from xr_toolz.metrics._src.spectral import (
    PSDScore,
    find_intercept_1D,
    psd_error,
    psd_score,
    resolved_scale,
)


__all__ = [
    "PSDScore",
    "find_intercept_1D",
    "psd_error",
    "psd_score",
    "resolved_scale",
]
