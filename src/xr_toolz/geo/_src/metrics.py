"""Deprecated — moved to :mod:`xr_toolz.metrics`.

This module re-exports the pixel and spectral metrics from their new
home (:mod:`xr_toolz.metrics.pixel` and :mod:`xr_toolz.metrics.spectral`)
for one release. The re-export is lazy via :pep:`562`: importing this
module is silent, but accessing a moved name emits a
:class:`DeprecationWarning`. Schedule removal in the next minor release.
"""

from __future__ import annotations

import warnings
from typing import Any


_DEPRECATED_NAMES = {
    "bias": "xr_toolz.metrics.pixel",
    "correlation": "xr_toolz.metrics.pixel",
    "mae": "xr_toolz.metrics.pixel",
    "mse": "xr_toolz.metrics.pixel",
    "nrmse": "xr_toolz.metrics.pixel",
    "r2_score": "xr_toolz.metrics.pixel",
    "rmse": "xr_toolz.metrics.pixel",
    "find_intercept_1D": "xr_toolz.metrics.spectral",
    "psd_error": "xr_toolz.metrics.spectral",
    "psd_score": "xr_toolz.metrics.spectral",
    "resolved_scale": "xr_toolz.metrics.spectral",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_NAMES:
        from importlib import import_module

        target = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"xr_toolz.geo._src.metrics.{name} is deprecated; "
            f"import from {target} instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(import_module(target), name)
    raise AttributeError(
        f"module 'xr_toolz.geo._src.metrics' has no attribute {name!r}"
    )
