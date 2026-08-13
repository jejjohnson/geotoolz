"""Deprecated — moved to :mod:`xrtoolz.metrics`.

This module re-exports the pixel and spectral metrics from their new
home (:mod:`xrtoolz.metrics.pixel` and :mod:`xrtoolz.metrics.spectral`)
for one release. The re-export is lazy via :pep:`562`: importing this
module is silent, but accessing a moved name emits a
:class:`DeprecationWarning`. Schedule removal in the next minor release.
"""

from __future__ import annotations

import warnings
from typing import Any


_DEPRECATED_NAMES = {
    "bias": "xrtoolz.metrics.pixel",
    "correlation": "xrtoolz.metrics.pixel",
    "mae": "xrtoolz.metrics.pixel",
    "mse": "xrtoolz.metrics.pixel",
    "nrmse": "xrtoolz.metrics.pixel",
    "r2_score": "xrtoolz.metrics.pixel",
    "rmse": "xrtoolz.metrics.pixel",
    "find_intercept_1D": "xrtoolz.metrics.spectral",
    "psd_error": "xrtoolz.metrics.spectral",
    "psd_score": "xrtoolz.metrics.spectral",
    "resolved_scale": "xrtoolz.metrics.spectral",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_NAMES:
        from importlib import import_module

        target = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"xrtoolz.geo._src.metrics.{name} is deprecated; "
            f"import from {target} instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(import_module(target), name)
    raise AttributeError(f"module 'xrtoolz.geo._src.metrics' has no attribute {name!r}")
