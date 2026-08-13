"""Deprecated — moved to :mod:`xrtoolz.viz`.

This module re-exports the wavelet/PSD plotting helpers from their new
home (:mod:`xrtoolz.viz`, implementation in
:mod:`xrtoolz.viz._src.wavelet`) for one release. The re-export is lazy
via :pep:`562`: importing this module is silent, but accessing a moved
name emits a :class:`DeprecationWarning`. Schedule removal in the next
minor release.
"""

from __future__ import annotations

import warnings
from typing import Any


_DEPRECATED_NAMES = {
    "plot_dominant_period_map": "xrtoolz.viz",
    "plot_global_wavelet_spectrum": "xrtoolz.viz",
    "plot_resolved_scale_map": "xrtoolz.viz",
    "plot_scalogram": "xrtoolz.viz",
    "plot_wavelet_anisotropy": "xrtoolz.viz",
    "plot_wavelet_spectrum_1d": "xrtoolz.viz",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_NAMES:
        from importlib import import_module

        target = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"xrtoolz.geo.plot.{name} is deprecated; "
            f"import from {target} instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(import_module(target), name)
    raise AttributeError(f"module 'xrtoolz.geo.plot' has no attribute {name!r}")
