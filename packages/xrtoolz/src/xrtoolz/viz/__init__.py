"""Visualization operators for xrtoolz.

Per D10, viz operators are first-class :class:`Operator` instances
that return :class:`matplotlib.figure.Figure` (or
``(Figure, Axes)``). They are *terminal* — a non-Dataset return must
appear only as the last step of a :class:`Sequential` or as one of
the leaves of a :class:`Graph`.

Submodules:

- :mod:`xrtoolz.viz.validation` — V6 validation panels keyed to
  the V1–V5 metric outputs.

Also exported here: the wavelet/PSD plotting helpers
(``plot_resolved_scale_map``, ``plot_scalogram``, …) — plain functions
returning :class:`matplotlib.axes.Axes`, implemented in
:mod:`xrtoolz.viz._src.wavelet`.
"""

from xrtoolz.viz._src.cmaps import cmap_for
from xrtoolz.viz._src.norm import shared_norm
from xrtoolz.viz._src.projections import PRESETS, make_axes
from xrtoolz.viz._src.wavelet import (
    plot_dominant_period_map,
    plot_global_wavelet_spectrum,
    plot_resolved_scale_map,
    plot_scalogram,
    plot_wavelet_anisotropy,
    plot_wavelet_spectrum_1d,
)


__all__ = [
    "PRESETS",
    "cmap_for",
    "make_axes",
    "plot_dominant_period_map",
    "plot_global_wavelet_spectrum",
    "plot_resolved_scale_map",
    "plot_scalogram",
    "plot_wavelet_anisotropy",
    "plot_wavelet_spectrum_1d",
    "shared_norm",
]
