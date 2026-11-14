"""Extreme-value helpers — block maxima/minima, peaks-over-threshold, point processes.

This family holds Layer-0 functions only; it has no ``Operator`` wrappers.
"""

from __future__ import annotations

from xrops.extremes._src.extremes import (
    block_maxima,
    block_minima,
    pot_exceedances,
    pot_threshold,
    pp_counts,
    pp_stats,
)


__all__ = [
    "block_maxima",
    "block_minima",
    "pot_exceedances",
    "pot_threshold",
    "pp_counts",
    "pp_stats",
]
