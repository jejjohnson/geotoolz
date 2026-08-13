"""Structural evaluation metrics — public re-export.

Layer-0 functions: :func:`ssim`, :func:`gradient_difference`,
:func:`phase_shift_error`, :func:`centroid_displacement`.

Layer-1 operators: :class:`SSIM`, :class:`GradientDifference`,
:class:`PhaseShiftError`, :class:`CentroidDisplacement`.

Implementation lives in :mod:`xrtoolz.metrics._src.structural`.
"""

from xrtoolz.metrics._src.structural import (
    SSIM,
    CentroidDisplacement,
    GradientDifference,
    PhaseShiftError,
    centroid_displacement,
    gradient_difference,
    phase_shift_error,
    ssim,
)


__all__ = [
    "SSIM",
    "CentroidDisplacement",
    "GradientDifference",
    "PhaseShiftError",
    "centroid_displacement",
    "gradient_difference",
    "phase_shift_error",
    "ssim",
]
