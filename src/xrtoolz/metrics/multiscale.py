"""Multiscale (per-region) evaluation metrics — public re-export.

Layer-0 functions: :func:`evaluate_by_region`,
:func:`normalize_regions`.

Layer-1 operators: :class:`EvaluateByRegion`.

Implementation lives in :mod:`xrtoolz.metrics._src.multiscale`.
"""

from xrtoolz.metrics._src.multiscale import (
    EvaluateByRegion,
    evaluate_by_region,
    normalize_regions,
)


__all__ = [
    "EvaluateByRegion",
    "evaluate_by_region",
    "normalize_regions",
]
