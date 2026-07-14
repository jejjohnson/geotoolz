"""Masked-metric wrappers — public re-export.

Layer-0 functions: :func:`masked_metric`.

Layer-1 operators: :class:`MaskedMetric`.

Implementation lives in :mod:`xrtoolz.metrics._src.masked`.
"""

from xrtoolz.metrics._src.masked import MaskedMetric, masked_metric


__all__ = [
    "MaskedMetric",
    "masked_metric",
]
