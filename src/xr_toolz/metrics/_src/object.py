"""Object-verification metric class-name reservations (V5 surface).

This module is *importable today* and reserves the canonical long-form
class names per D14 (no short-form ``POD`` / ``FAR`` / ``CSI`` / ``IoU``
aliases on the public surface). The classes carry no implementation
yet: each ``__init__`` raises ``NotImplementedError`` so accidental use
produces a clear failure pointing at V5 (Epic).

V5 (Epic) lands the real implementations, conditional moments, and
detector / matcher integration. Until then, the public name surface is
stable so downstream code can ``import`` it without churning when V5
ships.
"""

from __future__ import annotations

from typing import Any

from xr_toolz.core import Operator


class _ObjectMetricStub(Operator):
    """Common ``NotImplementedError`` shell for V5 stubs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} is a name reservation; "
            "implementation lands with V5 (Epic V5)."
        )


class ProbabilityOfDetection(_ObjectMetricStub):
    """Hits / (hits + misses). Implementation pending V5."""


class FalseAlarmRatio(_ObjectMetricStub):
    """False alarms / (hits + false alarms). Implementation pending V5."""


class CriticalSuccessIndex(_ObjectMetricStub):
    """Threat score: ``hits / (hits + misses + false alarms)``. V5 stub."""


class IntersectionOverUnion(_ObjectMetricStub):
    """Object overlap: |A ∩ B| / |A ∪ B|. Implementation pending V5."""


class DurationError(_ObjectMetricStub):
    """Bias in matched-event duration. Implementation pending V5."""


class IntensityBias(_ObjectMetricStub):
    """Bias in matched-event intensity. Implementation pending V5."""


class CentroidDistance(_ObjectMetricStub):
    """Centroid displacement between matched objects. Implementation pending V5."""


__all__ = [
    "CentroidDistance",
    "CriticalSuccessIndex",
    "DurationError",
    "FalseAlarmRatio",
    "IntensityBias",
    "IntersectionOverUnion",
    "ProbabilityOfDetection",
]
