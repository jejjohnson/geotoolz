"""Object-verification metric class names — public re-export.

The classes below are name reservations only (D14: spelled-out canonical
names). Implementations land with V5 (Epic V5); calling any of them today
raises ``NotImplementedError``.
"""

from xr_toolz.metrics._src.object import (
    CentroidDistance,
    CriticalSuccessIndex,
    DurationError,
    FalseAlarmRatio,
    IntensityBias,
    IntersectionOverUnion,
    ProbabilityOfDetection,
)


__all__ = [
    "CentroidDistance",
    "CriticalSuccessIndex",
    "DurationError",
    "FalseAlarmRatio",
    "IntensityBias",
    "IntersectionOverUnion",
    "ProbabilityOfDetection",
]
