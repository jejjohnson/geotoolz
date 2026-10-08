"""`geopatcher.spatial.aggregation` — How patch outputs merge back into a field.

Overlap-add reconstruction (`OverlapAdd`), reductions (`Sum`, `Mean`,
`Max`, `Min`, `Median`, `Mode`, `Variance`, `MeanStd`, `MinMax`,
`WeightedSum`, `InvVarWeightedMean`), voting (`HardVote`, `SoftVote`),
`ByIndex`, `Learned`, and streaming sketches (`ApproxQuantile`,
`ApproxCardinality`, `ApproxMode`, `StreamingHistogram`, `Reservoir`);
`Aggregation` is the base.
"""

from __future__ import annotations

from geopatcher._src.spatial.aggregation import (
    Aggregation,
    ApproxCardinality,
    ApproxMode,
    ApproxQuantile,
    ByIndex,
    HardVote,
    InvVarWeightedMean,
    Learned,
    Max,
    Mean,
    MeanStd,
    Median,
    Min,
    MinMax,
    Mode,
    OverlapAdd,
    Reservoir,
    SoftVote,
    StreamingHistogram,
    Sum,
    Variance,
    WeightedSum,
)


__all__ = [
    "Aggregation",
    "ApproxCardinality",
    "ApproxMode",
    "ApproxQuantile",
    "ByIndex",
    "HardVote",
    "InvVarWeightedMean",
    "Learned",
    "Max",
    "Mean",
    "MeanStd",
    "Median",
    "Min",
    "MinMax",
    "Mode",
    "OverlapAdd",
    "Reservoir",
    "SoftVote",
    "StreamingHistogram",
    "Sum",
    "Variance",
    "WeightedSum",
]
