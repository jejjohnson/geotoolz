"""Spatial counterparts of the four-axis Patcher.

Mirror of the layout in `geopatcher._src.time`. Each spatial axis
lives in its own module; `SpatialPatcher` composes them.
"""

from __future__ import annotations

from geopatcher._src.spatial.aggregation import (
    SpatialAggregation,
    SpatialApproxCardinality,
    SpatialApproxMode,
    SpatialApproxQuantile,
    SpatialByIndex,
    SpatialHardVote,
    SpatialInvVarWeightedMean,
    SpatialLearned,
    SpatialMax,
    SpatialMean,
    SpatialMeanStd,
    SpatialMedian,
    SpatialMin,
    SpatialMinMax,
    SpatialMode,
    SpatialOverlapAdd,
    SpatialReservoir,
    SpatialSoftVote,
    SpatialStreamingHistogram,
    SpatialSum,
    SpatialVariance,
    SpatialWeightedSum,
)
from geopatcher._src.spatial.geometry import (
    SpatialGeometry,
    SpatialKNNGraph,
    SpatialPolygonIntersection,
    SpatialRadiusGraph,
    SpatialRectangular,
    SpatialSphericalCap,
)
from geopatcher._src.spatial.patcher import (
    AsyncSpatialPatcher,
    PatchErrorRecord,
    SpatialPatcher,
)
from geopatcher._src.spatial.sampler import (
    SpatialAlongTrack,
    SpatialExplicit,
    SpatialExplicitCoords,
    SpatialJitteredStride,
    SpatialPoissonDisk,
    SpatialRandom,
    SpatialRegularStride,
    SpatialSampler,
)
from geopatcher._src.spatial.window import (
    SpatialBoxcar,
    SpatialCustom,
    SpatialGaussian,
    SpatialHann,
    SpatialTukey,
    SpatialWindow,
)


__all__ = [
    "AsyncSpatialPatcher",
    "PatchErrorRecord",
    "SpatialAggregation",
    "SpatialAlongTrack",
    "SpatialApproxCardinality",
    "SpatialApproxMode",
    "SpatialApproxQuantile",
    "SpatialBoxcar",
    "SpatialByIndex",
    "SpatialCustom",
    "SpatialExplicit",
    "SpatialExplicitCoords",
    "SpatialGaussian",
    "SpatialGeometry",
    "SpatialHann",
    "SpatialHardVote",
    "SpatialInvVarWeightedMean",
    "SpatialJitteredStride",
    "SpatialKNNGraph",
    "SpatialLearned",
    "SpatialMax",
    "SpatialMean",
    "SpatialMeanStd",
    "SpatialMedian",
    "SpatialMin",
    "SpatialMinMax",
    "SpatialMode",
    "SpatialOverlapAdd",
    "SpatialPatcher",
    "SpatialPoissonDisk",
    "SpatialPolygonIntersection",
    "SpatialRadiusGraph",
    "SpatialRandom",
    "SpatialRectangular",
    "SpatialRegularStride",
    "SpatialReservoir",
    "SpatialSampler",
    "SpatialSoftVote",
    "SpatialSphericalCap",
    "SpatialStreamingHistogram",
    "SpatialSum",
    "SpatialTukey",
    "SpatialVariance",
    "SpatialWeightedSum",
    "SpatialWindow",
]
