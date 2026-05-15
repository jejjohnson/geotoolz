"""Spatial counterparts of the four-axis Patcher.

Mirror of the layout in `geotoolz.patch._src.time`. Each spatial axis
lives in its own module; `SpatialPatcher` composes them.
"""

from __future__ import annotations

from geotoolz.patch._src.spatial.aggregation import (
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
    SpatialMedian,
    SpatialMin,
    SpatialMode,
    SpatialOverlapAdd,
    SpatialReservoir,
    SpatialSoftVote,
    SpatialStreamingHistogram,
    SpatialSum,
    SpatialVariance,
    SpatialWeightedSum,
)
from geotoolz.patch._src.spatial.geometry import (
    SpatialGeometry,
    SpatialKNNGraph,
    SpatialPolygonIntersection,
    SpatialRadiusGraph,
    SpatialRectangular,
    SpatialSphericalCap,
)
from geotoolz.patch._src.spatial.patcher import (
    AsyncSpatialPatcher,
    SpatialPatcher,
)
from geotoolz.patch._src.spatial.sampler import (
    SpatialExplicit,
    SpatialJitteredStride,
    SpatialPoissonDisk,
    SpatialRandom,
    SpatialRegularStride,
    SpatialSampler,
)
from geotoolz.patch._src.spatial.window import (
    SpatialBoxcar,
    SpatialCustom,
    SpatialGaussian,
    SpatialHann,
    SpatialTukey,
    SpatialWindow,
)


__all__ = [
    "AsyncSpatialPatcher",
    "SpatialAggregation",
    "SpatialApproxCardinality",
    "SpatialApproxMode",
    "SpatialApproxQuantile",
    "SpatialBoxcar",
    "SpatialByIndex",
    "SpatialCustom",
    "SpatialExplicit",
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
    "SpatialMedian",
    "SpatialMin",
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
