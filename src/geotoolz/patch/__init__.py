"""`geotoolz.patch` — the four-axis Patcher framework (geopatcher).

Public surface re-exports:

- Carriers: `Patch`, `TemporalPatch`, `SpatioTemporalPatch`.
- Protocols: `Field`, `AsyncField`, `Domain`.
- Concrete domains: `RasterDomain`, `GridDomain`, `VectorDomain`, `PointDomain`.
- Field adapters: `RasterField`, `AsyncRasterField`. Non-raster adapters
  (`XarrayField`, `GeoPandasField`, `XvecField`, `RioXarrayField`) live
  under `geotoolz.patch.fields` and lazy-import their optional extras.
- Top-level patchers: `SpatialPatcher`, `AsyncSpatialPatcher`,
  `TemporalPatcher`, `SpatioTemporalPatcher`.
- Spatial axes: re-exported from `geotoolz.patch.spatial`.
- Temporal axes: re-exported from `geotoolz.patch.time`.
- Operator wrappers (`GridSampler`, `ApplyToChips`, `Stitch`).
"""

from __future__ import annotations

from geotoolz.patch._src import spatial, time
from geotoolz.patch._src.domains import (
    GridDomain,
    PointDomain,
    RasterDomain,
    VectorDomain,
)
from geotoolz.patch._src.fields import (
    AsyncRasterField,
    RasterField,
)
from geotoolz.patch._src.ops import (
    ApplyToChips,
    GridSampler,
    Stitch,
)
from geotoolz.patch._src.patch import (
    Patch,
    SpatioTemporalPatch,
    TemporalPatch,
)
from geotoolz.patch._src.protocols import (
    AsyncField,
    Domain,
    Field,
)
from geotoolz.patch._src.spatial import (  # re-export of all spatial concretes + bases
    AsyncSpatialPatcher,
    SpatialAggregation,
    SpatialApproxCardinality,
    SpatialApproxMode,
    SpatialApproxQuantile,
    SpatialBoxcar,
    SpatialByIndex,
    SpatialCustom,
    SpatialExplicit,
    SpatialGaussian,
    SpatialGeometry,
    SpatialHann,
    SpatialHardVote,
    SpatialInvVarWeightedMean,
    SpatialJitteredStride,
    SpatialKNNGraph,
    SpatialLearned,
    SpatialMax,
    SpatialMean,
    SpatialMedian,
    SpatialMin,
    SpatialMode,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialPoissonDisk,
    SpatialPolygonIntersection,
    SpatialRadiusGraph,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
    SpatialReservoir,
    SpatialSampler,
    SpatialSoftVote,
    SpatialSphericalCap,
    SpatialStreamingHistogram,
    SpatialSum,
    SpatialTukey,
    SpatialVariance,
    SpatialWeightedSum,
    SpatialWindow,
)
from geotoolz.patch._src.spatial_time import SpatioTemporalPatcher
from geotoolz.patch._src.time import (  # re-export of all temporal concretes + bases
    TemporalAggregation,
    TemporalCausalBoxcar,
    TemporalCausalRolling,
    TemporalEventTriggered,
    TemporalExplicit,
    TemporalExponentialDecay,
    TemporalFixedLookback,
    TemporalFold,
    TemporalForecast,
    TemporalGeometry,
    TemporalHierarchicalCombine,
    TemporalLookbackHorizon,
    TemporalMean,
    TemporalMultiScale,
    TemporalPatcher,
    TemporalPeriodic,
    TemporalPhaseWindow,
    TemporalRandom,
    TemporalRegularStride,
    TemporalSampler,
    TemporalTaperedTukey,
    TemporalWindow,
)


__all__ = [
    "ApplyToChips",
    "AsyncField",
    "AsyncRasterField",
    "AsyncSpatialPatcher",
    "Domain",
    "Field",
    "GridDomain",
    "GridSampler",
    "Patch",
    "PointDomain",
    "RasterDomain",
    "RasterField",
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
    "SpatioTemporalPatch",
    "SpatioTemporalPatcher",
    "Stitch",
    "TemporalAggregation",
    "TemporalCausalBoxcar",
    "TemporalCausalRolling",
    "TemporalEventTriggered",
    "TemporalExplicit",
    "TemporalExponentialDecay",
    "TemporalFixedLookback",
    "TemporalFold",
    "TemporalForecast",
    "TemporalGeometry",
    "TemporalHierarchicalCombine",
    "TemporalLookbackHorizon",
    "TemporalMean",
    "TemporalMultiScale",
    "TemporalPatch",
    "TemporalPatcher",
    "TemporalPeriodic",
    "TemporalPhaseWindow",
    "TemporalRandom",
    "TemporalRegularStride",
    "TemporalSampler",
    "TemporalTaperedTukey",
    "TemporalWindow",
    "VectorDomain",
    "spatial",
    "time",
]


# Lazy field adapters keyed off optional extras.
def __getattr__(name: str):
    """Lazy-load optional Field adapters from `geotoolz.patch.fields`."""
    if name in {"XarrayField", "GeoPandasField", "XvecField", "RioXarrayField"}:
        from geotoolz.patch._src import fields as _f

        return getattr(_f, name)
    raise AttributeError(name)
