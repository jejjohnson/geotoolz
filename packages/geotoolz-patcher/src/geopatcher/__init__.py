"""`geopatcher` — the four-axis Patcher framework.

Public surface re-exports:

- Carriers: `Patch`, `TemporalPatch`, `SpatioTemporalPatch`.
- Protocols: `Field`, `AsyncField`, `Domain`.
- Concrete domains: `RasterDomain`, `GridDomain`, `VectorDomain`, `PointDomain`.
- Field adapters: `RasterField`, `AsyncRasterField`. Optional adapters
  (`XarrayField`, `GeoPandasField`, `XvecField`, `RioXarrayField`,
  `DaskField`, `ObstoreCogField`) resolve lazily here and under
  `geopatcher.fields`, importing their extras on first access.
- Top-level patchers: `SpatialPatcher`, `AsyncSpatialPatcher`,
  `TemporalPatcher`, `SpatioTemporalPatcher`.
- ML / random access: `IndexedPatchView`, `PatchCache`, `stack_patches`.
- Observability: `PatcherHook` callback protocol, `PatchJournal`,
  `PatchErrorRecord`, `get_strict` / `set_strict`,
  `IncompleteScanConfiguration`.
- Spatial axes: re-exported from `geopatcher.spatial`.
- Temporal axes + stencils: re-exported from `geopatcher.time`
  (`Stencil`, `TimeStencil`, `Closed`, ...).
- Matched multi-source patching: `geopatcher.matched` (kept off the
  root namespace by ADR design).

Operator-graph wrappers (`GridSampler`, `ApplyToChips`, `Stitch`) that bridge
the patcher into the `pipekit` composition framework live in the optional
`geopatcher.integrations.pipekit` submodule, gated behind the `[pipekit]`
extra. While `pipekit` is pre-PyPI, install with ``uv sync --extra pipekit``
(or ``uv pip install`` — see the "Pre-PyPI install" section of the README).
The patcher core itself remains framework-free.
"""

from __future__ import annotations

from geopatcher import fields, spatial, time
from geopatcher._src.cache import PatchCache
from geopatcher._src.config import (
    get_strict,
    set_strict,
)
from geopatcher._src.domains import (
    GridDomain,
    PointDomain,
    RasterDomain,
    VectorDomain,
)
from geopatcher._src.exceptions import IncompleteScanConfiguration
from geopatcher._src.fields import (
    AsyncRasterField,
    RasterField,
    ReprojectingRasterField,
)
from geopatcher._src.hooks import PatcherHook
from geopatcher._src.indexed import IndexedPatchView
from geopatcher._src.journal import PatchJournal
from geopatcher._src.patch import (
    Patch,
    SpatioTemporalPatch,
    TemporalPatch,
)
from geopatcher._src.protocols import (
    AsyncField,
    Domain,
    Field,
)
from geopatcher._src.spatial import (  # re-export of all spatial concretes + bases
    AsyncSpatialPatcher,
    PatchErrorRecord,
    SpatialAggregation,
    SpatialAlongTrack,
    SpatialApproxCardinality,
    SpatialApproxMode,
    SpatialApproxQuantile,
    SpatialBoxcar,
    SpatialByIndex,
    SpatialCustom,
    SpatialExplicit,
    SpatialExplicitCoords,
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
    SpatialMeanStd,
    SpatialMedian,
    SpatialMin,
    SpatialMinMax,
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
from geopatcher._src.spatial_time import SpatioTemporalPatcher
from geopatcher._src.stacking import stack_patches
from geopatcher._src.time import (  # re-export of all temporal concretes + bases
    Closed,
    Stencil,
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
    TemporalStencilGeometry,
    TemporalStencilSampler,
    TemporalTaperedTukey,
    TemporalWindow,
    TimeStencil,
    build_sampling_slices,
    divide_evenly,
    valid_origin_points,
)


__version__ = "0.0.6"

__all__ = [
    "AsyncField",
    "AsyncRasterField",
    "AsyncSpatialPatcher",
    "Closed",
    "Domain",
    "Field",
    "GridDomain",
    "IncompleteScanConfiguration",
    "IndexedPatchView",
    "ObstoreCogField",
    "Patch",
    "PatchCache",
    "PatchErrorRecord",
    "PatchJournal",
    "PatcherHook",
    "PointDomain",
    "RasterDomain",
    "RasterField",
    "ReprojectingRasterField",
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
    "SpatioTemporalPatch",
    "SpatioTemporalPatcher",
    "Stencil",
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
    "TemporalStencilGeometry",
    "TemporalStencilSampler",
    "TemporalTaperedTukey",
    "TemporalWindow",
    "TimeStencil",
    "VectorDomain",
    "__version__",
    "build_sampling_slices",
    "divide_evenly",
    "fields",
    "get_strict",
    "set_strict",
    "spatial",
    "stack_patches",
    "time",
    "valid_origin_points",
]


# Lazy field adapters keyed off optional extras — defer to the public
# `geopatcher.fields` submodule's own lazy loader.
def __getattr__(name: str):
    """Lazy-load optional Field adapters from `geopatcher.fields`."""
    if name in {
        "XarrayField",
        "GeoPandasField",
        "XvecField",
        "RioXarrayField",
        "DaskField",
        "ObstoreCogField",
    }:
        return getattr(fields, name)
    raise AttributeError(name)
