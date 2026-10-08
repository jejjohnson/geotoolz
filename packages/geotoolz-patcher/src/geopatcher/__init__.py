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
- Axis helpers: `config_from_fields` (dataclass axis → ``get_config``
  dict) and `geom_shape` (a fixed-size geometry's weight-array shape),
  for third-party axes such as custom `SpatialWindow` subclasses.
- Config round-trip: `axis_envelope` (any axis / stencil / patcher →
  ``{"class", "config"}``) and its inverse `from_config`.
- Observability: `PatcherHook` callback protocol, `PatchJournal`
  (+ `normalize_anchor`, its anchor-key normaliser),
  `PatchErrorRecord`, `get_strict` / `set_strict`,
  `IncompleteScanConfiguration`.
- Spatial axes: re-exported from `geopatcher.spatial`.
- Temporal axes + stencils: re-exported from `geopatcher.time`
  (`Stencil`, `TimeStencil`, `Closed`, ...).
- Matched multi-source patching: `geopatcher.matched` (kept off the
  root namespace by ADR design).
- Object-store client pool: `geopatcher.objstore` (the one process-wide
  ``obstore`` pool, shared with geotoolz and geocatalog; ``[obstore]`` extra).
- Async COG reading: `geopatcher.cog` (`AsyncCogReader` and async
  ``read_*`` mirrors of ``georeader.read`` on `ObstoreCogField`'s engine;
  ``[obstore-cog]`` extra).

Operator-graph wrappers (`GridSampler`, `ApplyToChips`, `Stitch`) that bridge
the patcher into the `pipekit` composition framework live in the optional
`geopatcher.integrations.pipekit` submodule, gated behind the `[pipekit]`
extra. While `pipekit` is pre-PyPI, install with ``uv sync --extra pipekit``
(see the "Install" section of the README).
The patcher core itself remains framework-free.
"""

from __future__ import annotations

from typing import Any

from geopatcher import fields, spatial, time
from geopatcher._src._serialize import (
    axis_envelope,
    config_from_fields,
    from_config,
)
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
    LAZY_ADAPTERS as _LAZY_ADAPTERS,
    AsyncRasterField,
    RasterField,
    ReprojectingRasterField,
)
from geopatcher._src.hooks import PatcherHook
from geopatcher._src.indexed import IndexedPatchView
from geopatcher._src.journal import PatchJournal, normalize_anchor
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
    geom_shape,
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


__version__ = "0.8.0"  # x-release-please-version

__all__ = [
    "AsyncField",
    "AsyncRasterField",
    "AsyncSpatialPatcher",
    "Closed",
    "DaskField",
    "Domain",
    "Field",
    "GeoPandasField",
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
    "RioXarrayField",
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
    "XarrayField",
    "XvecField",
    "__version__",
    "axis_envelope",
    "build_sampling_slices",
    "config_from_fields",
    "divide_evenly",
    "fields",
    "from_config",
    "geom_shape",
    "get_strict",
    "normalize_anchor",
    "set_strict",
    "spatial",
    "stack_patches",
    "time",
    "valid_origin_points",
]


# Lazy field adapters keyed off optional extras — defer to the public
# `geopatcher.fields` submodule's own lazy loader.
def __getattr__(name: str) -> Any:
    """Lazy-load optional Field adapters from `geopatcher.fields`."""
    if name in _LAZY_ADAPTERS:
        return getattr(fields, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
