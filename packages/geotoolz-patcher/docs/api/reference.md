# `geopatcher` — Patcher framework API

Curated mkdocstrings reference, grouped by family. For the conceptual
walkthrough see [Patching](../patching.md). The matched multi-source
family lives on its own page ([Matched patching](matched.md)), as do
the framework bridges ([Integrations](integrations.md)).

## Carriers

::: geopatcher._src.patch.Patch
::: geopatcher._src.patch.TemporalPatch
::: geopatcher._src.patch.SpatioTemporalPatch

## Random access and stacking

::: geopatcher._src.indexed.IndexedPatchView
::: geopatcher._src.stacking.stack_patches

## Operational scale

::: geopatcher._src.journal.PatchJournal
::: geopatcher.runners.parallel_map
::: geopatcher._src.prefetch.prefetch_iterable

## Strictness and errors

::: geopatcher._src.config.get_strict
::: geopatcher._src.config.set_strict
::: geopatcher._src.exceptions.IncompleteScanConfiguration
::: geopatcher._src.spatial.patcher.PatchErrorRecord

## Protocols

::: geopatcher._src.hooks.PatcherHook
::: geopatcher._src.protocols.Field
::: geopatcher._src.protocols.AsyncField
::: geopatcher._src.protocols.Domain

## Domains

::: geopatcher._src.domains.GridDomain
::: geopatcher._src.domains.VectorDomain
::: geopatcher._src.domains.PointDomain

`RasterDomain` is the existing `GeoDataBase` protocol re-exported from
[`georeader`](https://github.com/IPL-UV/georeader) — import it as
`from geopatcher import RasterDomain`; see georeader's docs for the
protocol members.

## Field adapters

::: geopatcher._src.fields.raster.RasterField
::: geopatcher._src.fields.raster.AsyncRasterField

The remaining adapters are extras-gated; import via the public
submodule path:

```python
from geopatcher.fields import XarrayField, GeoPandasField, XvecField
from geopatcher.fields import RioXarrayField, DaskField, ObstoreCogField
```

::: geopatcher._src.fields.rio_xarray.RioXarrayField
::: geopatcher._src.fields.dask.DaskField
::: geopatcher._src.fields.obstore_cog.ObstoreCogField

## Top-level patchers

::: geopatcher._src.spatial.patcher.SpatialPatcher
::: geopatcher._src.spatial.patcher.AsyncSpatialPatcher
::: geopatcher._src.time.patcher.TemporalPatcher
::: geopatcher._src.spatial_time.SpatioTemporalPatcher

## Spatial axes

### Geometry

::: geopatcher._src.spatial.geometry.SpatialGeometry
::: geopatcher._src.spatial.geometry.SpatialRectangular
::: geopatcher._src.spatial.geometry.SpatialSphericalCap
::: geopatcher._src.spatial.geometry.SpatialKNNGraph
::: geopatcher._src.spatial.geometry.SpatialRadiusGraph
::: geopatcher._src.spatial.geometry.SpatialPolygonIntersection

### Sampler

::: geopatcher._src.spatial.sampler.SpatialSampler
::: geopatcher._src.spatial.sampler.SpatialRegularStride
::: geopatcher._src.spatial.sampler.SpatialJitteredStride
::: geopatcher._src.spatial.sampler.SpatialRandom
::: geopatcher._src.spatial.sampler.SpatialPoissonDisk
::: geopatcher._src.spatial.sampler.SpatialExplicit
::: geopatcher._src.spatial.sampler.SpatialAlongTrack

### Window

::: geopatcher._src.spatial.window.SpatialWindow
::: geopatcher._src.spatial.window.SpatialBoxcar
::: geopatcher._src.spatial.window.SpatialHann
::: geopatcher._src.spatial.window.SpatialTukey
::: geopatcher._src.spatial.window.SpatialGaussian
::: geopatcher._src.spatial.window.SpatialCustom

### Aggregation

::: geopatcher._src.spatial.aggregation.SpatialAggregation
::: geopatcher._src.spatial.aggregation.SpatialSum
::: geopatcher._src.spatial.aggregation.SpatialMean
::: geopatcher._src.spatial.aggregation.SpatialVariance
::: geopatcher._src.spatial.aggregation.SpatialOverlapAdd
::: geopatcher._src.spatial.aggregation.SpatialWeightedSum
::: geopatcher._src.spatial.aggregation.SpatialInvVarWeightedMean
::: geopatcher._src.spatial.aggregation.SpatialMax
::: geopatcher._src.spatial.aggregation.SpatialMin
::: geopatcher._src.spatial.aggregation.SpatialMeanStd
::: geopatcher._src.spatial.aggregation.SpatialMinMax
::: geopatcher._src.spatial.aggregation.SpatialHardVote
::: geopatcher._src.spatial.aggregation.SpatialSoftVote
::: geopatcher._src.spatial.aggregation.SpatialByIndex
::: geopatcher._src.spatial.aggregation.SpatialMedian
::: geopatcher._src.spatial.aggregation.SpatialMode
::: geopatcher._src.spatial.aggregation.SpatialLearned

#### Approximate (sketches)

::: geopatcher._src.spatial.aggregation.SpatialApproxQuantile
::: geopatcher._src.spatial.aggregation.SpatialApproxCardinality
::: geopatcher._src.spatial.aggregation.SpatialApproxMode
::: geopatcher._src.spatial.aggregation.SpatialStreamingHistogram
::: geopatcher._src.spatial.aggregation.SpatialReservoir

## Temporal axes

### Geometry

::: geopatcher._src.time.geometry.TemporalGeometry
::: geopatcher._src.time.geometry.TemporalFixedLookback
::: geopatcher._src.time.geometry.TemporalLookbackHorizon
::: geopatcher._src.time.geometry.TemporalMultiScale
::: geopatcher._src.time.geometry.TemporalPhaseWindow

### Sampler

::: geopatcher._src.time.sampler.TemporalSampler
::: geopatcher._src.time.sampler.TemporalRegularStride
::: geopatcher._src.time.sampler.TemporalCausalRolling
::: geopatcher._src.time.sampler.TemporalEventTriggered
::: geopatcher._src.time.sampler.TemporalRandom
::: geopatcher._src.time.sampler.TemporalExplicit

### Window

::: geopatcher._src.time.window.TemporalWindow
::: geopatcher._src.time.window.TemporalCausalBoxcar
::: geopatcher._src.time.window.TemporalExponentialDecay
::: geopatcher._src.time.window.TemporalTaperedTukey
::: geopatcher._src.time.window.TemporalPeriodic

### Aggregation

::: geopatcher._src.time.aggregation.TemporalAggregation
::: geopatcher._src.time.aggregation.TemporalFold
::: geopatcher._src.time.aggregation.TemporalMean
::: geopatcher._src.time.aggregation.TemporalHierarchicalCombine
::: geopatcher._src.time.aggregation.TemporalForecast

## Temporal stencils

Coordinate-aware time windows (see ADR-004 and the
[temporal stencils recipe](../recipes/temporal-stencils.md)). `Closed`
is the `Literal["left", "right", "both", "neither"]` alias used by the
stencil endpoints.

::: geopatcher._src.time.stencils.Stencil
::: geopatcher._src.time.stencils.TimeStencil
::: geopatcher._src.time.stencils.build_sampling_slices
::: geopatcher._src.time.stencils.divide_evenly
::: geopatcher._src.time.stencils.valid_origin_points

The four-axis integration points are documented with the other temporal
axes above:

::: geopatcher._src.time.geometry.TemporalStencilGeometry
::: geopatcher._src.time.sampler.TemporalStencilSampler
