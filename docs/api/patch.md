# `geotoolz.patch` — Patcher framework API

Curated mkdocstrings reference, grouped by family. For the conceptual
walkthrough see [Patching](../patching.md).

## Carriers

::: geotoolz.patch._src.patch.Patch
::: geotoolz.patch._src.patch.TemporalPatch
::: geotoolz.patch._src.patch.SpatioTemporalPatch

## Protocols

::: geotoolz.patch._src.protocols.Field
::: geotoolz.patch._src.protocols.AsyncField
::: geotoolz.patch._src.protocols.Domain

## Domains

::: geotoolz.patch._src.domains.GridDomain
::: geotoolz.patch._src.domains.VectorDomain
::: geotoolz.patch._src.domains.PointDomain

## Field adapters

::: geotoolz.patch._src.fields.raster.RasterField
::: geotoolz.patch._src.fields.raster.AsyncRasterField

The non-raster adapters are extras-gated; import via the submodule path:

```python
from geotoolz.patch.fields import XarrayField, GeoPandasField, XvecField
```

## Top-level patchers

::: geotoolz.patch._src.spatial.patcher.SpatialPatcher
::: geotoolz.patch._src.spatial.patcher.AsyncSpatialPatcher
::: geotoolz.patch._src.time.patcher.TemporalPatcher
::: geotoolz.patch._src.spatial_time.SpatioTemporalPatcher

## Spatial axes

### Geometry

::: geotoolz.patch._src.spatial.geometry.SpatialGeometry
::: geotoolz.patch._src.spatial.geometry.SpatialRectangular
::: geotoolz.patch._src.spatial.geometry.SpatialSphericalCap
::: geotoolz.patch._src.spatial.geometry.SpatialKNNGraph
::: geotoolz.patch._src.spatial.geometry.SpatialRadiusGraph
::: geotoolz.patch._src.spatial.geometry.SpatialPolygonIntersection

### Sampler

::: geotoolz.patch._src.spatial.sampler.SpatialSampler
::: geotoolz.patch._src.spatial.sampler.SpatialRegularStride
::: geotoolz.patch._src.spatial.sampler.SpatialJitteredStride
::: geotoolz.patch._src.spatial.sampler.SpatialRandom
::: geotoolz.patch._src.spatial.sampler.SpatialPoissonDisk
::: geotoolz.patch._src.spatial.sampler.SpatialExplicit

### Window

::: geotoolz.patch._src.spatial.window.SpatialWindow
::: geotoolz.patch._src.spatial.window.SpatialBoxcar
::: geotoolz.patch._src.spatial.window.SpatialHann
::: geotoolz.patch._src.spatial.window.SpatialTukey
::: geotoolz.patch._src.spatial.window.SpatialGaussian
::: geotoolz.patch._src.spatial.window.SpatialCustom

### Aggregation

::: geotoolz.patch._src.spatial.aggregation.SpatialAggregation
::: geotoolz.patch._src.spatial.aggregation.SpatialSum
::: geotoolz.patch._src.spatial.aggregation.SpatialMean
::: geotoolz.patch._src.spatial.aggregation.SpatialVariance
::: geotoolz.patch._src.spatial.aggregation.SpatialOverlapAdd
::: geotoolz.patch._src.spatial.aggregation.SpatialWeightedSum
::: geotoolz.patch._src.spatial.aggregation.SpatialInvVarWeightedMean
::: geotoolz.patch._src.spatial.aggregation.SpatialMax
::: geotoolz.patch._src.spatial.aggregation.SpatialMin
::: geotoolz.patch._src.spatial.aggregation.SpatialHardVote
::: geotoolz.patch._src.spatial.aggregation.SpatialSoftVote
::: geotoolz.patch._src.spatial.aggregation.SpatialByIndex
::: geotoolz.patch._src.spatial.aggregation.SpatialMedian
::: geotoolz.patch._src.spatial.aggregation.SpatialMode
::: geotoolz.patch._src.spatial.aggregation.SpatialLearned

#### Approximate (sketches) — v0.2 stubs

::: geotoolz.patch._src.spatial.aggregation.SpatialApproxQuantile
::: geotoolz.patch._src.spatial.aggregation.SpatialApproxCardinality
::: geotoolz.patch._src.spatial.aggregation.SpatialApproxMode
::: geotoolz.patch._src.spatial.aggregation.SpatialStreamingHistogram
::: geotoolz.patch._src.spatial.aggregation.SpatialReservoir

## Temporal axes

### Geometry

::: geotoolz.patch._src.time.geometry.TemporalGeometry
::: geotoolz.patch._src.time.geometry.TemporalFixedLookback
::: geotoolz.patch._src.time.geometry.TemporalLookbackHorizon
::: geotoolz.patch._src.time.geometry.TemporalMultiScale
::: geotoolz.patch._src.time.geometry.TemporalPhaseWindow

### Sampler

::: geotoolz.patch._src.time.sampler.TemporalSampler
::: geotoolz.patch._src.time.sampler.TemporalRegularStride
::: geotoolz.patch._src.time.sampler.TemporalCausalRolling
::: geotoolz.patch._src.time.sampler.TemporalEventTriggered
::: geotoolz.patch._src.time.sampler.TemporalRandom
::: geotoolz.patch._src.time.sampler.TemporalExplicit

### Window

::: geotoolz.patch._src.time.window.TemporalWindow
::: geotoolz.patch._src.time.window.TemporalCausalBoxcar
::: geotoolz.patch._src.time.window.TemporalExponentialDecay
::: geotoolz.patch._src.time.window.TemporalTaperedTukey
::: geotoolz.patch._src.time.window.TemporalPeriodic

### Aggregation

::: geotoolz.patch._src.time.aggregation.TemporalAggregation
::: geotoolz.patch._src.time.aggregation.TemporalFold
::: geotoolz.patch._src.time.aggregation.TemporalMean
::: geotoolz.patch._src.time.aggregation.TemporalHierarchicalCombine
::: geotoolz.patch._src.time.aggregation.TemporalForecast

## Operator wrappers

::: geotoolz.patch._src.ops.GridSampler
::: geotoolz.patch._src.ops.ApplyToChips
::: geotoolz.patch._src.ops.Stitch
