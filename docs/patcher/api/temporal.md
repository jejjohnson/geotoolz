# Temporal axes

The four temporal axes composed by `TemporalPatcher`, plus the
coordinate-aware `TimeStencil` machinery.

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
