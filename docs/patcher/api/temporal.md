# Temporal axes

The four temporal axes composed by `TemporalPatcher`, plus the
coordinate-aware `TimeStencil` machinery.

## Temporal axes

### Geometry

::: geopatcher.TemporalGeometry
::: geopatcher.TemporalFixedLookback
::: geopatcher.TemporalLookbackHorizon
::: geopatcher.TemporalMultiScale
::: geopatcher.TemporalPhaseWindow

### Sampler

::: geopatcher.TemporalSampler
::: geopatcher.TemporalRegularStride
::: geopatcher.TemporalRandom
::: geopatcher.TemporalExplicit

`TemporalCausalRolling` is an alias of `TemporalRegularStride` and
`TemporalEventTriggered` an alias of `TemporalExplicit`.

### Window

::: geopatcher.TemporalWindow
::: geopatcher.TemporalCausalBoxcar
::: geopatcher.TemporalExponentialDecay
::: geopatcher.TemporalTaperedTukey
::: geopatcher.TemporalPeriodic

### Aggregation

::: geopatcher.TemporalAggregation
::: geopatcher.TemporalFold
::: geopatcher.TemporalMean
::: geopatcher.TemporalHierarchicalCombine
::: geopatcher.TemporalForecast

## Temporal stencils

Coordinate-aware time windows (see ADR-004 and the
[temporal stencils recipe](../recipes/temporal-stencils.md)). `Closed`
is the `Literal["left", "right", "both", "neither"]` alias used by the
stencil endpoints.

::: geopatcher.Stencil
::: geopatcher.TimeStencil
::: geopatcher.build_sampling_slices
::: geopatcher.time.coord_step
::: geopatcher.time.stencil_offsets
::: geopatcher.divide_evenly
::: geopatcher.valid_origin_points

The four-axis integration points are documented with the other temporal
axes above:

::: geopatcher.TemporalStencilGeometry
::: geopatcher.TemporalStencilSampler
