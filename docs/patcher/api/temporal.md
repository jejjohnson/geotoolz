# Temporal axes — `geopatcher.temporal`

The four temporal axes composed by `TemporalPatcher`, plus the
coordinate-aware `TimeStencil` machinery.

## Temporal axes

### Geometry

::: geopatcher.temporal.geometry.Geometry
::: geopatcher.temporal.geometry.FixedLookback
::: geopatcher.temporal.geometry.LookbackHorizon
::: geopatcher.temporal.geometry.MultiScale
::: geopatcher.temporal.geometry.PhaseWindow

### Sampler

::: geopatcher.temporal.sampler.Sampler
::: geopatcher.temporal.sampler.RegularStride
::: geopatcher.temporal.sampler.Random
::: geopatcher.temporal.sampler.Explicit

### Window

::: geopatcher.temporal.window.Window
::: geopatcher.temporal.window.CausalBoxcar
::: geopatcher.temporal.window.ExponentialDecay
::: geopatcher.temporal.window.TaperedTukey
::: geopatcher.temporal.window.Periodic

### Aggregation

::: geopatcher.temporal.aggregation.Aggregation
::: geopatcher.temporal.aggregation.Fold
::: geopatcher.temporal.aggregation.Mean
::: geopatcher.temporal.aggregation.HierarchicalCombine
::: geopatcher.temporal.aggregation.Forecast

## Temporal stencils

Coordinate-aware time windows (see ADR-004 and the
[temporal stencils recipe](../recipes/temporal-stencils.md)). `Closed`
is the `Literal["left", "right", "both", "neither"]` alias used by the
stencil endpoints.

::: geopatcher.temporal.stencils.Stencil
::: geopatcher.temporal.stencils.TimeStencil
::: geopatcher.temporal.stencils.build_sampling_slices
::: geopatcher.temporal.stencils.coord_step
::: geopatcher.temporal.stencils.stencil_offsets
::: geopatcher.temporal.stencils.exact_quotient
::: geopatcher.temporal.stencils.valid_origin_points

The four-axis integration points are documented with the other temporal
axes above:

::: geopatcher.temporal.geometry.StencilGeometry
::: geopatcher.temporal.sampler.StencilSampler
