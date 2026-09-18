# Interpolation

Value resampling between grids and point clouds: regrid, coarsen / refine,
gap-fill, bin scattered observations onto a grid, smooth, and sample a grid
at arbitrary points (e.g. along a satellite track). `Grid`, `SpaceTimeGrid`,
and `Period` are the lightweight target-grid carriers.

## Regridding & resolution

Three ways to move a field between grids, by what they preserve:

- `RegridLike` — `xr.interp` onto another grid's coordinates. Smooth
  point values; does **not** preserve integrals.
- `Coarsen(conservative=True)` — integer-factor block averaging with
  cosine-latitude weights.
- `RegridConservative` — first-order conservative regrid onto an
  arbitrary rectilinear grid (`mode="mean"` for intensive fields such as
  concentrations, `mode="sum"` for extensive per-cell totals such as
  emissions). Cell bounds are inferred from the centres (midpoints, end
  cells symmetric) unless CF `bounds` are present; on the default
  spherical geometry longitude is periodic with period 360°, so a
  0–360 source regrids onto a −180–180 target directly. For an aligned
  integer factor it reproduces `Coarsen(conservative=True)`.

```python
import numpy as np
from xrtoolz.interpolate import RegridConservative

# 0.1° inventory -> analysis grid; basin totals are preserved.
regrid = RegridConservative(
    {"lat": np.arange(40.05, 45.0, 0.1), "lon": np.arange(-5.95, 0.0, 0.1)},
    mode="sum",
)
prior = regrid(inventory)
```

::: xrtoolz.interpolate.operators.RegridLike

::: xrtoolz.interpolate.operators.RegridConservative

::: xrtoolz.interpolate.operators.Coarsen

::: xrtoolz.interpolate.operators.Refine

::: xrtoolz.interpolate.operators.Upscale

::: xrtoolz.interpolate.operators.Downscale

::: xrtoolz.interpolate.operators.ResampleTime

## Gap-filling

::: xrtoolz.interpolate.operators.FillNaNSpatial

::: xrtoolz.interpolate.operators.FillNaNTemporal

::: xrtoolz.interpolate.operators.FillNaNLaplacian

::: xrtoolz.interpolate.operators.FillNaNBiharmonic

::: xrtoolz.interpolate.operators.FillNaNRBF

::: xrtoolz.interpolate.operators.FillNaNIDW

::: xrtoolz.interpolate.operators.FillNaNClimatology

## Binning & gridding scattered data

::: xrtoolz.interpolate.operators.Bin2D

::: xrtoolz.interpolate.operators.Histogram2D

::: xrtoolz.interpolate.operators.KDEToGrid

::: xrtoolz.interpolate.operators.PointsToGrid

::: xrtoolz.interpolate.operators.IDWToGrid

::: xrtoolz.interpolate.operators.IDWToPoints

## Smoothing & filtering

::: xrtoolz.interpolate.operators.GaussianSmooth

::: xrtoolz.interpolate.operators.GaussianSmoothMasked

::: xrtoolz.interpolate.operators.MovingAverage

::: xrtoolz.interpolate.operators.LowpassFilter

## Mask cleanup

::: xrtoolz.interpolate.operators.CleanMask

::: xrtoolz.interpolate.operators.MaskBinaryOpening

::: xrtoolz.interpolate.operators.MaskBinaryClosing

::: xrtoolz.interpolate.operators.MaskRemoveSmallHoles

::: xrtoolz.interpolate.operators.MaskRemoveSmallObjects

## Point sampling

::: xrtoolz.interpolate.operators.SampleAtPoints

::: xrtoolz.interpolate.operators.AlongTrack

## Vertical & axis remapping

`RemapAxis` replaces one dimension of every numeric variable by a new
axis of target values; the vertical presets pin conventional source /
target names. Two options cover terrain-following (WRF η) and hybrid
(ERA5 model-level) grids, whose source levels vary per column:

- `source_coords=` names a coordinate on the input — e.g.
  `z_agl(time, level, y, x)` — that supplies the source axis values for
  every column. Each column must be strictly monotone along the source
  dim (ascending or descending, and the orientation may differ between
  columns).
- `extrapolate="nearest"` holds each column's end value beyond its own
  range instead of the default `"nan"`. An analysis level that sits below
  the lowest model level of a column thus gets the lowest-level value.

```python
import numpy as np
from xrtoolz.interpolate import ToHeight

# ds: (time, level, y, x) model-level fields plus a 4-D ``z_agl`` coordinate.
to_height = ToHeight(
    np.array([10.0, 50.0, 100.0, 250.0]),
    source_coords="z_agl",
    extrapolate="nearest",
)
out = to_height(ds)  # ``level`` -> ``height``; variables without ``level`` pass through
```

Chunked inputs stay lazy along every dim except the source dim, which must
sit in a single chunk. With a 1-D dimension coordinate and the default
`extrapolate="nan"` the operator is unchanged from earlier releases.

::: xrtoolz.interpolate.operators.RemapAxis

::: xrtoolz.interpolate.operators.ToHeight

::: xrtoolz.interpolate.operators.ToPressureLevels

::: xrtoolz.interpolate.operators.ToSigma

::: xrtoolz.interpolate.operators.FromSigma

::: xrtoolz.interpolate.operators.ToIsopycnal

::: xrtoolz.interpolate.operators.ToPhase

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.interpolate.regrid_like

::: xrtoolz.interpolate.regrid_conservative

::: xrtoolz.interpolate.overlap_weights_1d

::: xrtoolz.interpolate.coarsen

::: xrtoolz.interpolate.coarsen_conservative

::: xrtoolz.interpolate.refine

::: xrtoolz.interpolate.resample_time

::: xrtoolz.interpolate.fillnan_spatial

::: xrtoolz.interpolate.fillnan_temporal

::: xrtoolz.interpolate.fillnan_laplacian

::: xrtoolz.interpolate.fillnan_biharmonic

::: xrtoolz.interpolate.fillnan_rbf

::: xrtoolz.interpolate.fillnan_idw

::: xrtoolz.interpolate.fillnan_climatology

::: xrtoolz.interpolate.bin_2d

::: xrtoolz.interpolate.histogram_2d

::: xrtoolz.interpolate.kde_to_grid

::: xrtoolz.interpolate.points_to_grid

::: xrtoolz.interpolate.idw_to_grid

::: xrtoolz.interpolate.idw_to_points

::: xrtoolz.interpolate.gaussian_smooth

::: xrtoolz.interpolate.gaussian_smooth_masked

::: xrtoolz.interpolate.moving_average

::: xrtoolz.interpolate.lowpass_filter

::: xrtoolz.interpolate.fir_filter

::: xrtoolz.interpolate.sample_at_points

::: xrtoolz.interpolate.along_track

::: xrtoolz.interpolate.remap_axis

::: xrtoolz.interpolate.to_phase

## Grid carriers

::: xrtoolz.interpolate.Grid

::: xrtoolz.interpolate.SpaceTimeGrid

::: xrtoolz.interpolate.Period
