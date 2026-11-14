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
from xrops.interpolate import RegridConservative

# 0.1° inventory -> analysis grid; basin totals are preserved.
regrid = RegridConservative(
    {"lat": np.arange(40.05, 45.0, 0.1), "lon": np.arange(-5.95, 0.0, 0.1)},
    mode="sum",
)
prior = regrid(inventory)
```

::: xrops.interpolate.operators.RegridLike

::: xrops.interpolate.operators.RegridConservative

::: xrops.interpolate.operators.Coarsen

::: xrops.interpolate.operators.Refine

::: xrops.interpolate.operators.Upscale

::: xrops.interpolate.operators.Downscale

::: xrops.interpolate.operators.ResampleTime

## Gap-filling

The Laplace, biharmonic, global RBF and inverse-distance fills run on [functax](https://github.com/jejjohnson/functax) plans, the same engine as geoops's fills, so a cube pipeline and a raster pipeline give the same answer. functax runs on JAX, so these fills need the `functax` extra (`uv add 'xrtoolz-ops[functax]'` in a project that pins functax in `[tool.uv.sources]`, or, with pip, `'xrtoolz-ops[functax]'` plus `'functax[jax,scipy] @ git+https://github.com/jejjohnson/functax.git@v0.0.6'`, since functax is not on PyPI); the base install stays without JAX, and each fill raises an `ImportError` naming the extra when it is missing.

- `FillNaNLaplacian` and `FillNaNBiharmonic` solve Δu = 0 and Δ²u = 0 in the holes with `functax.project.fill_gaps`, with mirror edges and the finite cells as boundary values. `boundary="wrap"` (a periodic longitude) still runs the NumPy red/black SOR, as functax's fill has no periodic edge yet.
- `FillNaNRBF` with `neighbors=None` and a polyharmonic kernel (`"linear"`, `"thin_plate_spline"`, `"cubic"`, `"quintic"`) uses `functax.interp.RBF`; a local interpolant (`neighbors` set, the default) stays on SciPy's `RBFInterpolator` until functax has a local RBF.
- `FillNaNIDW`, `IDWToGrid` and `IDWToPoints` use `functax.interp.InverseDistance` under `metric="euclidean"`. `metric="haversine"` stays on scikit-learn's `BallTree` (the `sklearn` extra), since functax's spherical search weighs neighbours by chord length rather than great-circle angle.

::: xrops.interpolate.operators.FillNaNSpatial

::: xrops.interpolate.operators.FillNaNTemporal

::: xrops.interpolate.operators.FillNaNLaplacian

::: xrops.interpolate.operators.FillNaNBiharmonic

::: xrops.interpolate.operators.FillNaNRBF

::: xrops.interpolate.operators.FillNaNIDW

::: xrops.interpolate.operators.FillNaNClimatology

## Binning & gridding scattered data

::: xrops.interpolate.operators.Bin2D

::: xrops.interpolate.operators.Histogram2D

::: xrops.interpolate.operators.KDEToGrid

::: xrops.interpolate.operators.PointsToGrid

::: xrops.interpolate.operators.IDWToGrid

::: xrops.interpolate.operators.IDWToPoints

## Smoothing & filtering

::: xrops.interpolate.operators.GaussianSmooth

::: xrops.interpolate.operators.GaussianSmoothMasked

::: xrops.interpolate.operators.MovingAverage

::: xrops.interpolate.operators.LowpassFilter

## Mask cleanup

::: xrops.interpolate.operators.CleanMask

::: xrops.interpolate.operators.MaskBinaryOpening

::: xrops.interpolate.operators.MaskBinaryClosing

::: xrops.interpolate.operators.MaskRemoveSmallHoles

::: xrops.interpolate.operators.MaskRemoveSmallObjects

## Point sampling

::: xrops.interpolate.operators.SampleAtPoints

::: xrops.interpolate.operators.AlongTrack

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
from xrops.interpolate import ToHeight

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

::: xrops.interpolate.operators.RemapAxis

::: xrops.interpolate.operators.ToHeight

::: xrops.interpolate.operators.ToPressureLevels

::: xrops.interpolate.operators.ToSigma

::: xrops.interpolate.operators.FromSigma

::: xrops.interpolate.operators.ToIsopycnal

::: xrops.interpolate.operators.ToPhase

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.interpolate.regrid_like

::: xrops.interpolate.regrid_conservative

::: xrops.interpolate.overlap_weights_1d

::: xrops.interpolate.coarsen

::: xrops.interpolate.coarsen_conservative

::: xrops.interpolate.refine

::: xrops.interpolate.resample_time

::: xrops.interpolate.fillnan_spatial

::: xrops.interpolate.fillnan_temporal

::: xrops.interpolate.fillnan_laplacian

::: xrops.interpolate.fillnan_biharmonic

::: xrops.interpolate.fillnan_rbf

::: xrops.interpolate.fillnan_idw

::: xrops.interpolate.fillnan_climatology

::: xrops.interpolate.bin_2d

::: xrops.interpolate.histogram_2d

::: xrops.interpolate.kde_to_grid

::: xrops.interpolate.points_to_grid

::: xrops.interpolate.idw_to_grid

::: xrops.interpolate.idw_to_points

::: xrops.interpolate.gaussian_smooth

::: xrops.interpolate.gaussian_smooth_masked

::: xrops.interpolate.moving_average

::: xrops.interpolate.lowpass_filter

::: xrops.interpolate.fir_filter

::: xrops.interpolate.sample_at_points

::: xrops.interpolate.along_track

::: xrops.interpolate.remap_axis

::: xrops.interpolate.to_phase

## Grid carriers

::: xrops.interpolate.Grid

::: xrops.interpolate.SpaceTimeGrid

::: xrops.interpolate.Period
