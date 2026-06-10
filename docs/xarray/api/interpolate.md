# Interpolation

Value resampling between grids and point clouds: regrid, coarsen / refine,
gap-fill, bin scattered observations onto a grid, smooth, and sample a grid
at arbitrary points (e.g. along a satellite track). `Grid`, `SpaceTimeGrid`,
and `Period` are the lightweight target-grid carriers.

## Regridding & resolution

::: xrtoolz.interpolate.operators.RegridLike

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

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.interpolate.regrid_like

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

## Grid carriers

::: xrtoolz.interpolate.Grid

::: xrtoolz.interpolate.SpaceTimeGrid

::: xrtoolz.interpolate.Period
