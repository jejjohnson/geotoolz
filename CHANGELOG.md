# Changelog

## Unreleased

### Added

* `xr_toolz.metrics.array` — Tier A array kernels (D11) for the canonical pixel metrics: `mse`, `rmse`, `mae`, `bias`, `nrmse`, `correlation`, `r2_score`. Tier B (`xr_toolz.metrics._src.pixel`) now delegates via `xr.apply_ufunc` rather than reimplementing the math.
* `xr_toolz.transforms.array` — Tier A array kernels (D11) for the canonical Fourier entry points: `fft`, `ifft`, `power_spectrum`. Numpy-only computational core; Tier B (`xrft`-backed) is unchanged.
* `xr_toolz.calc.array` — Tier A array kernels (D11) for the canonical finite-difference primitives: `partial`, `gradient` (2nd-order central, uniform spacing). Numpy-only core complementing the `finitediffx`-backed Tier B.
* `tests/test_tier_contract.py` — three-tier contract harness (Tier A reachable, Tier B numerically agrees with Tier A, Tier C numerically agrees with Tier B) for the metrics/transforms/calc pilots.
* `xr_toolz.transforms.operators` — Tier C wrappers for the encoder primitives: `CyclicalEncode`, `FourierFeatures`, `RandomFourierFeatures`, `PositionalEncoding`, `EncodeTimeCyclical`, `EncodeTimeOrdinal`, `TimeRescale`, `TimeUnrescale` (#95).

### Removed

* Value-resampling primitives moved out of `xr_toolz.geo` into the new `xr_toolz.interpolate` package (D8/D12, Epic F3). No deprecation shim — the package is pre-1.0 and has no external users.
  * `xr_toolz.geo.{fillnan_spatial, fillnan_temporal, fillnan_rbf, resample_time, coarsen, refine, bin_2d, histogram_2d, points_to_grid, Grid, Period, SpaceTimeGrid}` → `xr_toolz.interpolate.<same-name>`
  * `xr_toolz.geo.operators.{FillNaNSpatial, FillNaNTemporal, ResampleTime}` → `xr_toolz.interpolate.operators.<same-name>`

### Added

* `xr_toolz.interpolate` — value-resampling package, sub-organized by source/target structure (`gap_fill`, `grid_to_grid`, `resample`, `binning`, `points_to_grid`) with placeholder submodules (`coord_remap`, `smooth`, `downscale`, `grid_to_points`) for upcoming work.
* `xr_toolz.interpolate.operators` — `Bin2D`, `Coarsen`, `FillNaNRBF`, `FillNaNSpatial`, `FillNaNTemporal`, `Histogram2D`, `PointsToGrid`, `Refine`, `ResampleTime`.
* `xr_toolz.inference` — Layer-1 wrappers for trained ML models (D4, Epic F4): `ModelOp` (framework-agnostic, duck-typed dispatch on a configurable method, xarray↔array marshalling, leading non-feature dims supported by flattening into a single ``(N, F)`` model call), plus `SklearnModelOp` and `JaxModelOp` thin subclasses. `sklearn`, `jax`, `equinox`, and `torch` are never imported at top level — backend imports are lazy. Not auto-exported from `xr_toolz`; opt-in via `from xr_toolz.inference import ModelOp`.

### Deprecated

* `xr_toolz.geo.{cyclical_encode, fourier_features, positional_encoding, random_fourier_features, lat_90_to_180, lat_180_to_90, lon_180_to_360, lon_360_to_180, encode_time_cyclical, encode_time_ordinal, time_rescale, time_unrescale}` — moved to `xr_toolz.transforms.encoders` (D8). The legacy paths still resolve via PEP-562 with a `DeprecationWarning` for one release; removal scheduled for the next minor.

## [0.0.2](https://github.com/jejjohnson/xr_toolz/compare/v0.0.1...v0.0.2) (2026-04-30)


### Features

* **inference:** modelop + Sklearn/Jax wrappers (Epic F4) ([#100](https://github.com/jejjohnson/xr_toolz/issues/100)) ([e00db49](https://github.com/jejjohnson/xr_toolz/commit/e00db49f9074918279fe92a1de167a15ddec8370))
* **transforms:** tier c encoder operators ([#95](https://github.com/jejjohnson/xr_toolz/issues/95)) ([#99](https://github.com/jejjohnson/xr_toolz/issues/99)) ([a28f15f](https://github.com/jejjohnson/xr_toolz/commit/a28f15f889da00400da9f0e113089f211b29375d))

## 0.0.1 (2026-04-30)


### Features

* add xr_toolz.types primitives and xr_toolz.data downloaders ([#8](https://github.com/jejjohnson/xr_toolz/issues/8)) ([06d2e7b](https://github.com/jejjohnson/xr_toolz/commit/06d2e7b7dfb64e901d14ca7b16332cd6d1ac58d0))
* **data:** aemet OpenData adapter + Station type + GeoParquet archive ([#11](https://github.com/jejjohnson/xr_toolz/issues/11)) ([b3db90f](https://github.com/jejjohnson/xr_toolz/commit/b3db90f409523d0a81fcbad992ee121d137e2c86))
* **data:** cds in-situ surface-land / surface-marine adapter + archive ([#12](https://github.com/jejjohnson/xr_toolz/issues/12)) ([fc80abb](https://github.com/jejjohnson/xr_toolz/commit/fc80abb73096b3255fb4b7fb0d36431d33e2b2ed))
* seed xr_toolz with core, geo primitives, ocn physics, and L1 operators ([#7](https://github.com/jejjohnson/xr_toolz/issues/7)) ([424ec89](https://github.com/jejjohnson/xr_toolz/commit/424ec891119f3a4cb1211d1e7284e3b5cb7577bd))
* **transforms:** xr_toolz.transforms + utils.XarrayEstimator ([#15](https://github.com/jejjohnson/xr_toolz/issues/15)) ([fd92ccd](https://github.com/jejjohnson/xr_toolz/commit/fd92ccd8fdb9fa013fec170268f06537980db0dc))
* xr_toolz.calc finite-diff primitives + ocn.kinematics refactor ([#14](https://github.com/jejjohnson/xr_toolz/issues/14)) ([a828788](https://github.com/jejjohnson/xr_toolz/commit/a82878816138195fd269b0be748b62ef9ff9fa24))

## Changelog

All notable changes to this project will be documented in this file.

See [Conventional Commits](https://www.conventionalcommits.org/) for commit guidelines.
