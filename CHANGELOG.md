# Changelog

## Unreleased

### Added

* `xr_toolz.metrics.array` — Tier A array kernels (D11) for the canonical pixel metrics: `mse`, `rmse`, `mae`, `bias`, `nrmse`, `correlation`, `r2_score`. Tier B (`xr_toolz.metrics._src.pixel`) now delegates via `xr.apply_ufunc` rather than reimplementing the math.
* `xr_toolz.transforms.array` — Tier A array kernels (D11) for the canonical Fourier entry points: `fft`, `ifft`, `power_spectrum`. Numpy-only computational core; Tier B (`xrft`-backed) is unchanged.
* `xr_toolz.calc.array` — Tier A array kernels (D11) for the canonical finite-difference primitives: `partial`, `gradient` (2nd-order central, uniform spacing). Numpy-only core complementing the `finitediffx`-backed Tier B.
* `xr_toolz.interpolate.refine_2d` — scikit-image-backed 2-D resize for bicubic/biquintic refinement; `Refine(order=...)` opts into this path.
* `tests/test_tier_contract.py` — three-tier contract harness (Tier A reachable, Tier B numerically agrees with Tier A, Tier C numerically agrees with Tier B) for the metrics/transforms/calc pilots.
* `xr_toolz.transforms.operators` — Tier C wrappers for the encoder primitives: `CyclicalEncode`, `FourierFeatures`, `RandomFourierFeatures`, `PositionalEncoding`, `EncodeTimeCyclical`, `EncodeTimeOrdinal`, `TimeRescale`, `TimeUnrescale` (#95).
* `xr_toolz.interpolate` temporal helpers — `resample_time(..., method="interpolate")` for time upsampling and `fillnan_climatology` / `FillNaNClimatology` for climatological gap fill.

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

## [0.0.6](https://github.com/jejjohnson/xr_toolz/compare/v0.0.5...v0.0.6) (2026-05-08)


### Features

* **core:** signature protocol + sequential.summary / graph.summary ([#140](https://github.com/jejjohnson/xr_toolz/issues/140)) ([b6d1996](https://github.com/jejjohnson/xr_toolz/commit/b6d19967ffd880cdb4afaee89aab4309e60d604c))
* **geo:** add decode_cf_time, validate_time, check_dataset_coords + operators ([#169](https://github.com/jejjohnson/xr_toolz/issues/169)) ([8307ee8](https://github.com/jejjohnson/xr_toolz/commit/8307ee821380a531a7d6803c9b408d045609a28b))
* **interpolate:** add FIR filters and along-track wavelength bandpass ([#166](https://github.com/jejjohnson/xr_toolz/issues/166)) ([507355e](https://github.com/jejjohnson/xr_toolz/commit/507355e8476c2fe160b1f53f23b109aadb0c6611))
* **interpolate:** add Laplacian NaN gap fill primitive ([#170](https://github.com/jejjohnson/xr_toolz/issues/170)) ([7e298ae](https://github.com/jejjohnson/xr_toolz/commit/7e298ae0f50bd9fbc3550117c8906bbd50606518))
* **metrics:** add gap-tolerant segmented along-track PSD score ([#165](https://github.com/jejjohnson/xr_toolz/issues/165)) ([e6e82ba](https://github.com/jejjohnson/xr_toolz/commit/e6e82bad56321c4bffdada68d5a1d7bdfd50fe83))
* **metrics:** add ODC-1.4 residual binning, regional scoring, and DM test ([#168](https://github.com/jejjohnson/xr_toolz/issues/168)) ([db8e6c0](https://github.com/jejjohnson/xr_toolz/commit/db8e6c02c73b6a47f7142ef972caf8cd5b2d6427))
* **ocn:** make `lwe` optional in `calculate_ssh_alongtrack` + document MDT-regrid pattern ([#167](https://github.com/jejjohnson/xr_toolz/issues/167)) ([852e1b7](https://github.com/jejjohnson/xr_toolz/commit/852e1b7f4a89889efe10910ea5c0df0ca99dbcee))
* **transforms:** sklearn integration (SklearnOp, accessor, nan-mask) ([#139](https://github.com/jejjohnson/xr_toolz/issues/139)) ([690363f](https://github.com/jejjohnson/xr_toolz/commit/690363ff774955dd77ed4dbde161e0c0430ddd4c))

## [0.0.5](https://github.com/jejjohnson/xr_toolz/compare/v0.0.4...v0.0.5) (2026-05-05)


### Features

* foundations bundle — operators, viz utilities, metric helpers ([#131](https://github.com/jejjohnson/xr_toolz/issues/131)) ([79c24fc](https://github.com/jejjohnson/xr_toolz/commit/79c24fcb85ce463fec417f3f042fbde0dbf81bba))
* **viz:** v1.5 PSD plot panels + savefig/show on all panels ([#129](https://github.com/jejjohnson/xr_toolz/issues/129)) ([1b34e1d](https://github.com/jejjohnson/xr_toolz/commit/1b34e1d65bb099ab1bafde7a0397c2af84049cf1))

## [0.0.4](https://github.com/jejjohnson/xr_toolz/compare/v0.0.3...v0.0.4) (2026-05-05)


### Features

* **core:** operator combinators + ocean tutorial notebooks ([#115](https://github.com/jejjohnson/xr_toolz/issues/115)) ([b1e4cb3](https://github.com/jejjohnson/xr_toolz/commit/b1e4cb3eb9945f089a3a5635977a7c86a33a9656))
* **metrics:** add V1.3 FrequencyBandSkill + V1.4 demo + pipeline ops ([#112](https://github.com/jejjohnson/xr_toolz/issues/112)) ([0d6f3f3](https://github.com/jejjohnson/xr_toolz/commit/0d6f3f32f4721bd6875b0fa3843301b5b0db2ff8))

## [0.0.3](https://github.com/jejjohnson/xr_toolz/compare/v0.0.2...v0.0.3) (2026-05-04)


### Features

* **interpolate:** coord_remap submodule (F3.2) ([#110](https://github.com/jejjohnson/xr_toolz/issues/110)) ([9a5ab15](https://github.com/jejjohnson/xr_toolz/commit/9a5ab1572797145df326f0383df2ddbebb6b4aae))
* **interpolate:** smooth submodule (F3.3) ([#109](https://github.com/jejjohnson/xr_toolz/issues/109)) ([d331b4c](https://github.com/jejjohnson/xr_toolz/commit/d331b4c812d98f53f3739f279499fc0f6f419146))
* **metrics:** add v1.1 SkillByLeadTime + v1.2 EvaluateByRegion ([#103](https://github.com/jejjohnson/xr_toolz/issues/103)) ([19741d2](https://github.com/jejjohnson/xr_toolz/commit/19741d2d5b311a47b1cebf13905aba8f0321a02c))
* **metrics:** add v4 process-evaluation (physical metrics + budgets) ([#106](https://github.com/jejjohnson/xr_toolz/issues/106)) ([b46582a](https://github.com/jejjohnson/xr_toolz/commit/b46582ab85ed7a1f715d0f2bb2ba4a4ce4b3a76e))
* **metrics:** land V2 data-representation metrics (structural / probabilistic / distributional / masked) ([#104](https://github.com/jejjohnson/xr_toolz/issues/104)) ([b64ac6f](https://github.com/jejjohnson/xr_toolz/commit/b64ac6f024c8401de6a2a7e099b7ebc009d1e4da))
* **viz:** add v6 validation panels (viz.validation) ([#107](https://github.com/jejjohnson/xr_toolz/issues/107)) ([263fbdf](https://github.com/jejjohnson/xr_toolz/commit/263fbdf9bab4ba7cb6f2571aa67c02475d631a89))

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
