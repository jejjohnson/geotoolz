# Changelog

## Unreleased

### Deprecated

* `xr_toolz.geo.{cyclical_encode, fourier_features, positional_encoding, random_fourier_features, lat_90_to_180, lat_180_to_90, lon_180_to_360, lon_360_to_180, encode_time_cyclical, encode_time_ordinal, time_rescale, time_unrescale}` — moved to `xr_toolz.transforms.encoders` (D8). The legacy paths still resolve via PEP-562 with a `DeprecationWarning` for one release; removal scheduled for the next minor.

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
