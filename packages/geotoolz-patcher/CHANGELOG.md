# Changelog

## [0.0.6](https://github.com/jejjohnson/geopatcher/compare/v0.0.5...v0.0.6) (2026-07-11)


### Features

* **spatial_time:** thread coord= through SpatioTemporalPatcher; fix zarr sharding ([#68](https://github.com/jejjohnson/geopatcher/issues/68)) ([dcf18ea](https://github.com/jejjohnson/geopatcher/commit/dcf18ea060c6d36d42ba66636fc700fa0ce10d45))
* **spatial:** SpatialAlongTrack sampler + PointDomain nearest/bilinear sampling ([#67](https://github.com/jejjohnson/geopatcher/issues/67)) ([4ae0054](https://github.com/jejjohnson/geopatcher/commit/4ae00546b98d74e6bdc1338becf7b46d7ae080db))


### Bug Fixes

* **tests:** add missing rasterio and GeoTensor imports in test_operational_scale ([#70](https://github.com/jejjohnson/geopatcher/issues/70)) ([aabc10b](https://github.com/jejjohnson/geopatcher/commit/aabc10bd83292e4fa67decbd25b25471f9320bcd))

## [0.0.5](https://github.com/jejjohnson/geopatcher/compare/v0.0.4...v0.0.5) (2026-06-03)


### Features

* **ml:** xrpatcher → geopatcher port (indexed view + cache + check_full_scan + xarray reconstruct) ([#61](https://github.com/jejjohnson/geopatcher/issues/61)) ([4afa248](https://github.com/jejjohnson/geopatcher/commit/4afa248f25846f442c2bba3530d42bbbedbe4c1a)), closes [#60](https://github.com/jejjohnson/geopatcher/issues/60)

## [0.0.4](https://github.com/jejjohnson/geopatcher/compare/v0.0.3...v0.0.4) (2026-06-03)


### Features

* **fields:** obstore COG field + batched parallel_map via select_many duck-typing ([#53](https://github.com/jejjohnson/geopatcher/issues/53)) ([1d12ce2](https://github.com/jejjohnson/geopatcher/commit/1d12ce27c58b0d25ef05c443b1d9839bdf4a5ee2))
* **time:** coordinate-aware temporal patching via TimeStencil (closes [#56](https://github.com/jejjohnson/geopatcher/issues/56)) ([#57](https://github.com/jejjohnson/geopatcher/issues/57)) ([351050b](https://github.com/jejjohnson/geopatcher/commit/351050b40584f6f47b289b05e6c1730cf3b3d44a))

## [0.0.3](https://github.com/jejjohnson/geopatcher/compare/v0.0.2...v0.0.3) (2026-05-25)


### Features

* add pipekit Operator integration behind [pipekit] extra ([#35](https://github.com/jejjohnson/geopatcher/issues/35)) ([2fda4eb](https://github.com/jejjohnson/geopatcher/commit/2fda4eb24573a9622cd7291e5a83475980ddd224))
* **geometry:** first-class boundary policy on SpatialRectangular ([#38](https://github.com/jejjohnson/geopatcher/issues/38)) ([54eaf79](https://github.com/jejjohnson/geopatcher/commit/54eaf792b368fd65e622fecad74eca7240c173c1))
* lock foundation ADRs, add strict mode + n_anchors() ([#37](https://github.com/jejjohnson/geopatcher/issues/37)) ([f1e7dc9](https://github.com/jejjohnson/geopatcher/commit/f1e7dc9bde9eec4ca27a34c47832455401f25405))
* **matched:** composite Field + MatchedPatch carrier (ADR-003) ([#48](https://github.com/jejjohnson/geopatcher/issues/48)) ([ca989a7](https://github.com/jejjohnson/geopatcher/commit/ca989a709f63e3e7409b20e5bfcb52c5b34b29a8))
* **matched:** implement MatchedField.select + MatchedSpatialPatcher ([#49](https://github.com/jejjohnson/geopatcher/issues/49)) ([a8d4da3](https://github.com/jejjohnson/geopatcher/commit/a8d4da3ad461f0d94850cbd9a0b52d8299543712))
* **matched:** MatchedTemporalPatcher + MatchedSpatioTemporalPatcher ([#51](https://github.com/jejjohnson/geopatcher/issues/51)) ([1b5a8b7](https://github.com/jejjohnson/geopatcher/commit/1b5a8b73acce0b4860d5030bc589253eb2b79dc1))
* ml primitives and torch/jax/grain recipe notebooks ([#23](https://github.com/jejjohnson/geopatcher/issues/23)) ([#41](https://github.com/jejjohnson/geopatcher/issues/41)) ([5e4c505](https://github.com/jejjohnson/geopatcher/commit/5e4c505152878503a47595cbdb3e7f123adb5832))
* **patcher:** async + parallel patching helpers (asplit, prefetch, dask, batched) ([#42](https://github.com/jejjohnson/geopatcher/issues/42)) ([8fe56d0](https://github.com/jejjohnson/geopatcher/commit/8fe56d0e6a6110581109b5c60d81931fb9114c71))
* **patcher:** observability hooks (split/patch/merge/error) ([#43](https://github.com/jejjohnson/geopatcher/issues/43)) ([48a0294](https://github.com/jejjohnson/geopatcher/commit/48a02944cef0577bb46b1dbce5d3b845f967f07f))
* **patcher:** on_error policy (raise/skip/mask/retry) on spatial patchers ([#44](https://github.com/jejjohnson/geopatcher/issues/44)) ([0c7c9fd](https://github.com/jejjohnson/geopatcher/commit/0c7c9fd37b581f0ec0a02e84439b25400f92eb4a))
* **patcher:** operational-scale primitives (journal, in-flight bounds, sketches, cog/zarr writers) ([#45](https://github.com/jejjohnson/geopatcher/issues/45)) ([fe089b2](https://github.com/jejjohnson/geopatcher/commit/fe089b2783288bc3dd097b1d0d648b13c0850381))
* **patcher:** parallel_map + two-pass / reduce primitives ([#46](https://github.com/jejjohnson/geopatcher/issues/46)) ([29f70d8](https://github.com/jejjohnson/geopatcher/commit/29f70d872676897445d0c113163988fcc3707e61))

## [0.0.2](https://github.com/jejjohnson/geopatcher/compare/v0.0.1...v0.0.2) (2026-05-16)


### Features

* port four-axis Patcher framework from geotoolz, clean up template ([#6](https://github.com/jejjohnson/geopatcher/issues/6)) ([e6e22b3](https://github.com/jejjohnson/geopatcher/commit/e6e22b31073e09ce6359e16b562a6778c61474c2))

## Changelog

All notable changes to this project will be documented in this file.

See [Conventional Commits](https://www.conventionalcommits.org/) for commit guidelines.
