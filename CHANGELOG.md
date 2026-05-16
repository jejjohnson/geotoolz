# Changelog

## [0.1.0](https://github.com/jejjohnson/geotoolz/compare/v0.0.6...v0.1.0) (2026-05-16)


### ⚠ BREAKING CHANGES

* `from geotoolz import GeoSlice` and any `from geotoolz.catalog import ...` now fail. Install `geocatalog` (https://github.com/jejjohnson/geocatalog) and import from there.

### Code Refactoring

* extract catalog and GeoSlice into the geocatalog package ([#70](https://github.com/jejjohnson/geotoolz/issues/70)) ([830e842](https://github.com/jejjohnson/geotoolz/commit/830e842c48c0eb3de7672475c9fba41980d01eee))

## [0.0.6](https://github.com/jejjohnson/geotoolz/compare/v0.0.5...v0.0.6) (2026-05-16)


### Features

* **augment:** rs-safe spatial and spectral augmentations ([#43](https://github.com/jejjohnson/geotoolz/issues/43)) ([b9f1fba](https://github.com/jejjohnson/geotoolz/commit/b9f1fbaa3b9386d6a155c3e062637cc9b2a7289d))
* **geom:** geotoolz.geom operator surface ([#35](https://github.com/jejjohnson/geotoolz/issues/35)) ([fdc860a](https://github.com/jejjohnson/geotoolz/commit/fdc860ab15d002f4974af2ee49425e8bdc79b07c))
* **indices:** vegetation/water/snow/burn/mineral indices + named-band resolution ([#38](https://github.com/jejjohnson/geotoolz/issues/38)) ([1f6594b](https://github.com/jejjohnson/geotoolz/commit/1f6594b94a7198705a30a8b874fbcb06124f95f5))
* **io:** reader/writer operators for georeader IO ([#34](https://github.com/jejjohnson/geotoolz/issues/34)) ([530145f](https://github.com/jejjohnson/geotoolz/commit/530145f1e5955f3d414b1019942ab1999a125a06))
* **mask:** geometry rasterization, morphology, and boolean algebra ops ([#40](https://github.com/jejjohnson/geotoolz/issues/40)) ([bb507ef](https://github.com/jejjohnson/geotoolz/commit/bb507ef9804f6ee5ccab7f15c13d63a396fb7643))
* **normalize:** per-band min-max, z-score, robust, and fixed-stats ops ([#41](https://github.com/jejjohnson/geotoolz/issues/41)) ([4e6efad](https://github.com/jejjohnson/geotoolz/commit/4e6efad60accb8b34e2f3805e679e7b3f34dcdc4))
* **plume:** ch4/co2 retrieval, segmentation, and flux operators ([#45](https://github.com/jejjohnson/geotoolz/issues/45)) ([757e26d](https://github.com/jejjohnson/geotoolz/commit/757e26d220b0a8920189a4b37efd953007cadf72))
* **qa:** sensor-specific QA-bit decoders (Landsat, MODIS, S2) ([#39](https://github.com/jejjohnson/geotoolz/issues/39)) ([04eff6f](https://github.com/jejjohnson/geotoolz/commit/04eff6f811c066de2159b531ec4883757917b46d))
* **radiometry:** toa/boa pipeline operators (planck, dos1, srf, sza) ([#37](https://github.com/jejjohnson/geotoolz/issues/37)) ([0d57cff](https://github.com/jejjohnson/geotoolz/commit/0d57cffab9a1061b62d2ff5748e52d502d03a18f))
* **restore:** inpainting, gap-fill, despiking, and smoothing ops ([#42](https://github.com/jejjohnson/geotoolz/issues/42)) ([18524ae](https://github.com/jejjohnson/geotoolz/commit/18524aeece47c31c9bcfeab54731267299173586))
* **spectral:** band-space operators (SelectBands, BandMath, SRF, continuum removal) ([#36](https://github.com/jejjohnson/geotoolz/issues/36)) ([79504ac](https://github.com/jejjohnson/geotoolz/commit/79504acfe503b73e4a95aaa7bf51c922279357df))
* **viz:** rgb/false-color composites, stretches, and colormaps ([#44](https://github.com/jejjohnson/geotoolz/issues/44)) ([293acb5](https://github.com/jejjohnson/geotoolz/commit/293acb5cf5f7fcb733748f3de2d8b93d3b32419e))

## [0.0.5](https://github.com/jejjohnson/geotoolz/compare/v0.0.4...v0.0.5) (2026-05-15)


### Features

* **catalog:** streaming backend="duckdb" builders ([#15](https://github.com/jejjohnson/geotoolz/issues/15)) ([75fdb70](https://github.com/jejjohnson/geotoolz/commit/75fdb701423d132e8703fbc7ff1f3238c3c1d118))
* indices + radiometry + cloud operator stdlib (v0.1) ([#17](https://github.com/jejjohnson/geotoolz/issues/17)) ([48a8bed](https://github.com/jejjohnson/geotoolz/commit/48a8bed005c2eb80bc55ae94ed02fe27f8d7d5e3))

## [0.0.4](https://github.com/jejjohnson/geotoolz/compare/v0.0.3...v0.0.4) (2026-05-15)


### Features

* **catalog:** duckdb-backed GeoCatalog (Phase 2 of geodatabase) ([#14](https://github.com/jejjohnson/geotoolz/issues/14)) ([1a4a31e](https://github.com/jejjohnson/geotoolz/commit/1a4a31e6ccd7df44c93eed02eb5ab18efef74dc1))
* **catalog:** in-memory GeoCatalog + GeoSlice (Phase 1 of geodatabase) ([#12](https://github.com/jejjohnson/geotoolz/issues/12)) ([7b8a7a0](https://github.com/jejjohnson/geotoolz/commit/7b8a7a0d5a19da912b957cfa8dfcff66f165cc84))

## [0.0.3](https://github.com/jejjohnson/geotoolz/compare/v0.0.2...v0.0.3) (2026-05-15)


### Features

* **patch:** four-axis Patcher framework (geopatcher) ([#10](https://github.com/jejjohnson/geotoolz/issues/10)) ([c38e19d](https://github.com/jejjohnson/geotoolz/commit/c38e19d59ee8e0091fdf7af5b7f1de8a5432a1fa))

## [0.0.2](https://github.com/jejjohnson/geotoolz/compare/v0.0.1...v0.0.2) (2026-05-15)


### Features

* composition core — Operator, Sequential, Graph, ModelOp + v0.1 idiom library ([#8](https://github.com/jejjohnson/geotoolz/issues/8)) ([9c669ee](https://github.com/jejjohnson/geotoolz/commit/9c669eeaaab1fbd0fc3868f0a8230fe6e33bb689))

## 0.0.1 (2026-05-14)


### Features

* scaffold geotoolz package from pypackage_template ([4e02738](https://github.com/jejjohnson/geotoolz/commit/4e02738c4501baaea07395c6f794608b37123e36))
* scaffold geotoolz package from pypackage_template ([2d1d6fc](https://github.com/jejjohnson/geotoolz/commit/2d1d6fc97c55157b28be6707248acdcd8fa14783))

## Changelog

All notable changes to this project will be documented in this file.

See [Conventional Commits](https://www.conventionalcommits.org/) for commit guidelines.

## Unreleased
