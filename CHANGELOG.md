# Changelog

## [0.1.0](https://github.com/jejjohnson/geotoolz/compare/v0.0.6...v0.1.0) (2026-06-10)


### ⚠ BREAKING CHANGES

* The composition core (`Operator`, `Sequential`, `Graph`, `Input`, `Node`, `Branch`, `Switch`, `Fanout`, `Identity`, `Const`, `Lambda`, `Sink`, `Tap`, `Snapshot`, `ShapeTrace`, `Carrier`) is now re-exported from the carrier-agnostic `pipekit` framework rather than implemented inside `geotoolz.core`.
* `from geotoolz import GeoSlice` and any `from geotoolz.catalog import ...` now fail. Install `geocatalog` (https://github.com/jejjohnson/geocatalog) and import from there.

### Features

* **compositing:** implement BlendMatched (mean / weighted_mean / ivw) ([#84](https://github.com/jejjohnson/geotoolz/issues/84)) ([b5a8337](https://github.com/jejjohnson/geotoolz/commit/b5a8337568a40a26b03efa2274a5345c0e3408e3))
* **compositing:** median/max-ndvi/cloud-free/bap/min-cloud reduction operators ([#64](https://github.com/jejjohnson/geotoolz/issues/64)) ([d689f72](https://github.com/jejjohnson/geotoolz/commit/d689f72a83c1923e538eaa5e296911898a052906))
* **coregister:** implement RasterToPoints + PointsToRaster ([#82](https://github.com/jejjohnson/geotoolz/issues/82)) ([46fcafb](https://github.com/jejjohnson/geotoolz/commit/46fcafb968f995293272260e60943c25e411e22f))
* **coregister:** implement RasterToRasterLike + StackMatched ([#81](https://github.com/jejjohnson/geotoolz/issues/81)) ([26cc60a](https://github.com/jejjohnson/geotoolz/commit/26cc60a2098e307f5bff4df2c4d1b163e9ce2958))
* **coregister:** point-cloud and vector-aggregation operators ([#83](https://github.com/jejjohnson/geotoolz/issues/83)) ([3faafd8](https://github.com/jejjohnson/geotoolz/commit/3faafd8311be4014538f060b273baceef9482bd2))
* cross-modality coregister operators + matched compositing ([#80](https://github.com/jejjohnson/geotoolz/issues/80)) ([19a7679](https://github.com/jejjohnson/geotoolz/commit/19a7679bd66e354c49a51ec0b2482a87c08a6cce))
* depend on pipekit for Operator / Sequential / Graph composition ([#74](https://github.com/jejjohnson/geotoolz/issues/74)) ([8558d05](https://github.com/jejjohnson/geotoolz/commit/8558d055abf6867a061366d65d5381c0da455ad2))
* **geom:** bowtie/antimeridian/parallax/segment sensor helpers ([#65](https://github.com/jejjohnson/geotoolz/issues/65)) ([621f0d0](https://github.com/jejjohnson/geotoolz/commit/621f0d0544ae270631cceba353aadbc9104529eb))
* **io:** hdf5/hdf4/netcdf-cf multi-format readers ([#63](https://github.com/jejjohnson/geotoolz/issues/63)) ([48f25b1](https://github.com/jejjohnson/geotoolz/commit/48f25b176034ba60af989f686a645c780554396d))
* **learn:** scikit-learn estimator adapter (phase-1, provisional API) ([#67](https://github.com/jejjohnson/geotoolz/issues/67)) ([8f4cea1](https://github.com/jejjohnson/geotoolz/commit/8f4cea190fcd5aadad6aab25aec2ee1f24771229))
* **matched_filter:** pure-numpy hyperspectral matched filter (path B) ([#62](https://github.com/jejjohnson/geotoolz/issues/62)) ([27e2531](https://github.com/jejjohnson/geotoolz/commit/27e253150749c7d5974a1365607967de1aed6acb))
* **measure:** skimage region properties bridge + plume regionprops upgrade ([#75](https://github.com/jejjohnson/geotoolz/issues/75)) ([517a01e](https://github.com/jejjohnson/geotoolz/commit/517a01e5cacd35acb1fb407d157763011536c869))
* **plume:** postprocessing operators from MethaneSAT segmentation paper ([#87](https://github.com/jejjohnson/geotoolz/issues/87)) ([95149da](https://github.com/jejjohnson/geotoolz/commit/95149da80c4632785402646be8497ef222f43876))
* **readers:** optional obstore byte path for SensorReader ([#86](https://github.com/jejjohnson/geotoolz/issues/86)) ([77231df](https://github.com/jejjohnson/geotoolz/commit/77231dfedee98cdd650cadb789b60f4d0d4f54d9))
* **readers:** sensor reader framework + toy_sensor reference reader ([#66](https://github.com/jejjohnson/geotoolz/issues/66)) ([0548f53](https://github.com/jejjohnson/geotoolz/commit/0548f53a57b5dcff9f2edb49c88add2a976e0b97))
* **segment:** skimage segmentation bridge ([#76](https://github.com/jejjohnson/geotoolz/issues/76)) ([08c239d](https://github.com/jejjohnson/geotoolz/commit/08c239d1bd4e96ce5158e4b5d7e3dee84d6336f3))
* **skimage:** feature extraction + CLAHE + geom registration ops ([#77](https://github.com/jejjohnson/geotoolz/issues/77)) ([e355788](https://github.com/jejjohnson/geotoolz/commit/e355788410bf7e38e98a7e9b9f283da1d5c43922))


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
