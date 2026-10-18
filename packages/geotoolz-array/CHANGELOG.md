# Changelog

## [0.2.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-array-v0.1.0...geotoolz-array-v0.2.0) (2026-10-18)


### ⚠ BREAKING CHANGES

* **ops:** geoops operators no longer accept georeader GeoTensor or GeoData; convert with geoarray.interop.georeader.from_georeader / source_from_georeader. wrap_like(fill_value_default=) is now wrap_like(fill_value=); require_geotensor is require_geoarray; GeoTensorEstimator is GeoArrayEstimator.

### Features

* **array:** add the RasterSource protocols and georeader reader adapters ([#500](https://github.com/jejjohnson/geotoolz/issues/500)) ([d4aea92](https://github.com/jejjohnson/geotoolz/commit/d4aea9287f52e8f7bdd513a373763d73a7294b1e))
* **ops:** take and return geoarray.GeoArray instead of GeoTensor ([#501](https://github.com/jejjohnson/geotoolz/issues/501)) ([7522c70](https://github.com/jejjohnson/geotoolz/commit/7522c7016f1a3c18b769d587a46540e1e205578b))

## 0.1.0 (2026-10-17)


### Features

* **array:** add geotoolz-array with the GeoArray carrier ([#496](https://github.com/jejjohnson/geotoolz/issues/496)) ([0c901c6](https://github.com/jejjohnson/geotoolz/commit/0c901c6f236d141e93ab5f7e8590066c59ec8983))
* **array:** geometry primitives for geoarray ([#497](https://github.com/jejjohnson/geotoolz/issues/497)) ([1b3e2d5](https://github.com/jejjohnson/geotoolz/commit/1b3e2d53eca02161f13994a8d0af1beffc671cb4))
* **array:** numpy/jax interop and the georeader adapter ([#499](https://github.com/jejjohnson/geotoolz/issues/499)) ([5315b5e](https://github.com/jejjohnson/geotoolz/commit/5315b5e88a779ca28a606033326d44cca8c3c820))

## Changelog
