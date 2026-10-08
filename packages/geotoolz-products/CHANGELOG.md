# Changelog

## [0.2.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-products-v0.1.1...geotoolz-products-v0.2.0) (2026-10-08)


### ⚠ BREAKING CHANGES

* **patcher:** no aliases are kept. The prefixed axis names (SpatialHann, TemporalMean, ...) and the root re-exports of axes, runners, hooks and caches are gone; geopatcher.time is geopatcher.temporal; geopatcher.objstore / geopatcher.cog become geocloud.store / geocloud.cog; geopatcher.runners / dask / jax / hooks fold into geopatcher.run and geopatcher.observe; ObstoreCogField becomes geopatcher.fields.CogField; the patcher extras obstore / obstore-cog are replaced by [cog]; and geotoolz-catalog[obstore] is removed (install geotoolz-cloud). Saved config envelopes that use the old class names must be regenerated.

### Code Refactoring

* **patcher:** organise geopatcher by task and split object storage into geotoolz-cloud ([#431](https://github.com/jejjohnson/geotoolz/issues/431)) ([a7a4d6e](https://github.com/jejjohnson/geotoolz/commit/a7a4d6e4fbec075260f8f4a9fd371a791ff13b30))

## [0.1.1](https://github.com/jejjohnson/geotoolz/compare/geotoolz-products-v0.1.0...geotoolz-products-v0.1.1) (2026-10-08)


### Features

* **products:** add the GOES-R ABI reader — L1b, L2 products, channel stacking and RGB recipes ([#426](https://github.com/jejjohnson/geotoolz/issues/426)) ([d54bbbf](https://github.com/jejjohnson/geotoolz/commit/d54bbbf616b74d0b26c26209c98e80cd4a022580))
* **products:** add the Himawari AHI reader — native HSD decoding, L2 cloud products, NOAA buckets and RGB recipes ([#430](https://github.com/jejjohnson/geotoolz/issues/430)) ([6153c98](https://github.com/jejjohnson/geotoolz/commit/6153c98f37d53b7e00b92f378bfa8d472ea327de))

## 0.1.0 (2026-10-08)


### ⚠ BREAKING CHANGES

* **products:** geotoolz no longer ships `geotoolz.readers`, the top-level `geotoolz.SensorReader` / `geotoolz.readers` names, or the `[obstore]` extra (it only served the readers). Install `geotoolz-products` and import from `geoproducts`; use `geotoolz-products[obstore]` for pooled cloud reads. No compatibility shims, per the repo's removal policy.

### Features

* **products:** add geotoolz-products (geoproducts) and move the readers out of geotoolz ([#420](https://github.com/jejjohnson/geotoolz/issues/420)) ([c5e1a10](https://github.com/jejjohnson/geotoolz/commit/c5e1a1017ac410dcd08dee3e699e3a3d198b9e52))
* **products:** add the Carbon Mapper reader as geoproducts.carbonmapper ([#421](https://github.com/jejjohnson/geotoolz/issues/421)) ([256d6bf](https://github.com/jejjohnson/geotoolz/commit/256d6bfb82c04f58185a70698b6bf4d80c9f467a))
