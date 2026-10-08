# Changelog

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
