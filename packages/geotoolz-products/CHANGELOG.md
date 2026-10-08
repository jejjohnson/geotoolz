# Changelog

## 0.1.0 (2026-10-08)


### ⚠ BREAKING CHANGES

* **products:** geotoolz no longer ships `geotoolz.readers`, the top-level `geotoolz.SensorReader` / `geotoolz.readers` names, or the `[obstore]` extra (it only served the readers). Install `geotoolz-products` and import from `geoproducts`; use `geotoolz-products[obstore]` for pooled cloud reads. No compatibility shims, per the repo's removal policy.

### Features

* **products:** add geotoolz-products (geoproducts) and move the readers out of geotoolz ([#420](https://github.com/jejjohnson/geotoolz/issues/420)) ([c5e1a10](https://github.com/jejjohnson/geotoolz/commit/c5e1a1017ac410dcd08dee3e699e3a3d198b9e52))
* **products:** add the Carbon Mapper reader as geoproducts.carbonmapper ([#421](https://github.com/jejjohnson/geotoolz/issues/421)) ([256d6bf](https://github.com/jejjohnson/geotoolz/commit/256d6bfb82c04f58185a70698b6bf4d80c9f467a))
