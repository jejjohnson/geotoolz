# Changelog

## [0.14.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-v0.13.0...geotoolz-v0.14.0) (2026-10-27)


### ⚠ BREAKING CHANGES

* **ops:** code that installs geotoolz-ops directly and uses reprojection, file I/O, vector outputs, scikit-image operators or georeader-backed operators must add the matching extra (`pip install 'geotoolz-ops[raster,vector,morph,sensors]'` restores the old set). `pip install geotoolz` is unaffected.

### Features

* **ops:** slim base install with raster, vector, morph and sensors extras ([#577](https://github.com/jejjohnson/geotoolz/issues/577)) ([72456af](https://github.com/jejjohnson/geotoolz/commit/72456af981a168ba6d4e8f61e3fa5dec0b87a68f))

## [0.13.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-v0.12.0...geotoolz-v0.13.0) (2026-10-26)


### ⚠ BREAKING CHANGES

* **patcher:** batch_split yields PatchBatch instead of BatchedPatch. Read batch.patches.data, batch.valid and the transform rows instead of batch.data / anchors / indices / weights, and pass patcher= and domain= to unbatch. DataArray patches come back from unbatch as GeoArray. Plain array patches need the deprecated BatchedPatch.split.

### Features

* **array:** importing geoarray never imports JAX; pytrees register when JAX loads ([#566](https://github.com/jejjohnson/geotoolz/issues/566)) ([c370115](https://github.com/jejjohnson/geotoolz/commit/c370115c44debe3c846e796c92899192e258c843))
* **patcher:** batch_split yields a PatchBatch, one GeoArray with a patch dim ([#567](https://github.com/jejjohnson/geotoolz/issues/567)) ([7ede14d](https://github.com/jejjohnson/geotoolz/commit/7ede14df13ea8539dddd6494f885be09d9e540f4))

## [0.12.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-v0.11.0...geotoolz-v0.12.0) (2026-10-19)


### ⚠ BREAKING CHANGES

* **geotoolz:** geotoolz installs a stack whose public APIs take and return geoarray.GeoArray and geoarray.RasterSource; a georeader GeoTensor or reader is refused with a TypeError naming from_georeader / source_from_georeader.

### Features

* **geotoolz:** describe the stack on geoarray.GeoArray and require the GeoArray releases ([#507](https://github.com/jejjohnson/geotoolz/issues/507)) ([336156d](https://github.com/jejjohnson/geotoolz/commit/336156da80925020cfc2247692ba9674dc0b5cd7))

## [0.11.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-v0.10.3...geotoolz-v0.11.0) (2026-10-17)


### ⚠ BREAKING CHANGES

* `import geotoolz` no longer provides the operators. Use `import geoops` (distribution `geotoolz-ops`). `pip install geotoolz` now installs the whole stack.

### Features

* **patcher:** add geopatcher.integrations.train with XarrayWindowDataset ([#486](https://github.com/jejjohnson/geotoolz/issues/486)) ([f89118e](https://github.com/jejjohnson/geotoolz/commit/f89118e4cab93bced56893b319208d53cebb3433))


### Bug Fixes

* **build:** ship a LICENSE in each package so it builds ([#491](https://github.com/jejjohnson/geotoolz/issues/491)) ([48a705a](https://github.com/jejjohnson/geotoolz/commit/48a705a382a53ec0125c52817f3b4f29353de2aa))


### Code Refactoring

* rename the operator package to geotoolz-ops (import geoops) ([#489](https://github.com/jejjohnson/geotoolz/issues/489)) ([278b8af](https://github.com/jejjohnson/geotoolz/commit/278b8afbc675b9e082ac75488200a4729da38097))

## Changelog

Releases of the `geotoolz` distribution up to 0.10.3 were the operator
package, now `geotoolz-ops`; their notes are in
[`packages/geotoolz-ops/CHANGELOG.md`](../geotoolz-ops/CHANGELOG.md).
