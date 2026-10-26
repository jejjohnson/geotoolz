# Changelog

## [0.3.1](https://github.com/jejjohnson/geotoolz/compare/geotoolz-array-v0.3.0...geotoolz-array-v0.3.1) (2026-10-26)


### Features

* **array:** importing geoarray never imports JAX; pytrees register when JAX loads ([#566](https://github.com/jejjohnson/geotoolz/issues/566)) ([c370115](https://github.com/jejjohnson/geotoolz/commit/c370115c44debe3c846e796c92899192e258c843))

## [0.3.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-array-v0.2.1...geotoolz-array-v0.3.0) (2026-10-24)


### ⚠ BREAKING CHANGES

* **array:** geotoolz-array 0.3.0 stores dims and coordinates. Building a GeoArray without dims=, per-band names in attrs, stack() and expand_dims() without a dim name, and GeoArray[np.ndarray] are deprecated and removed in 0.4.0.
* **array:** to_pixel, interp_at and the adapter-level map_over ([#550](https://github.com/jejjohnson/geotoolz/issues/550))
* **array:** GeoArray's dims are stored. Construction without dims is deprecated and removed in 0.4.0; geoarray.stack without dim= is deprecated; geoarray.combine is pipekit_array.combine.

### Features

* **array:** add the GeoSwath and GeoPoints carriers with their pipekit adapters ([#548](https://github.com/jejjohnson/geotoolz/issues/548)) ([ae04513](https://github.com/jejjohnson/geotoolz/commit/ae04513d35e7b8322ca8a1a323ef8f4494e10b2b))
* **array:** readable types in geoarray.types and check_dims ([#549](https://github.com/jejjohnson/geotoolz/issues/549)) ([b9678c3](https://github.com/jejjohnson/geotoolz/commit/b9678c395a24f840dfa815b862d543c49d2afbd6)), closes [#521](https://github.com/jejjohnson/geotoolz/issues/521)
* **array:** standard coordinates in geoarray.coords ([#545](https://github.com/jejjohnson/geotoolz/issues/545)) ([1235980](https://github.com/jejjohnson/geotoolz/commit/12359807a51f495820e1ea771a91fdc17b8c95a1))
* **array:** store dims and coordinates on GeoArray and register its pipekit adapter ([#546](https://github.com/jejjohnson/geotoolz/issues/546)) ([8480596](https://github.com/jejjohnson/geotoolz/commit/84805969f6e633299ef0b447a10578f6af993628))
* **array:** store TimeAxis stamps as an int64 pytree leaf; to_jax and to_numpy move them ([#547](https://github.com/jejjohnson/geotoolz/issues/547)) ([dee154e](https://github.com/jejjohnson/geotoolz/commit/dee154e5d122ba02db250f5739184239675dacb9))
* **array:** to_pixel, interp_at and the adapter-level map_over ([#550](https://github.com/jejjohnson/geotoolz/issues/550)) ([e563d70](https://github.com/jejjohnson/geotoolz/commit/e563d70e1489680cbcb59904aa82c1050117cc8f))
* **array:** xarray converters with a CF spatial_ref and GeoTransform ([#551](https://github.com/jejjohnson/geotoolz/issues/551)) ([5a1ff0c](https://github.com/jejjohnson/geotoolz/commit/5a1ff0c708372856c88dd886956e214ed6f21fbf))


### Miscellaneous

* **array:** read 0.2 band_names into a SpectralAxis, toward geotoolz-array 0.3.0 ([#552](https://github.com/jejjohnson/geotoolz/issues/552)) ([7b1a091](https://github.com/jejjohnson/geotoolz/commit/7b1a0919c6c646cdd660cc8801316255382a5992))

## [0.2.1](https://github.com/jejjohnson/geotoolz/compare/geotoolz-array-v0.2.0...geotoolz-array-v0.2.1) (2026-10-24)


### Bug Fixes

* **array:** check transforms at run time when joins run under jit ([#542](https://github.com/jejjohnson/geotoolz/issues/542)) ([44af6d9](https://github.com/jejjohnson/geotoolz/commit/44af6d9e050e6475d36e697453efd33b429b9e90))

## [0.2.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-array-v0.1.0...geotoolz-array-v0.2.0) (2026-10-19)


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
