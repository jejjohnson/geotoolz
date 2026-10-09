# Changelog

## [0.4.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-products-v0.3.0...geotoolz-products-v0.4.0) (2026-10-09)


### ⚠ BREAKING CHANGES

* **products:** geotoolz-products drops ProductReader.set_obstore_client, obstore_client and _read_bytes, and its [obstore] extra.
* **cloud:** geotoolz.io no longer exports select_indexes, read_indexes, fill_value_from_attrs or affine_from_geotransform (they are geocloud.hdf's, and raise ValueError rather than GeoToolzIOError). gz.io.WriteCOG takes write_cog's options (compress, level, predictor, blocksize, overviews, resampling, nodata, descriptions, tags, creation_options) instead of a rasterio profile, and needs the new geotoolz [cloud] extra; geotoolz's [hdf5] / [hdf4] / [netcdf] extras now install geotoolz-cloud with that backend. geoproducts' [goes] / [himawari] extras install geotoolz-cloud[hdf5] in place of h5py.

### Features

* **cloud:** geocloud.hdf — the stack's HDF5 / NetCDF / HDF4 readers; gz.io and the product readers build on it ([#458](https://github.com/jejjohnson/geotoolz/issues/458)) ([5396274](https://github.com/jejjohnson/geotoolz/commit/539627449a3901c3c17e9ace74ba1ea09d529359))


### Code Refactoring

* **products:** drop the dead ProductReader byte path; cloud writes and atomic downloads ([#459](https://github.com/jejjohnson/geotoolz/issues/459)) ([903ced0](https://github.com/jejjohnson/geotoolz/commit/903ced0adaa68078d3a207dfe9c8ba2d5086467f))

## [0.3.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-products-v0.2.1...geotoolz-products-v0.3.0) (2026-10-08)


### ⚠ BREAKING CHANGES

* geocatalog.staging.stage() needs the [cloud] extra (geotoolz-cloud) for remote URIs instead of [fsspec]. goes.aws / himawari.aws need geotoolz-cloud (the [goes] / [himawari] / [obstore] extras) and raise obstore errors (FileNotFoundError for a missing key) instead of urllib.error.URLError.

### Features

* **cloud:** add geocloud.files — list, read, write, move and sign objects by uri ([#441](https://github.com/jejjohnson/geotoolz/issues/441)) ([a6a4371](https://github.com/jejjohnson/geotoolz/commit/a6a437110e30b6f5c66abcdaeffe6abbf4afd57f))


### Code Refactoring

* download through geocloud.files in catalog staging and the NOAA bucket helpers ([#443](https://github.com/jejjohnson/geotoolz/issues/443)) ([1450117](https://github.com/jejjohnson/geotoolz/commit/14501176866ff3003405cebffc45b99d7758f9a7))

## [0.2.1](https://github.com/jejjohnson/geotoolz/compare/geotoolz-products-v0.2.0...geotoolz-products-v0.2.1) (2026-10-08)


### Features

* **agents:** agent rules, contracts, capability index, recipe skills and a downstream plugin ([#436](https://github.com/jejjohnson/geotoolz/issues/436)) ([b11e390](https://github.com/jejjohnson/geotoolz/commit/b11e390c24de0d0c29d1c3ee8c1f9bb6681bbc15))

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
