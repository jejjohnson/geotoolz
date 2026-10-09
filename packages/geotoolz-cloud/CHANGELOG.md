# Changelog

## [0.4.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-cloud-v0.3.0...geotoolz-cloud-v0.4.0) (2026-10-09)


### ⚠ BREAKING CHANGES

* **cloud:** geotoolz.io no longer exports select_indexes, read_indexes, fill_value_from_attrs or affine_from_geotransform (they are geocloud.hdf's, and raise ValueError rather than GeoToolzIOError). gz.io.WriteCOG takes write_cog's options (compress, level, predictor, blocksize, overviews, resampling, nodata, descriptions, tags, creation_options) instead of a rasterio profile, and needs the new geotoolz [cloud] extra; geotoolz's [hdf5] / [hdf4] / [netcdf] extras now install geotoolz-cloud with that backend. geoproducts' [goes] / [himawari] extras install geotoolz-cloud[hdf5] in place of h5py.
* **cloud:** geocatalog.staging.LocalCache and cache_key move to geocloud.cache (no alias). The default cache directory is $GEOCLOUD_CACHE, else ~/.cache/geocloud (was $GEOCATALOG_CACHE / ~/.cache/geocatalog).
* **cloud:** geotoolz-catalog drops its [fsspec] extra; remote reads need [cloud] (now geotoolz-cloud[fsspec]). Catalog storage_options are obstore store options forwarded to geocloud.store.get_obstore ({"skip_signature": True} for a public bucket), not s3fs / gcsfs options ({"anon": True}).
* **cloud:** SUPPORTED_SCHEMES still lists only remote schemes, but get_obstore no longer raises for a local path or file:// URI - it returns the pooled LocalStore.

### Features

* **cloud:** geocloud.cache — local copies of remote objects; catalog staging fetches into it ([#457](https://github.com/jejjohnson/geotoolz/issues/457)) ([f605f9b](https://github.com/jejjohnson/geotoolz/commit/f605f9be383229f4394e0f3320eb1658e6c37af1))
* **cloud:** geocloud.fs, an fsspec filesystem on the pool; the catalog reads through it ([#455](https://github.com/jejjohnson/geotoolz/issues/455)) ([951b85a](https://github.com/jejjohnson/geotoolz/commit/951b85a7938adec28d4fbc29b3923573cb0db317))
* **cloud:** geocloud.hdf — the stack's HDF5 / NetCDF / HDF4 readers; gz.io and the product readers build on it ([#458](https://github.com/jejjohnson/geotoolz/issues/458)) ([5396274](https://github.com/jejjohnson/geotoolz/commit/539627449a3901c3c17e9ace74ba1ea09d529359))
* **cloud:** serve local paths from the pool, and one public local-vs-remote rule ([#454](https://github.com/jejjohnson/geotoolz/issues/454)) ([eb994cb](https://github.com/jejjohnson/geotoolz/commit/eb994cba402ec2bb3ba30ffb70ac62850eca3ba7))

## [0.3.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-cloud-v0.2.1...geotoolz-cloud-v0.3.0) (2026-10-08)


### ⚠ BREAKING CHANGES

* geocatalog.staging.stage() needs the [cloud] extra (geotoolz-cloud) for remote URIs instead of [fsspec]. goes.aws / himawari.aws need geotoolz-cloud (the [goes] / [himawari] / [obstore] extras) and raise obstore errors (FileNotFoundError for a missing key) instead of urllib.error.URLError.

### Features

* **cloud:** add geocloud.cog.write_cog — validated COG writes, local or to a bucket ([#445](https://github.com/jejjohnson/geotoolz/issues/445)) ([562e00b](https://github.com/jejjohnson/geotoolz/commit/562e00b553acb64214343971b4c3c3e9f2b434ad))
* **cloud:** add geocloud.credentials — register credentials once per store root ([#442](https://github.com/jejjohnson/geotoolz/issues/442)) ([0bfe99a](https://github.com/jejjohnson/geotoolz/commit/0bfe99ac8fb924e195c54010c22f9c18609689f7))
* **cloud:** add geocloud.files — list, read, write, move and sign objects by uri ([#441](https://github.com/jejjohnson/geotoolz/issues/441)) ([a6a4371](https://github.com/jejjohnson/geotoolz/commit/a6a437110e30b6f5c66abcdaeffe6abbf4afd57f))


### Code Refactoring

* download through geocloud.files in catalog staging and the NOAA bucket helpers ([#443](https://github.com/jejjohnson/geotoolz/issues/443)) ([1450117](https://github.com/jejjohnson/geotoolz/commit/14501176866ff3003405cebffc45b99d7758f9a7))

## [0.2.1](https://github.com/jejjohnson/geotoolz/compare/geotoolz-cloud-v0.2.0...geotoolz-cloud-v0.2.1) (2026-10-08)


### Features

* **agents:** agent rules, contracts, capability index, recipe skills and a downstream plugin ([#436](https://github.com/jejjohnson/geotoolz/issues/436)) ([b11e390](https://github.com/jejjohnson/geotoolz/commit/b11e390c24de0d0c29d1c3ee8c1f9bb6681bbc15))

## [0.2.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-cloud-v0.1.0...geotoolz-cloud-v0.2.0) (2026-10-08)


### ⚠ BREAKING CHANGES

* **patcher:** no aliases are kept. The prefixed axis names (SpatialHann, TemporalMean, ...) and the root re-exports of axes, runners, hooks and caches are gone; geopatcher.time is geopatcher.temporal; geopatcher.objstore / geopatcher.cog become geocloud.store / geocloud.cog; geopatcher.runners / dask / jax / hooks fold into geopatcher.run and geopatcher.observe; ObstoreCogField becomes geopatcher.fields.CogField; the patcher extras obstore / obstore-cog are replaced by [cog]; and geotoolz-catalog[obstore] is removed (install geotoolz-cloud). Saved config envelopes that use the old class names must be regenerated.

### Code Refactoring

* **patcher:** organise geopatcher by task and split object storage into geotoolz-cloud ([#431](https://github.com/jejjohnson/geotoolz/issues/431)) ([a7a4d6e](https://github.com/jejjohnson/geotoolz/commit/a7a4d6e4fbec075260f8f4a9fd371a791ff13b30))
