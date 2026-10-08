# Changelog

## [0.2.1](https://github.com/jejjohnson/geotoolz/compare/geotoolz-cloud-v0.2.0...geotoolz-cloud-v0.2.1) (2026-10-08)


### Features

* **agents:** agent rules, contracts, capability index, recipe skills and a downstream plugin ([#436](https://github.com/jejjohnson/geotoolz/issues/436)) ([b11e390](https://github.com/jejjohnson/geotoolz/commit/b11e390c24de0d0c29d1c3ee8c1f9bb6681bbc15))

## [0.2.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-cloud-v0.1.0...geotoolz-cloud-v0.2.0) (2026-10-08)


### ⚠ BREAKING CHANGES

* **patcher:** no aliases are kept. The prefixed axis names (SpatialHann, TemporalMean, ...) and the root re-exports of axes, runners, hooks and caches are gone; geopatcher.time is geopatcher.temporal; geopatcher.objstore / geopatcher.cog become geocloud.store / geocloud.cog; geopatcher.runners / dask / jax / hooks fold into geopatcher.run and geopatcher.observe; ObstoreCogField becomes geopatcher.fields.CogField; the patcher extras obstore / obstore-cog are replaced by [cog]; and geotoolz-catalog[obstore] is removed (install geotoolz-cloud). Saved config envelopes that use the old class names must be regenerated.

### Code Refactoring

* **patcher:** organise geopatcher by task and split object storage into geotoolz-cloud ([#431](https://github.com/jejjohnson/geotoolz/issues/431)) ([a7a4d6e](https://github.com/jejjohnson/geotoolz/commit/a7a4d6e4fbec075260f8f4a9fd371a791ff13b30))
