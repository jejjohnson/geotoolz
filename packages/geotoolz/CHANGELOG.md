# Changelog

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
