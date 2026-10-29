# Changelog

## [0.3.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-ops-v0.2.0...xrtoolz-ops-v0.3.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **xrops:** knn regridding, kde_to_grid, xrops.learn and the decomposition presets need pip install 'xrtoolz-ops[sklearn]'.
* the xrtoolz-einx (xreinx) and xrtoolz-sklearn (xrsklearn) distributions are gone, with no shim (PL-Q5).
* **xrops:** the presets take sample_dims= (one dim or several) instead of sample_dim=, and name a new output dim with feature_dim. xrtoolz-ops no longer depends on xrtoolz-einx or xrtoolz-sklearn.

### Features

* **xrops:** xrops.learn and pack_dataset on pipekit-array's shared bridges ([#375](https://github.com/jejjohnson/xrtoolz/issues/375)) ([9c61801](https://github.com/jejjohnson/xrtoolz/commit/9c6180191af592b6a9dc5e7418a51435c73b0119))


### Code Refactoring

* delete xrtoolz-einx and xrtoolz-sklearn ([#376](https://github.com/jejjohnson/xrtoolz/issues/376)) ([6b51221](https://github.com/jejjohnson/xrtoolz/commit/6b51221771b2975cdeb019e5637e5f5ef4922539))
* **xrops:** a light base install without einx or scikit-learn ([#377](https://github.com/jejjohnson/xrtoolz/issues/377)) ([2b9bd37](https://github.com/jejjohnson/xrtoolz/commit/2b9bd37f7d7e2edb4b9d451b9e20ae6e4aa271a2))

## [0.2.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-ops-v0.1.0...xrtoolz-ops-v0.2.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **products:** install xrtoolz-products and import xrproducts; xrreader no longer exists, and the cache env var is XRPRODUCTS_CACHE.

### Code Refactoring

* **products:** rename xrtoolz-reader to xrtoolz-products (import xrproducts) ([#372](https://github.com/jejjohnson/xrtoolz/issues/372)) ([b1e642d](https://github.com/jejjohnson/xrtoolz/commit/b1e642d27ea6a42d0dc661634f993814f38a613b))

## 0.1.0 (2026-10-29)


### ⚠ BREAKING CHANGES

* **xrops:** import the operators from xrops (pip install xrtoolz-ops); xrtoolz.* paths, xrtoolz.Operator/Signature/Sequential and the calc, einx, signature and geo metric/encoder shims are gone.

### Code Refactoring

* **xrops:** create xrtoolz-ops (import xrops) and a code-free xrtoolz umbrella ([#368](https://github.com/jejjohnson/xrtoolz/issues/368)) ([d654087](https://github.com/jejjohnson/xrtoolz/commit/d654087aa6aeec67683930b8c98c492ed1e67083))

## Changelog
