# Changelog

## [0.3.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-core-v0.2.0...xrtoolz-core-v0.3.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* the xrtoolz-einx (xreinx) and xrtoolz-sklearn (xrsklearn) distributions are gone, with no shim (PL-Q5).

### Code Refactoring

* delete xrtoolz-einx and xrtoolz-sklearn ([#376](https://github.com/jejjohnson/xrtoolz/issues/376)) ([6b51221](https://github.com/jejjohnson/xrtoolz/commit/6b51221771b2975cdeb019e5637e5f5ef4922539))

## [0.2.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-core-v0.1.0...xrtoolz-core-v0.2.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **products:** install xrtoolz-products and import xrproducts; xrreader no longer exists, and the cache env var is XRPRODUCTS_CACHE.

### Code Refactoring

* **products:** rename xrtoolz-reader to xrtoolz-products (import xrproducts) ([#372](https://github.com/jejjohnson/xrtoolz/issues/372)) ([b1e642d](https://github.com/jejjohnson/xrtoolz/commit/b1e642d27ea6a42d0dc661634f993814f38a613b))

## [0.1.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-core-v0.0.2...xrtoolz-core-v0.1.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **xrops:** import the operators from xrops (pip install xrtoolz-ops); xrtoolz.* paths, xrtoolz.Operator/Signature/Sequential and the calc, einx, signature and geo metric/encoder shims are gone.

### Code Refactoring

* **xrops:** create xrtoolz-ops (import xrops) and a code-free xrtoolz umbrella ([#368](https://github.com/jejjohnson/xrtoolz/issues/368)) ([d654087](https://github.com/jejjohnson/xrtoolz/commit/d654087aa6aeec67683930b8c98c492ed1e67083))

## [0.0.2](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-core-v0.0.1...xrtoolz-core-v0.0.2) (2026-10-05)


### Features

* **sklearn,core:** support DataTree in XarrayEstimator, accessor and SklearnOp ([#333](https://github.com/jejjohnson/xrtoolz/issues/333)) ([a9fe358](https://github.com/jejjohnson/xrtoolz/commit/a9fe35810484f7397c7881a795a0142cc51d8939))

## 0.0.1 (2026-08-13)


### Features

* split xrtoolz into a uv workspace (core, grad, einx, sklearn) ([#252](https://github.com/jejjohnson/xrtoolz/issues/252)) ([f37a532](https://github.com/jejjohnson/xrtoolz/commit/f37a53286aa622c8bb4abe7a3f83df1ddc324bb4))
