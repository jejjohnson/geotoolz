# Changelog

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
