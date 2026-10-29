# Changelog

## [0.1.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-einx-v0.0.1...xrtoolz-einx-v0.1.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **xrops:** import the operators from xrops (pip install xrtoolz-ops); xrtoolz.* paths, xrtoolz.Operator/Signature/Sequential and the calc, einx, signature and geo metric/encoder shims are gone.

### Code Refactoring

* **xrops:** create xrtoolz-ops (import xrops) and a code-free xrtoolz umbrella ([#368](https://github.com/jejjohnson/xrtoolz/issues/368)) ([d654087](https://github.com/jejjohnson/xrtoolz/commit/d654087aa6aeec67683930b8c98c492ed1e67083))

## 0.0.1 (2026-08-13)


### Features

* split xrtoolz into a uv workspace (core, grad, einx, sklearn) ([#252](https://github.com/jejjohnson/xrtoolz/issues/252)) ([f37a532](https://github.com/jejjohnson/xrtoolz/commit/f37a53286aa622c8bb4abe7a3f83df1ddc324bb4))
