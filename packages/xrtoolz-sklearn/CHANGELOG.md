# Changelog

## [0.1.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-sklearn-v0.0.2...xrtoolz-sklearn-v0.1.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **xrops:** import the operators from xrops (pip install xrtoolz-ops); xrtoolz.* paths, xrtoolz.Operator/Signature/Sequential and the calc, einx, signature and geo metric/encoder shims are gone.

### Code Refactoring

* **xrops:** create xrtoolz-ops (import xrops) and a code-free xrtoolz umbrella ([#368](https://github.com/jejjohnson/xrtoolz/issues/368)) ([d654087](https://github.com/jejjohnson/xrtoolz/commit/d654087aa6aeec67683930b8c98c492ed1e67083))

## [0.0.2](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-sklearn-v0.0.1...xrtoolz-sklearn-v0.0.2) (2026-10-05)


### Features

* **sklearn,core:** support DataTree in XarrayEstimator, accessor and SklearnOp ([#333](https://github.com/jejjohnson/xrtoolz/issues/333)) ([a9fe358](https://github.com/jejjohnson/xrtoolz/commit/a9fe35810484f7397c7881a795a0142cc51d8939))
* **sklearn:** add a NaN module with feature (land) and sample masking ([#332](https://github.com/jejjohnson/xrtoolz/issues/332)) ([dfd82d1](https://github.com/jejjohnson/xrtoolz/commit/dfd82d1f99204f2eb239353d281fe4608dd3fe5f))
* **sklearn:** choose output layout per method and round-trip Datasets ([#330](https://github.com/jejjohnson/xrtoolz/issues/330)) ([106919e](https://github.com/jejjohnson/xrtoolz/commit/106919e39e9f99440ec2546e00e8cfb5a116e134))


### Bug Fixes

* **sklearn:** validate inputs against the fit-time feature layout ([#329](https://github.com/jejjohnson/xrtoolz/issues/329)) ([6bfd1cb](https://github.com/jejjohnson/xrtoolz/commit/6bfd1cb979dc9c2f8ca437bd46a3df489a9fe3fb))

## 0.0.1 (2026-08-13)


### Features

* split xrtoolz into a uv workspace (core, grad, einx, sklearn) ([#252](https://github.com/jejjohnson/xrtoolz/issues/252)) ([f37a532](https://github.com/jejjohnson/xrtoolz/commit/f37a53286aa622c8bb4abe7a3f83df1ddc324bb4))
