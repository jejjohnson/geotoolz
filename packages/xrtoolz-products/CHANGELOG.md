# Changelog

## [0.4.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-products-v0.3.0...xrtoolz-products-v0.4.0) (2026-10-30)


### ⚠ BREAKING CHANGES

* the xrtoolz-grad (xrgrad) distribution is gone, with no shim (PL-Q5): use from xrops import grad.
* **xrops:** the xrtoolz-core distribution and the xrcore import are gone, with no shim (PL-Q5): use from xrops import Operator and from pipekit import Signature.

### Code Refactoring

* delete xrtoolz-grad; xrops.grad is the calculus ([#382](https://github.com/jejjohnson/xrtoolz/issues/382)) ([82f6fa7](https://github.com/jejjohnson/xrtoolz/commit/82f6fa78faf510fde4cd8a389dd97b529645a4fe))
* **xrops:** fold xrcore into xrops._src and delete xrtoolz-core ([#380](https://github.com/jejjohnson/xrtoolz/issues/380)) ([756fc30](https://github.com/jejjohnson/xrtoolz/commit/756fc3074317737e8546ffa08890d7565aee3aa0))

## [0.3.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-products-v0.2.0...xrtoolz-products-v0.3.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* the xrtoolz-einx (xreinx) and xrtoolz-sklearn (xrsklearn) distributions are gone, with no shim (PL-Q5).

### Code Refactoring

* delete xrtoolz-einx and xrtoolz-sklearn ([#376](https://github.com/jejjohnson/xrtoolz/issues/376)) ([6b51221](https://github.com/jejjohnson/xrtoolz/commit/6b51221771b2975cdeb019e5637e5f5ef4922539))

## [0.2.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-products-v0.1.0...xrtoolz-products-v0.2.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **products:** install xrtoolz-products and import xrproducts; xrreader no longer exists, and the cache env var is XRPRODUCTS_CACHE.

### Code Refactoring

* **products:** rename xrtoolz-reader to xrtoolz-products (import xrproducts) ([#372](https://github.com/jejjohnson/xrtoolz/issues/372)) ([b1e642d](https://github.com/jejjohnson/xrtoolz/commit/b1e642d27ea6a42d0dc661634f993814f38a613b))

## [0.1.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-reader-v0.0.5...xrtoolz-reader-v0.1.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* **xrops:** import the operators from xrops (pip install xrtoolz-ops); xrtoolz.* paths, xrtoolz.Operator/Signature/Sequential and the calc, einx, signature and geo metric/encoder shims are gone.

### Code Refactoring

* **xrops:** create xrtoolz-ops (import xrops) and a code-free xrtoolz umbrella ([#368](https://github.com/jejjohnson/xrtoolz/issues/368)) ([d654087](https://github.com/jejjohnson/xrtoolz/commit/d654087aa6aeec67683930b8c98c492ed1e67083))

## [0.0.5](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-reader-v0.0.4...xrtoolz-reader-v0.0.5) (2026-10-27)


### Features

* **reader:** open gridded CH₄ emission inventories as a CF flux ([#357](https://github.com/jejjohnson/xrtoolz/issues/357)) ([405466b](https://github.com/jejjohnson/xrtoolz/commit/405466bb39c696bcb532a666fbf1e0e52c33f245))

## [0.0.4](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-reader-v0.0.3...xrtoolz-reader-v0.0.4) (2026-09-18)


### Features

* **reader:** add ERA5 boundary-layer / pressure-level and methane L2 Variable entries ([#301](https://github.com/jejjohnson/xrtoolz/issues/301)) ([d8314da](https://github.com/jejjohnson/xrtoolz/commit/d8314dab3de52d4a3d4109ebd238a9083fadea4b))
* **reader:** add LocalL2Source with TROPOMI / EMIT / GHGSat methane L2 openers ([#303](https://github.com/jejjohnson/xrtoolz/issues/303)) ([d04768e](https://github.com/jejjohnson/xrtoolz/commit/d04768e3c489f09b8cf383db6ddf8ee8890980b0))

## [0.0.3](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-reader-v0.0.2...xrtoolz-reader-v0.0.3) (2026-08-17)


### Features

* fold xrreader into the workspace as xrtoolz-reader ([#259](https://github.com/jejjohnson/xrtoolz/issues/259)) ([aa0859b](https://github.com/jejjohnson/xrtoolz/commit/aa0859b1aafb123d6d4e21967ed46dcd9386b3ef))

## [0.0.2](https://github.com/jejjohnson/xrreader/compare/v0.0.1...v0.0.2) (2026-06-05)


### Features

* seed xrreader by migrating xrtoolz.data + types (P0) ([#7](https://github.com/jejjohnson/xrreader/issues/7)) ([4a440d1](https://github.com/jejjohnson/xrreader/commit/4a440d1f6ee24857d1d97beffb1ddf4429ba21b9))

## [0.1.6](https://github.com/jejjohnson/pypackage_template/compare/v0.1.5...v0.1.6) (2026-05-14)


### Features

* **gh:** add issue templates, label taxonomy, and contributing guide ([#34](https://github.com/jejjohnson/pypackage_template/issues/34)) ([66e36a1](https://github.com/jejjohnson/pypackage_template/commit/66e36a1848f1899ef85d900b06bf2d269d346d5f))
* **gh:** create-gh-issue + link-gh-issues skills for the issue-template workflow ([#36](https://github.com/jejjohnson/pypackage_template/issues/36)) ([caa5100](https://github.com/jejjohnson/pypackage_template/commit/caa51008dc7917f5cc526333dde6fb8abf310585))

## [0.1.5](https://github.com/jejjohnson/pypackage_template/compare/v0.1.4...v0.1.5) (2026-03-25)


### Features

* **docs:** add MathJax support for equations ([3bca9eb](https://github.com/jejjohnson/pypackage_template/commit/3bca9ebcf10212755859dfe13f4024d7bf3d05df))
* **docs:** add MathJax support for markdown and notebook equations ([b31ab41](https://github.com/jejjohnson/pypackage_template/commit/b31ab4155886ed610b7238a22bbfab0ac4723a1a))


### Bug Fixes

* **docs:** guard MathJax subscribe callback against load failures ([c7173a6](https://github.com/jejjohnson/pypackage_template/commit/c7173a6d3847e9c794f0213477278f6582edbd8e))
* **docs:** pin MathJax CDN to exact version 3.2.2 ([e3130ca](https://github.com/jejjohnson/pypackage_template/commit/e3130ca8f3af5e2597982fe5c1cb779b5f144a5e))

## [0.1.4](https://github.com/jejjohnson/pypackage_template/compare/v0.1.3...v0.1.4) (2026-03-24)


### Bug Fixes

* address PR [#19](https://github.com/jejjohnson/pypackage_template/issues/19) review comments ([65b8d57](https://github.com/jejjohnson/pypackage_template/commit/65b8d57e9792a1877d0fd3c5d04a29e121e9219a))
* resolve lint failures in example notebook and invalid ipynb ([bf55f60](https://github.com/jejjohnson/pypackage_template/commit/bf55f607db5c5daaaefa3f732eaeee06ac5c898c))

## [0.1.3](https://github.com/jejjohnson/pypackage_template/compare/v0.1.2...v0.1.3) (2026-03-02)


### Bug Fixes

* address review comments on Makefile .env export and test targets ([c91dfe5](https://github.com/jejjohnson/pypackage_template/commit/c91dfe51896e0e382084c689821eebebdb61eace))

## [0.1.2](https://github.com/jejjohnson/pypackage_template/compare/v0.1.1...v0.1.2) (2026-03-02)


### Features

* add CODE_REVIEW.md and AI code review workflow ([258275b](https://github.com/jejjohnson/pypackage_template/commit/258275b5867c28212a94608178511c1c4e9024a3))
* add CODE_REVIEW.md and Copilot code review instructions ([f4d2489](https://github.com/jejjohnson/pypackage_template/commit/f4d2489e4b31fd431a6e767b27a0af7319965588))


### Bug Fixes

* avoid shell interpretation of backticks in CODE_REVIEW.md content ([b733a16](https://github.com/jejjohnson/pypackage_template/commit/b733a169107c934c4961fe64f507068d05969912))
* pass PR title/body via env block to prevent shell interpretation ([93f1dcc](https://github.com/jejjohnson/pypackage_template/commit/93f1dcc985342d8e4f735d89cd17d75e3d181f66))
* prevent printf from parsing markdown bullets as options ([7bc501e](https://github.com/jejjohnson/pypackage_template/commit/7bc501e45715df03ee0efd9e30e499b80bb111f5))

## [0.1.1](https://github.com/jejjohnson/pypackage_template/compare/v0.1.0...v0.1.1) (2026-03-01)


### Features

* **ci:** add Dependabot, CodeQL, pre-commit autoupdate, Codecov, PR labeler, and README badges ([d6f293b](https://github.com/jejjohnson/pypackage_template/commit/d6f293b010a7c5eb03a401baf873d03788704be6))
* **docs:** add MkDocs + Material documentation infrastructure ([879be43](https://github.com/jejjohnson/pypackage_template/commit/879be4360f38307b124ebfedb07ea550bd51f83b))


### Bug Fixes

* address review comments on pyproject.toml and tests setup ([bc9f636](https://github.com/jejjohnson/pypackage_template/commit/bc9f636a6e20eb4f5fa45d2c64cbcb546383c252))

## Changelog

All notable changes to this project will be documented in this file.

See [Conventional Commits](https://www.conventionalcommits.org/) for commit guidelines.
