# Changelog

## [0.7.2](https://github.com/jejjohnson/geotoolz/compare/xrtoolz-ops-v0.7.1...xrtoolz-ops-v0.7.2) (2026-11-26)


### Bug Fixes

* **deps:** geotoolz imports the areas of pipekit-ops 0.4.0 and pipekit-train 0.3.0 ([#819](https://github.com/jejjohnson/geotoolz/issues/819)) ([2223b6c](https://github.com/jejjohnson/geotoolz/commit/2223b6c92669275b1ddc19ded418c2c9731879c5))

## [0.7.1](https://github.com/jejjohnson/geotoolz/compare/xrtoolz-ops-v0.7.0...xrtoolz-ops-v0.7.1) (2026-11-25)


### Bug Fixes

* **deps:** pipekit-array 0.2.0 and its areas ([#816](https://github.com/jejjohnson/geotoolz/issues/816)) ([52c0583](https://github.com/jejjohnson/geotoolz/commit/52c0583b26701e4054fc027c1ec0c74c06e17af9))

## [0.7.0](https://github.com/jejjohnson/geotoolz/compare/xrtoolz-ops-v0.6.0...xrtoolz-ops-v0.7.0) (2026-11-25)


### ⚠ BREAKING CHANGES

* **xrops:** spectral, physical and Diebold–Mariano metrics call diagnostix.skill ([#814](https://github.com/jejjohnson/geotoolz/issues/814))
* **xrops:** pixel and ensemble metrics call diagnostix.skill ([#813](https://github.com/jejjohnson/geotoolz/issues/813))
* **xrops:** the Torrence–Compo and 2-D Morlet wavelets call diagnostix.spectra ([#809](https://github.com/jejjohnson/geotoolz/issues/809))
* **xrops:** eof calls diagnostix.eofs, with weights and land ([#808](https://github.com/jejjohnson/geotoolz/issues/808))
* **xrops:** spectra, fluxes and shapes call diagnostix.spectra, and xrft goes ([#807](https://github.com/jejjohnson/geotoolz/issues/807))
* **xrops:** budgets call diagnostix.budgets, and boundary_flux is the flux through a boundary ([#806](https://github.com/jejjohnson/geotoolz/issues/806))
* **xrops:** climatologies have one slot per calendar date, and the smoothing is centred ([#805](https://github.com/jejjohnson/geotoolz/issues/805))
* **xrops:** ocn's and atm's column diagnostics call diagnostix.columns ([#803](https://github.com/jejjohnson/geotoolz/issues/803))

### Features

* **xrops:** budgets call diagnostix.budgets, and boundary_flux is the flux through a boundary ([#806](https://github.com/jejjohnson/geotoolz/issues/806)) ([65d04d9](https://github.com/jejjohnson/geotoolz/commit/65d04d987c55371f34f06c5993f686847ea0ba83))
* **xrops:** eof calls diagnostix.eofs, with weights and land ([#808](https://github.com/jejjohnson/geotoolz/issues/808)) ([a30fb26](https://github.com/jejjohnson/geotoolz/commit/a30fb26c4b7b7041a589f6b7dff499d72301ff2f))
* **xrops:** spectra, fluxes and shapes call diagnostix.spectra, and xrft goes ([#807](https://github.com/jejjohnson/geotoolz/issues/807)) ([bed49c6](https://github.com/jejjohnson/geotoolz/commit/bed49c6370528e5e89e48c89d9cbf51f6dc1d032))


### Bug Fixes

* **xrops:** a diagnostic does not have the units of its input ([#811](https://github.com/jejjohnson/geotoolz/issues/811)) ([a0e31ca](https://github.com/jejjohnson/geotoolz/commit/a0e31ca42b2f911a1961ada32e76d3f83f66b41a))
* **xrops:** climatologies have one slot per calendar date, and the smoothing is centred ([#805](https://github.com/jejjohnson/geotoolz/issues/805)) ([2b0e8b1](https://github.com/jejjohnson/geotoolz/commit/2b0e8b144043d0130931cfe97b2fac464a527834))
* **xrops:** ocn's and atm's column diagnostics call diagnostix.columns ([#803](https://github.com/jejjohnson/geotoolz/issues/803)) ([2804109](https://github.com/jejjohnson/geotoolz/commit/28041097c915c1e3f8f292bbb33fbba5f866082d))
* **xrops:** ocn's horizontal diagnostics call gridax and diagnostix, with the sphere's curvature terms ([#802](https://github.com/jejjohnson/geotoolz/issues/802)) ([030e333](https://github.com/jejjohnson/geotoolz/commit/030e33378daa88649c8f229e5c772c51fd4b88a8))
* **xrops:** pixel and ensemble metrics call diagnostix.skill ([#813](https://github.com/jejjohnson/geotoolz/issues/813)) ([d7b29ff](https://github.com/jejjohnson/geotoolz/commit/d7b29ff8c11fe1b788b301372d184f99e0a49d9d))
* **xrops:** spectral, physical and Diebold–Mariano metrics call diagnostix.skill ([#814](https://github.com/jejjohnson/geotoolz/issues/814)) ([fcb9506](https://github.com/jejjohnson/geotoolz/commit/fcb95062d68f141ee50a667298c66bf4436dd77c))
* **xrops:** the Torrence–Compo and 2-D Morlet wavelets call diagnostix.spectra ([#809](https://github.com/jejjohnson/geotoolz/issues/809)) ([f49ece3](https://github.com/jejjohnson/geotoolz/commit/f49ece3792092e8e65459f38127959fe008a28b4))

## [0.6.0](https://github.com/jejjohnson/geotoolz/compare/xrtoolz-ops-v0.5.1...xrtoolz-ops-v0.6.0) (2026-11-24)


### ⚠ BREAKING CHANGES

* **array:** carriers join NumPy and JAX, on pipekit's backend rule and guard ([#778](https://github.com/jejjohnson/geotoolz/issues/778))

### Bug Fixes

* **xrops:** geophysix takes the DataArrays itself, so the elementwise wrapper goes ([#781](https://github.com/jejjohnson/geotoolz/issues/781)) ([a02cebc](https://github.com/jejjohnson/geotoolz/commit/a02cebcbf1e57ff205507b8bd0f9687f6fc4184c))


### Code Refactoring

* **array:** carriers join NumPy and JAX, on pipekit's backend rule and guard ([#778](https://github.com/jejjohnson/geotoolz/issues/778)) ([f120d74](https://github.com/jejjohnson/geotoolz/commit/f120d749fe51968c3102dc3727ea687d22c4c8b8))

## [0.5.1](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-ops-v0.5.0...xrtoolz-ops-v0.5.1) (2026-11-12)


### Features

* **interpolate:** run the gap fills and IDW on functax plans ([#401](https://github.com/jejjohnson/xrtoolz/issues/401)) ([d81a2e8](https://github.com/jejjohnson/xrtoolz/commit/d81a2e8ae91fdf9cf34024604e4120b5a1c21125))

## [0.5.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-ops-v0.4.0...xrtoolz-ops-v0.5.0) (2026-10-30)


### ⚠ BREAKING CHANGES

* **xrops:** xrops.geo is gone, with no shim (PL-Q5): import from xrops.geom, xrops.mask, xrops.anomaly, xrops.extremes, xrops.wavelets or xrops.cf; xrops.geo.Reduce is xrops.utils.ReduceDims and FillNaN is xrops.interpolate.FillNaN.

### Features

* **xrops:** declared backends and the lane rules on every xrops operator ([#390](https://github.com/jejjohnson/xrtoolz/issues/390)) ([266dfbf](https://github.com/jejjohnson/xrtoolz/commit/266dfbfdaed7c279e455f7b1b391eaa2dd2136ee))
* **xrops:** xrops.lift, a geoops geometry operator per item over a cube ([#389](https://github.com/jejjohnson/xrtoolz/issues/389)) ([57151e3](https://github.com/jejjohnson/xrtoolz/commit/57151e313e4f6add4ae455a690a67e01df8513e1))


### Bug Fixes

* **xrops:** an unknown input signature passes through every operator ([#386](https://github.com/jejjohnson/xrtoolz/issues/386)) ([41bc514](https://github.com/jejjohnson/xrtoolz/commit/41bc514f8c8cb9c38895da58584e725f0f99489e))


### Code Refactoring

* **xrops:** split xrops.geo into geom, mask, anomaly, extremes, wavelets and cf ([#387](https://github.com/jejjohnson/xrtoolz/issues/387)) ([c23740b](https://github.com/jejjohnson/xrtoolz/commit/c23740ba3bd750c70dfa79bfc034a7803aa4e317))

## [0.4.0](https://github.com/jejjohnson/xrtoolz/compare/xrtoolz-ops-v0.3.0...xrtoolz-ops-v0.4.0) (2026-10-30)


### ⚠ BREAKING CHANGES

* the xrtoolz-grad (xrgrad) distribution is gone, with no shim (PL-Q5): use from xrops import grad.
* **xrops:** the xrtoolz-core distribution and the xrcore import are gone, with no shim (PL-Q5): use from xrops import Operator and from pipekit import Signature.

### Features

* **xrops:** lazy xrops.grad on dask, per chunk with stencil halos ([#383](https://github.com/jejjohnson/xrtoolz/issues/383)) ([e835d77](https://github.com/jejjohnson/xrtoolz/commit/e835d771d81052fb693401c2d41af66a746a1371))
* **xrops:** xrops.grad on geoops's difference kernels, with Gradient and Difference ([#381](https://github.com/jejjohnson/xrtoolz/issues/381)) ([3223e82](https://github.com/jejjohnson/xrtoolz/commit/3223e828090de98d5c3b5838363bcd13ccb9e098))


### Code Refactoring

* delete xrtoolz-grad; xrops.grad is the calculus ([#382](https://github.com/jejjohnson/xrtoolz/issues/382)) ([82f6fa7](https://github.com/jejjohnson/xrtoolz/commit/82f6fa78faf510fde4cd8a389dd97b529645a4fe))
* **xrops:** fold xrcore into xrops._src and delete xrtoolz-core ([#380](https://github.com/jejjohnson/xrtoolz/issues/380)) ([756fc30](https://github.com/jejjohnson/xrtoolz/commit/756fc3074317737e8546ffa08890d7565aee3aa0))

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
