# Changelog

## [0.6.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-patcher-v0.5.1...geotoolz-patcher-v0.6.0) (2026-10-05)


### ⚠ BREAKING CHANGES

* **patcher/cache+journal:** `PatchCache.stats()` "bytes" / "entries" are now this instance's incremental tally rather than a directory walk (entries written by other processes are counted once hit or on the next construction), and the `max_bytes` cap applies per writer. Entries larger than `max_bytes` are no longer written. Multi-file reader identities and `ObstoreCogField.cache_id()` changed, invalidating their existing cache entries. Journal rows now store normalised anchors (lists, Python scalars); anchors that previously only keyed through `default=str` (arbitrary objects, non-string dict keys) now raise TypeError, and numpy-anchor keys change from their stringified form.
* **patcher/cache:** the cache key and on-disk entry layout changed, so existing PatchCache directories are never hit again (call `cache.clear()` to reclaim space). `field_id_for` returns a different string. Patches whose carrier cannot be rebuilt bit-identically (GeoPandasField / XvecField patches, object-dtype data, attrs without a lossless encoding) now raise TypeError on `put` instead of being cached as bare ndarrays. A DataArray field opened from a file is now identified by `encoding["source"]` instead of requiring `field_id`.

### Bug Fixes

* **patcher/cache+journal:** incremental LRU eviction, full field identity, numpy anchors ([#378](https://github.com/jejjohnson/geotoolz/issues/378)) ([8e949b4](https://github.com/jejjohnson/geotoolz/commit/8e949b44892eac2c94ceb6cb10de76663f46e917))
* **patcher/cache:** key covers domain, bands and adapter identity; bit-identical, self-healing entries ([#377](https://github.com/jejjohnson/geotoolz/issues/377)) ([88e3066](https://github.com/jejjohnson/geotoolz/commit/88e306614b17aedfb6f002adb03ec3b8535654bc))

## [0.5.1](https://github.com/jejjohnson/geotoolz/compare/geotoolz-patcher-v0.5.0...geotoolz-patcher-v0.5.1) (2026-10-03)


### Bug Fixes

* **catalog,patcher:** get_config()["crs"] is one string on both backends and across round trips; pool key covers every endpoint variable; hf:// in the pool ([#371](https://github.com/jejjohnson/geotoolz/issues/371)) ([6b8b3ff](https://github.com/jejjohnson/geotoolz/commit/6b8b3ff34489958ed89ff210a3e4abb7961dd501))

## [0.5.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-patcher-v0.4.0...geotoolz-patcher-v0.5.0) (2026-10-03)


### ⚠ BREAKING CHANGES

* **patcher/aggregation:** `SpatialOverlapAdd(streaming=True, writer="zarr")` now requires `chunks=` and raises FileExistsError when `target_path` already holds a store unless `overwrite=True`; `writer="cog"` raises FileExistsError for an existing file unless `overwrite=True`, and writes a GDAL COG with `nodata = fill_value`. An unknown `writer` or a non-floating `dtype` raises ValueError.
* **patcher/aggregation:** cells no valid sample reached (uncovered, all-NaN, outside a mask, or zero accumulated weight) are NaN instead of 0.0 / ±inf in every dense aggregation, and -1 instead of class 0 in HardVote / SoftVote; pass `fill_value=` to override. SpatialVariance cells with one sample are NaN instead of 0.0. SpatialMode returns float64 unless an integral `fill_value` is given. SpatialByIndex returns a list of `(anchor, data)` pairs instead of a dict. Dense aggregations raise TypeError on indices that are not a raster / grid placement.
* **patcher/sampler:** invalid step / jitter / n_samples / min_dist / max_tries / k raise ValueError at construction; SpatialPoissonDisk and SpatialJitteredStride draw different anchors for the same seed; SpatialAlongTrack(spacing=...) may yield one more (final) point; SpatialSphericalCap on a GridDomain returns a masked {dim: slice} window instead of an argwhere array; SpatialGeometry.extent is removed; a TypeError raised by a window's weights now propagates.
* **patcher/boundary:** non-drop samplers emit fewer anchors (no trailing anchor past the edge-reaching one; random/jittered/Poisson anchors are bounded by L - P); drop with a patch larger than the domain yields no patches; reflect no longer raises when the overflow exceeds the in-domain extent; shrink windows never start at a negative offset; an unrepresentable pad_value raises ValueError and a non-numeric one TypeError.
* **patcher/window:** SpatialHann weights change from symmetric np.hanning(n) to periodic scipy.signal.windows.hann(n, sym=False) (e.g. n=8 trailing sample 0 -> 0.146); Hann/Tukey axes of length < 3 are now all ones. Merged outputs using SpatialHann change accordingly.

### Features

* **patcher:** add merge_to_field and route reduce/two_pass/amerge through split ([#349](https://github.com/jejjohnson/geotoolz/issues/349)) ([94ddc2b](https://github.com/jejjohnson/geotoolz/commit/94ddc2b03fa1d0dfd5795a31750bcba5cdb9c39f)), closes [#194](https://github.com/jejjohnson/geotoolz/issues/194)


### Bug Fixes

* **patcher/aggregation:** dense aggregations honour masks, fill uncovered cells with nan ([#351](https://github.com/jejjohnson/geotoolz/issues/351)) ([6fc8081](https://github.com/jejjohnson/geotoolz/commit/6fc8081899e318b0c27b676a5fcc11436dd8e2e7))
* **patcher/aggregation:** stream overlap-add normalisation chunk-wise, write real cogs ([#352](https://github.com/jejjohnson/geotoolz/issues/352)) ([34a4f64](https://github.com/jejjohnson/geotoolz/commit/34a4f64c674ad615b9c7a2042fe60491e2c7c0c2))
* **patcher/boundary:** make every boundary mode survive split and merge ([#343](https://github.com/jejjohnson/geotoolz/issues/343)) ([b0f83cd](https://github.com/jejjohnson/geotoolz/commit/b0f83cdd9d19de1628c860780296e4ae4e420aa1)), closes [#185](https://github.com/jejjohnson/geotoolz/issues/185)
* **patcher/geometry:** pixel-align polygon intersection windows and masks ([#341](https://github.com/jejjohnson/geotoolz/issues/341)) ([2293849](https://github.com/jejjohnson/geotoolz/commit/2293849233cbb96901a0b06e9a46f2bc3cd5ca2c))
* **patcher/sampler:** exact poisson-disk spacing, uniform jitter, reachable geometry combos ([#344](https://github.com/jejjohnson/geotoolz/issues/344)) ([8fcc141](https://github.com/jejjohnson/geotoolz/commit/8fcc141b09230727999f569777f17212caf47f95)), closes [#187](https://github.com/jejjohnson/geotoolz/issues/187)
* **patcher/window:** periodic hann and tukey with one documented taper convention ([#342](https://github.com/jejjohnson/geotoolz/issues/342)) ([e31258c](https://github.com/jejjohnson/geotoolz/commit/e31258cee03363064771bc30b0c5e0e13c4eab11)), closes [#186](https://github.com/jejjohnson/geotoolz/issues/186)

## [0.4.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-patcher-v0.3.0...geotoolz-patcher-v0.4.0) (2026-10-02)


### ⚠ BREAKING CHANGES

* **patcher/fields:** ObstoreCogField.select / select_many return georeader GeoTensor chips instead of bare numpy arrays (use `.values` for the array). Out-of-image pixels are filled with the COG's nodata instead of 0, and fractional windows are snapped outward to whole pixels.
* **patcher/fields:** RioXarrayField.select and DaskField.select return an xarray.DataArray instead of a RioXarrayField / DaskField; drop the `.da` / `.array` hop on chips (e.g. patch.data.values instead of patch.data.da.values).
* **objstore:** geotoolz.readers._src.obstore and geocatalog._src.objstore are removed (use geopatcher.objstore; the geotoolz read_byte_range helper is now get_range_bytes). The [obstore] extras of geotoolz and geotoolz-catalog install geotoolz-patcher. Pool keys changed shape, and object_key / get_obstore raise ValueError for malformed Azure URIs and abfs:// URIs without container@account.
* **patcher/fields:** XvecField(_geometry_dim=...) is now XvecField(geometry_dim=...), with no alias. GeoPandasField.select no longer accepts row labels. GridDomain.bounds values are no longer coerced to float. ReprojectingRasterField.domain.shape includes the source's leading dims.

### Bug Fixes

* **objstore:** one obstore pool in geopatcher with correct Azure and signed-URL handling ([#334](https://github.com/jejjohnson/geotoolz/issues/334)) ([946a514](https://github.com/jejjohnson/geotoolz/commit/946a51466a3a73d71ecc57f984cf44c672585d55)), closes [#180](https://github.com/jejjohnson/geotoolz/issues/180)
* **patcher/dask:** make from_zarr, to_dask_bag and to_delayed work with real dask ([#336](https://github.com/jejjohnson/geotoolz/issues/336)) ([b24717f](https://github.com/jejjohnson/geotoolz/commit/b24717f2b1882117497ddc31dd6df5c62846521b)), closes [#179](https://github.com/jejjohnson/geotoolz/issues/179)
* **patcher/fields:** datetime grid bounds, N-D reprojection, cached domains ([#333](https://github.com/jejjohnson/geotoolz/issues/333)) ([a54b9cf](https://github.com/jejjohnson/geotoolz/commit/a54b9cf0528847f94b7744be0d02e60ea20ea30f)), closes [#182](https://github.com/jejjohnson/geotoolz/issues/182)
* **patcher/fields:** make ObstoreCogField honour the Field contract for real COGs ([#337](https://github.com/jejjohnson/geotoolz/issues/337)) ([a73456e](https://github.com/jejjohnson/geotoolz/commit/a73456ec377ad0112910ecef0b5b557126adaab1)), closes [#181](https://github.com/jejjohnson/geotoolz/issues/181)
* **patcher/fields:** materialise RasterField chips and keep nodata/attrs in with_data ([#332](https://github.com/jejjohnson/geotoolz/issues/332)) ([edeab87](https://github.com/jejjohnson/geotoolz/commit/edeab877497be0d007f2558a90f9bc2037085b9f))
* **patcher/fields:** rioxarray and dask chips are georeferenced DataArrays ([#335](https://github.com/jejjohnson/geotoolz/issues/335)) ([f921cae](https://github.com/jejjohnson/geotoolz/commit/f921cae99b3a460d018b559e3ce14148029dd7ef)), closes [#178](https://github.com/jejjohnson/geotoolz/issues/178)

## [0.3.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-patcher-v0.2.0...geotoolz-patcher-v0.3.0) (2026-10-02)


### ⚠ BREAKING CHANGES

* **geotoolz:** renamed / removed public API (old -> new), no aliases:
    - keyword-only constructors: all 15 augment operators (e.g.
      `Compose([...])` -> `Compose(augmentations=[...])`,
      `RandomCrop((4, 4))` -> `RandomCrop(size=(4, 4))`);
      `learn.SklearnOp(est)` -> `SklearnOp(estimator=est)`;
      `learn.ModelOp(m)` -> `ModelOp(model=m)`;
      geopatcher `GridSampler(p)` -> `GridSampler(patcher=p)`,
      `ApplyToChips(op)` -> `ApplyToChips(operator=op)`,
      `Stitch(agg, domain=d)` / `patch_ops.MergePatches(agg, domain=d)` ->
      `(aggregation=agg, domain=d)`
    - learn wrappers (also top-level `gz.*`): PCA -> PixelwisePCA,
      IPCA -> PixelwiseIPCA, NMF -> PixelwiseNMF, KMeans -> PixelwiseKMeans,
      MiniBatchKMeans -> PixelwiseMiniBatchKMeans, GMM -> PixelwiseGMM,
      IsolationForest -> PixelwiseIsolationForest,
      OneClassSVM -> PixelwiseOneClassSVM,
      LocalOutlierFactor -> PixelwiseLocalOutlierFactor,
      KNNImputer -> PixelwiseKNNImputer,
      IterativeImputer -> PixelwiseIterativeImputer
    - indices (all 20 operators; ctor and config keys): `a_idx`/`b_idx` -> `a`/`b`
      (NormalizedDifference, now required); `blue_idx`, `green_idx`, `red_idx`,
      `red_edge1_idx`, `red_edge2_idx`, `nir_idx`, `swir_idx`, `swir1_idx`,
      `swir2_idx`, `cirrus_idx` -> `blue`, `green`, `red`, `red_edge1`,
      `red_edge2`, `nir`, `swir`, `swir1`, `swir2`, `cirrus` (same defaults)
    - axis -> reduce_axes: radiometry.PercentileClip, viz.StretchToUint8,
      radiometry.percentile_clip, radiometry.dos1, viz.stretch_to_uint8,
      normalize.per_band_stats / standard_scale / robust_scale / minmax_scale
    - axis -> direction: restore.DestripeColumn, restore.destripe_column,
      geom.SegmentStitch
    - channel_axis -> axis: segment.Felzenszwalb, Quickshift, SLIC
    - fill -> fill_value: augment.BandDropout, geom.PadTo, geom.Stitch,
      geom.Rasterize, geom.RasterizeLike, geom.SegmentStitch
    - fill -> strategy: restore.ReplaceOutliers, restore.replace_outliers
    - random_state -> seed: matched_filter.EstimateCovLowRank,
      GMMClusterBackground, estimate_cov_lowrank, gmm_cluster_background
    - window_size -> window: matched_filter.AdaptiveWindowBackground,
      adaptive_window_background
    - size -> window: restore.MedianDenoise, restore.median_denoise
    - patch_size / patch_distance -> window / search_radius: restore.NLMeans,
      restore.nl_means
    - kernel_size -> window: normalize.CLAHE, normalize.clahe
    - min_area -> min_area_px: measure.LabelConnectedComponents,
      plume.PlumeMask, PlumeContours, PlumeShapeFilter, plume.plume_mask,
      geom.Vectorize
    - min_size -> min_area_px: mask.RemoveSmallObjects,
      mask.remove_small_objects, segment.Felzenszwalb
    - area_threshold -> max_hole_area_px: mask.RemoveSmallHoles,
      mask.remove_small_holes
    - min_object_size / max_hole_size -> min_area_px / max_hole_area_px:
      mask.CleanMask, mask.clean_mask
    - segment.Watershed `connectivity` 1 | 2 -> 4 | 8 (default 4, the same
      neighbourhood as the old default 1)
    - spectral.SelectBands `indexes` -> `bands`
    - augment.AtmosphericHaze reads only `attrs["wavelengths"]` (nm);
      `attrs["wavelengths_nm"]` is no longer consulted

### Code Refactoring

* **geotoolz:** keyword-only constructors and one parameter vocabulary ([#324](https://github.com/jejjohnson/geotoolz/issues/324)) ([39626e5](https://github.com/jejjohnson/geotoolz/commit/39626e5266b7e824a2c849f7d34fd384dfbad61f))

## [0.2.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-patcher-v0.1.0...geotoolz-patcher-v0.2.0) (2026-09-30)


### ⚠ BREAKING CHANGES

* **geotoolz:** geotoolz.patch_ops.Stitch is removed; use geotoolz.patch_ops.MergePatches (the same class as geopatcher.integrations.pipekit.Stitch). The old name collided with geotoolz.geom.Stitch / top-level geotoolz.Stitch. patch_ops.Stitch is now forbid_in_yaml and emits `domain` in get_config(); geopatcher.integrations.pipekit.GridSampler (and patch_ops.GridSampler) is now forbid_in_yaml. SpatialTriangular.weights returns float64 instead of float32.

### Code Refactoring

* **geotoolz:** re-export patch_ops bridge from geopatcher, rename Stitch to MergePatches ([#305](https://github.com/jejjohnson/geotoolz/issues/305)) ([380e115](https://github.com/jejjohnson/geotoolz/commit/380e115f0e5915f4af3c601d76a5fc9b581ae136)), closes [#156](https://github.com/jejjohnson/geotoolz/issues/156)

## [0.1.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-patcher-v0.0.6...geotoolz-patcher-v0.1.0) (2026-07-13)


### ⚠ BREAKING CHANGES

* merge geopatcher + geocatalog into a uv workspace monorepo ([#97](https://github.com/jejjohnson/geotoolz/issues/97))

### Features

* merge geopatcher + geocatalog into a uv workspace monorepo ([#97](https://github.com/jejjohnson/geotoolz/issues/97)) ([202286c](https://github.com/jejjohnson/geotoolz/commit/202286c2e3ea976a18142bdbaba906211e17cba1))

## [0.0.6](https://github.com/jejjohnson/geopatcher/compare/v0.0.5...v0.0.6) (2026-07-11)


### Features

* **spatial_time:** thread coord= through SpatioTemporalPatcher; fix zarr sharding ([#68](https://github.com/jejjohnson/geopatcher/issues/68)) ([dcf18ea](https://github.com/jejjohnson/geopatcher/commit/dcf18ea060c6d36d42ba66636fc700fa0ce10d45))
* **spatial:** SpatialAlongTrack sampler + PointDomain nearest/bilinear sampling ([#67](https://github.com/jejjohnson/geopatcher/issues/67)) ([4ae0054](https://github.com/jejjohnson/geopatcher/commit/4ae00546b98d74e6bdc1338becf7b46d7ae080db))


### Bug Fixes

* **tests:** add missing rasterio and GeoTensor imports in test_operational_scale ([#70](https://github.com/jejjohnson/geopatcher/issues/70)) ([aabc10b](https://github.com/jejjohnson/geopatcher/commit/aabc10bd83292e4fa67decbd25b25471f9320bcd))

## [0.0.5](https://github.com/jejjohnson/geopatcher/compare/v0.0.4...v0.0.5) (2026-06-03)


### Features

* **ml:** xrpatcher → geopatcher port (indexed view + cache + check_full_scan + xarray reconstruct) ([#61](https://github.com/jejjohnson/geopatcher/issues/61)) ([4afa248](https://github.com/jejjohnson/geopatcher/commit/4afa248f25846f442c2bba3530d42bbbedbe4c1a)), closes [#60](https://github.com/jejjohnson/geopatcher/issues/60)

## [0.0.4](https://github.com/jejjohnson/geopatcher/compare/v0.0.3...v0.0.4) (2026-06-03)


### Features

* **fields:** obstore COG field + batched parallel_map via select_many duck-typing ([#53](https://github.com/jejjohnson/geopatcher/issues/53)) ([1d12ce2](https://github.com/jejjohnson/geopatcher/commit/1d12ce27c58b0d25ef05c443b1d9839bdf4a5ee2))
* **time:** coordinate-aware temporal patching via TimeStencil (closes [#56](https://github.com/jejjohnson/geopatcher/issues/56)) ([#57](https://github.com/jejjohnson/geopatcher/issues/57)) ([351050b](https://github.com/jejjohnson/geopatcher/commit/351050b40584f6f47b289b05e6c1730cf3b3d44a))

## [0.0.3](https://github.com/jejjohnson/geopatcher/compare/v0.0.2...v0.0.3) (2026-05-25)


### Features

* add pipekit Operator integration behind [pipekit] extra ([#35](https://github.com/jejjohnson/geopatcher/issues/35)) ([2fda4eb](https://github.com/jejjohnson/geopatcher/commit/2fda4eb24573a9622cd7291e5a83475980ddd224))
* **geometry:** first-class boundary policy on SpatialRectangular ([#38](https://github.com/jejjohnson/geopatcher/issues/38)) ([54eaf79](https://github.com/jejjohnson/geopatcher/commit/54eaf792b368fd65e622fecad74eca7240c173c1))
* lock foundation ADRs, add strict mode + n_anchors() ([#37](https://github.com/jejjohnson/geopatcher/issues/37)) ([f1e7dc9](https://github.com/jejjohnson/geopatcher/commit/f1e7dc9bde9eec4ca27a34c47832455401f25405))
* **matched:** composite Field + MatchedPatch carrier (ADR-003) ([#48](https://github.com/jejjohnson/geopatcher/issues/48)) ([ca989a7](https://github.com/jejjohnson/geopatcher/commit/ca989a709f63e3e7409b20e5bfcb52c5b34b29a8))
* **matched:** implement MatchedField.select + MatchedSpatialPatcher ([#49](https://github.com/jejjohnson/geopatcher/issues/49)) ([a8d4da3](https://github.com/jejjohnson/geopatcher/commit/a8d4da3ad461f0d94850cbd9a0b52d8299543712))
* **matched:** MatchedTemporalPatcher + MatchedSpatioTemporalPatcher ([#51](https://github.com/jejjohnson/geopatcher/issues/51)) ([1b5a8b7](https://github.com/jejjohnson/geopatcher/commit/1b5a8b73acce0b4860d5030bc589253eb2b79dc1))
* ml primitives and torch/jax/grain recipe notebooks ([#23](https://github.com/jejjohnson/geopatcher/issues/23)) ([#41](https://github.com/jejjohnson/geopatcher/issues/41)) ([5e4c505](https://github.com/jejjohnson/geopatcher/commit/5e4c505152878503a47595cbdb3e7f123adb5832))
* **patcher:** async + parallel patching helpers (asplit, prefetch, dask, batched) ([#42](https://github.com/jejjohnson/geopatcher/issues/42)) ([8fe56d0](https://github.com/jejjohnson/geopatcher/commit/8fe56d0e6a6110581109b5c60d81931fb9114c71))
* **patcher:** observability hooks (split/patch/merge/error) ([#43](https://github.com/jejjohnson/geopatcher/issues/43)) ([48a0294](https://github.com/jejjohnson/geopatcher/commit/48a02944cef0577bb46b1dbce5d3b845f967f07f))
* **patcher:** on_error policy (raise/skip/mask/retry) on spatial patchers ([#44](https://github.com/jejjohnson/geopatcher/issues/44)) ([0c7c9fd](https://github.com/jejjohnson/geopatcher/commit/0c7c9fd37b581f0ec0a02e84439b25400f92eb4a))
* **patcher:** operational-scale primitives (journal, in-flight bounds, sketches, cog/zarr writers) ([#45](https://github.com/jejjohnson/geopatcher/issues/45)) ([fe089b2](https://github.com/jejjohnson/geopatcher/commit/fe089b2783288bc3dd097b1d0d648b13c0850381))
* **patcher:** parallel_map + two-pass / reduce primitives ([#46](https://github.com/jejjohnson/geopatcher/issues/46)) ([29f70d8](https://github.com/jejjohnson/geopatcher/commit/29f70d872676897445d0c113163988fcc3707e61))

## [0.0.2](https://github.com/jejjohnson/geopatcher/compare/v0.0.1...v0.0.2) (2026-05-16)


### Features

* port four-axis Patcher framework from geotoolz, clean up template ([#6](https://github.com/jejjohnson/geopatcher/issues/6)) ([e6e22b3](https://github.com/jejjohnson/geopatcher/commit/e6e22b31073e09ce6359e16b562a6778c61474c2))

## Changelog

All notable changes to this project will be documented in this file.

See [Conventional Commits](https://www.conventionalcommits.org/) for commit guidelines.
