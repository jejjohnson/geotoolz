# Changelog

## [0.9.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-patcher-v0.8.0...geotoolz-patcher-v0.9.0) (2026-10-08)


### ⚠ BREAKING CHANGES

* geopatcher.temporal.stencils.divide_evenly is now exact_quotient, geocatalog.grid.divide_evenly is now count_steps, and geotoolz.patch_ops.SpatialTriangular is now TriangularWindow. No aliases; saved patcher configs naming geotoolz.patch_ops.SpatialTriangular must be regenerated.
* **patcher:** no aliases are kept. The prefixed axis names (SpatialHann, TemporalMean, ...) and the root re-exports of axes, runners, hooks and caches are gone; geopatcher.time is geopatcher.temporal; geopatcher.objstore / geopatcher.cog become geocloud.store / geocloud.cog; geopatcher.runners / dask / jax / hooks fold into geopatcher.run and geopatcher.observe; ObstoreCogField becomes geopatcher.fields.CogField; the patcher extras obstore / obstore-cog are replaced by [cog]; and geotoolz-catalog[obstore] is removed (install geotoolz-cloud). Saved config envelopes that use the old class names must be regenerated.

### Code Refactoring

* give the divide-evenly helpers distinct names and drop the last Spatial* prefix ([#434](https://github.com/jejjohnson/geotoolz/issues/434)) ([d8155d2](https://github.com/jejjohnson/geotoolz/commit/d8155d2f96439c3339d93fae219b82103ea6ed75))
* **patcher:** organise geopatcher by task and split object storage into geotoolz-cloud ([#431](https://github.com/jejjohnson/geotoolz/issues/431)) ([a7a4d6e](https://github.com/jejjohnson/geotoolz/commit/a7a4d6e4fbec075260f8f4a9fd371a791ff13b30))

## [0.8.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-patcher-v0.7.1...geotoolz-patcher-v0.8.0) (2026-10-08)


### ⚠ BREAKING CHANGES

* **patcher:** the [obstore-cog] extra now installs async-geotiff>=0.5,<0.6 (which requires numpy>=2) instead of async-tiff directly, and the ObstoreCogField dataclass holds the async-geotiff level (`level`) and the pixel `dtype` instead of the raw async-tiff `tiff` handle. Fields opened through from_url are unaffected.

### Features

* **patcher:** async COG reading — AsyncCogReader and geopatcher.cog read_* coroutines ([#417](https://github.com/jejjohnson/geotoolz/issues/417)) ([ac66772](https://github.com/jejjohnson/geotoolz/commit/ac66772cc40be3f3fb05205b1cf2c8c1c64214e2))
* **patcher:** ObstoreCogField reads through async-geotiff with concurrent tile groups ([#416](https://github.com/jejjohnson/geotoolz/issues/416)) ([4fa4da0](https://github.com/jejjohnson/geotoolz/commit/4fa4da0bb315bee4497b946bbe5d47500952ddfb))

## [0.7.1](https://github.com/jejjohnson/geotoolz/compare/geotoolz-patcher-v0.7.0...geotoolz-patcher-v0.7.1) (2026-10-05)


### Bug Fixes

* **patcher:** docs and facades describe the shipped surface; lint docs for retired names ([#411](https://github.com/jejjohnson/geotoolz/issues/411)) ([987495b](https://github.com/jejjohnson/geotoolz/commit/987495b42a57d36a399406e6f171b5182559e703)), closes [#205](https://github.com/jejjohnson/geotoolz/issues/205)
* **patcher:** extras install exactly what geopatcher uses; report the distribution version ([#409](https://github.com/jejjohnson/geotoolz/issues/409)) ([92ea356](https://github.com/jejjohnson/geotoolz/commit/92ea356d67f793f629a9e8e0bfb12ec628f02d33))

## [0.7.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-patcher-v0.6.0...geotoolz-patcher-v0.7.0) (2026-10-05)


### ⚠ BREAKING CHANGES

* **patcher/matched:** `MatchedSpatioTemporalPatcher` members carry each source's sliced carrier (e.g. `GeoTensor`) instead of an ndarray; chip read failures follow the spatial patcher's `on_error` policy (the coupled mode used to raise); `on_split_start` reports `-1` in product mode (the spatial anchor count before); coupled chips get the spatial pipeline's padded / reflected data and masked / cropped weights.
* **patcher:** `SpatioTemporalPatcher` patches carry the chip's carrier (e.g. `GeoTensor`) sliced along `time_axis` instead of an ndarray, and the coupled split's weights are the chip's (masked / shrink-cropped) weights; chip reads follow the spatial patcher's `on_error` policy; product mode reports `on_split_start(-1)`; `asplit` accepts `hooks` positionally. `on_error` hooks get the original exception instead of a `RuntimeError` summary for swallowed failures. `patcher.errors` is reset by each split / asplit / reduce / two_pass instead of accumulating across calls. `PatchErrorRecord` lives in `geopatcher._src.walk` (`geopatcher.PatchErrorRecord` is unchanged).
* **patcher/time:** TemporalPatcher methods take `time_axis` / `hooks` / `coord` as keyword-only arguments; TemporalRandom(n=) is now n_samples=; TemporalEventTriggered(event_times=) is TemporalExplicit(times=) and TemporalCausalRolling is TemporalRegularStride; TemporalTaperedTukey weights change (one-sided causal taper); TemporalStencilGeometry drops overflowing origins by default (boundary="drop") instead of raising; coord= must be strictly increasing; Stencil / divide_evenly reject inputs they used to accept and fail on later; a non-GridDomain Field passed to TemporalPatcher raises TypeError.
* **patcher/time:** integer temporal geometries default to boundary="drop": anchors whose window overflows the time axis yield no patch (pass boundary="shrink" for the old clamped edge windows). TemporalMean now reconstructs a per-time-step mean instead of averaging stacked patches; TemporalHierarchicalCombine keys inner dicts by scale (or window index), never a (start, stop) tuple; TemporalPhaseWindow yields one patch per cycle; TemporalForecast skips windows without a full horizon.
* **patcher:** GridSampler.get_config()["patcher"] is now a {"class": "SpatialPatcher", "config": ...} envelope instead of the bare patcher config.
* **patcher:** get_config payloads changed shape — stencil axes nest {"class", "config"} envelopes with {"value", "unit"} offsets, SpatioTemporalPatcher nests envelopes, retry_on is emitted as qualified names, and the summary keys (n_points, n_coords, n_events, n_times) are replaced by the data. SpatialExplicit and SpatialPolygonIntersection are forbid_in_yaml; PatchCache refuses SpatialPolygonIntersection. A retry_on string now also matches subclasses of the class it names.
* **patcher/matched:** MatchedTemporalPatcher.split / n_anchors / anchors read the series with an indexer derived from the primary's domain (or the new indexer= argument) instead of slice(None); a secondary whose time length differs from the primary's raises ValueError. The matched merges run each source's aggregation concurrently on a worker thread and close each MatchedPatch as it is consumed. Matched hooks fire once at the matched level: MatchedTemporalPatcher no longer forwards split hooks to the primary TemporalPatcher, on_patch_done / on_merge_end bytes sum every source, and a secondary aggregation failure now reaches on_error. on_error="mask" yields an all-NaN MatchedPatch instead of raising TypeError. MatchedPatch.weights is no longer always None.
* **patcher/runners:** parallel_map's process backend no longer forks; it uses forkserver (spawn where unavailable), so operators must be picklable module-level callables importable in a fresh interpreter (pass mp_context="fork" to opt back in). Read failures on the select_many path are now governed by SpatialPatcher.on_error, not by parallel_map(on_error=...), which applies to operator errors only. BatchedPatch gains a carriers field and unbatch returns GeoTensor / DataArray patches for carrier chips instead of bare arrays.

### Bug Fixes

* **patcher/backpressure:** lazy byte budget, interruptible prefetch, on-loop async release ([#382](https://github.com/jejjohnson/geotoolz/issues/382)) ([cddeb78](https://github.com/jejjohnson/geotoolz/commit/cddeb78dce9cb3baed7f8026d276f6c3404747dd)), closes [#195](https://github.com/jejjohnson/geotoolz/issues/195)
* **patcher/matched:** real-field temporal indexer, streaming per-source merge, on_error and hook parity ([#385](https://github.com/jejjohnson/geotoolz/issues/385)) ([d27fafc](https://github.com/jejjohnson/geotoolz/commit/d27fafc09c2318a0c72799861e3756ebb929980d))
* **patcher/runners:** stream parallel_map through split, explicit mp start method, journal ([#383](https://github.com/jejjohnson/geotoolz/issues/383)) ([a4e63a3](https://github.com/jejjohnson/geotoolz/commit/a4e63a3b79f526194d8cce73a5af25395fb28888)), closes [#196](https://github.com/jejjohnson/geotoolz/issues/196)
* **patcher/time:** temporal geometries drop overflowing windows instead of clamping ([#394](https://github.com/jejjohnson/geotoolz/issues/394)) ([f8aabf5](https://github.com/jejjohnson/geotoolz/commit/f8aabf506944b8fd73bbe70364d1bdcd6f736e11))
* **patcher:** envelope GridSampler's patcher and use Patch.with_data in matched split ([#393](https://github.com/jejjohnson/geotoolz/issues/393)) ([21dcc84](https://github.com/jejjohnson/geotoolz/commit/21dcc849e2193cdcd973f9c98811ba51299a3b3c)), closes [#201](https://github.com/jejjohnson/geotoolz/issues/201)
* **patcher:** make every get_config rebuild its object or flag forbid_in_yaml ([#392](https://github.com/jejjohnson/geotoolz/issues/392)) ([c693752](https://github.com/jejjohnson/geotoolz/commit/c69375297ed8085ce067d7592ea4a6a80ae5a247)), closes [#200](https://github.com/jejjohnson/geotoolz/issues/200)


### Code Refactoring

* **patcher/matched:** drive SpatioTemporalPatcher instead of forking it ([#408](https://github.com/jejjohnson/geotoolz/issues/408)) ([c21fdb7](https://github.com/jejjohnson/geotoolz/commit/c21fdb7500960fde976dddc60400423b5d186494))
* **patcher/time:** TemporalPatcher parity with SpatialPatcher ([#406](https://github.com/jejjohnson/geotoolz/issues/406)) ([f12cdb1](https://github.com/jejjohnson/geotoolz/commit/f12cdb18b8d813c1e09ddc466a9475d270c8daa9))
* **patcher:** one anchor-walk core for every split path ([#407](https://github.com/jejjohnson/geotoolz/issues/407)) ([57fe7d3](https://github.com/jejjohnson/geotoolz/commit/57fe7d33f83b33a81a06899d50b72c7fa90b9dc6))

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
