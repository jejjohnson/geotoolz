# Changelog

## [0.6.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-v0.5.1...geotoolz-v0.6.0) (2026-10-05)


### ⚠ BREAKING CHANGES

* **geotoolz:** the carrier constructor kwargs are removed with no aliases: IMEEstimate / CrossSectionalFlux(plume_mask=), PlumeFootprint(enhancement=), SBMP(reference_scene=), PlumeColumnStats(column=), PlumeQNDFeatures(column=, albedo=), RegionProps(intensity_image=), ApplyMask(mask=<array>), AltitudeMask / SlopeMask(dem=), SLIC / Felzenszwalb / Quickshift / Watershed(mask=), Watershed / RandomWalker(markers=), MarkBoundaries(label_img=), PhaseAlign / OpticalFlow*(reference=), CutMix(pool=) and BlendMatched(..., variances=). Pass the carriers at call time. Multi-input grid errors now read "<Op>: the <x> pixel grid (...) does not match the <y> pixel grid (...)".
* **patcher:** GridSampler.get_config()["patcher"] is now a {"class": "SpatialPatcher", "config": ...} envelope instead of the bare patcher config.
* **geotoolz:** PhaseAlign(apply=True) marks edge pixels uncovered by the shift, and pixels interpolated from nodata, as nodata (the fill value, or NaN), where it used to repeat edge pixels and blend fill values in. Its estimated shift ignores nodata pixels. Rasterize / RasterizeLike declare an integer fill on integer burns and write it into the input's nodata pixels. RasterizeLike rejects an input that is not on `like`'s grid. A GeoDataFrame burned without `column` is uint8, not float64. HistogramMatch on a (T, C, H, W) stack matches per band, pooled over frames.
* **geotoolz:** Hillshade, ShadedRelief and SlopeMask raise ValueError on a geographic-CRS DEM (pass explicit Hillshade resolutions or reproject); DistanceMask, BufferMask(unit="meters"), SlopeMask and Hillshade raise ValueError on a sheared transform; rotated grids now give different (correct) distance and slope results.
* **geotoolz:** restore.InverseMNF is removed; use MNF.inverse. Calling MNF reuses the basis fitted on the first call instead of refitting on every call (call fit() to refit). Learned statistics are stored in mean_ / std_ / median_ / iqr_ / vmin_ / vmax_ (scalers), mean_ / cov_op_ (MatchedFilter) and MNF.state_ (was the private _state), and are no longer written to the constructor attributes or get_config(); persist a fit by passing it back as constructor arguments. MatchedFilter(fit_on_call=True) no longer stores the per-call background. SklearnOp(fit_mode="fit_streaming") with a fit_predict task raises ValueError. ModelOp raises TypeError at construction for a non-Predictor model with method="predict", a non-callable model with method="__call__", or a missing method.

### Features

* **geotoolz:** fitted operators expose fit/transform seams ([#389](https://github.com/jejjohnson/geotoolz/issues/389)) ([d577a85](https://github.com/jejjohnson/geotoolz/commit/d577a8571e0c0526e66fb74239d91dbb7ca4ae8f)), closes [#143](https://github.com/jejjohnson/geotoolz/issues/143)


### Bug Fixes

* **geotoolz:** close the [#166](https://github.com/jejjohnson/geotoolz/issues/166) fill and 4-d contract gaps ([#391](https://github.com/jejjohnson/geotoolz/issues/391)) ([3995060](https://github.com/jejjohnson/geotoolz/commit/399506095e9dc50a2cfb04e4d61f582df8cd8738)), closes [#331](https://github.com/jejjohnson/geotoolz/issues/331)
* **geotoolz:** ground pixel steps on rotated grids, reject geographic-crs slopes ([#390](https://github.com/jejjohnson/geotoolz/issues/390)) ([3a9e1ee](https://github.com/jejjohnson/geotoolz/commit/3a9e1eed301be71caf4625e34cd307cc30ea0d75)), closes [#331](https://github.com/jejjohnson/geotoolz/issues/331)
* **geotoolz:** second carriers are positional _apply arguments ([#403](https://github.com/jejjohnson/geotoolz/issues/403)) ([2c347fd](https://github.com/jejjohnson/geotoolz/commit/2c347fd0646eefc106b5a2e28fa3a7080c26d7a5)), closes [#141](https://github.com/jejjohnson/geotoolz/issues/141)
* **patcher:** envelope GridSampler's patcher and use Patch.with_data in matched split ([#393](https://github.com/jejjohnson/geotoolz/issues/393)) ([21dcc84](https://github.com/jejjohnson/geotoolz/commit/21dcc849e2193cdcd973f9c98811ba51299a3b3c)), closes [#201](https://github.com/jejjohnson/geotoolz/issues/201)

## [0.5.1](https://github.com/jejjohnson/geotoolz/compare/geotoolz-v0.5.0...geotoolz-v0.5.1) (2026-10-03)


### Bug Fixes

* **catalog,patcher:** get_config()["crs"] is one string on both backends and across round trips; pool key covers every endpoint variable; hf:// in the pool ([#371](https://github.com/jejjohnson/geotoolz/issues/371)) ([6b8b3ff](https://github.com/jejjohnson/geotoolz/commit/6b8b3ff34489958ed89ff210a3e4abb7961dd501))

## [0.5.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-v0.4.0...geotoolz-v0.5.0) (2026-10-02)


### ⚠ BREAKING CHANGES

* **objstore:** geotoolz.readers._src.obstore and geocatalog._src.objstore are removed (use geopatcher.objstore; the geotoolz read_byte_range helper is now get_range_bytes). The [obstore] extras of geotoolz and geotoolz-catalog install geotoolz-patcher. Pool keys changed shape, and object_key / get_obstore raise ValueError for malformed Azure URIs and abfs:// URIs without container@account.

### Bug Fixes

* **objstore:** one obstore pool in geopatcher with correct Azure and signed-URL handling ([#334](https://github.com/jejjohnson/geotoolz/issues/334)) ([946a514](https://github.com/jejjohnson/geotoolz/commit/946a51466a3a73d71ecc57f984cf44c672585d55)), closes [#180](https://github.com/jejjohnson/geotoolz/issues/180)

## [0.4.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-v0.3.0...geotoolz-v0.4.0) (2026-10-02)


### ⚠ BREAKING CHANGES

* **packaging:** the base install no longer pulls in matplotlib, scikit-learn or joblib. Install `geotoolz[viz]` for `ApplyColormap`, and `geotoolz[learn]` for the `learn` imputer NaN strategies, `save_state` / `load_state` / `state_path=`, and plume's DBSCAN clump counting. The HDF / NetCDF reader ImportError message changed wording.
* **geotoolz:** removed / changed public API (old -> new), no aliases:
    - `geotoolz.model` module removed -> `geotoolz.learn.ModelOp` (or
      `gz.ModelOp`)
    - `geotoolz.plume.otsu_threshold`, `plume.resolve_threshold` ->
      `geotoolz.segment.otsu_threshold`, `segment.resolve_threshold`
    - `geotoolz.plume.label_components` -> `geotoolz.measure.label_components`
    - `SklearnOp.save_state` / `GeoTensorEstimator.save_state` no longer write
      the `<path>.meta.json` sidecar by default -> pass `write_meta=True`
    - `mask.CountryMask` with unknown `iso_a3` codes raises `ValueError` on its
      first call instead of in the constructor; LandMask / OceanMask /
      CountryMask download or read their source on first call, not at
      construction
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
* **geotoolz:** removed public API (no aliases): geotoolz.geom.coregister.SwathToGrid -> removed (always raised NotImplementedError); geotoolz.geom.coregister.GridToSwath -> removed (always raised NotImplementedError); geotoolz.readers.require_optional_dependency -> removed (unused; guard optional imports inline); geotoolz.readers.toy_sensor.ops -> removed (empty module). Private module paths moved: geotoolz.compositing._src.fusion -> geotoolz.compositing._src.operators; geotoolz.learn._src.model -> geotoolz.learn._src.operators; geotoolz.readers._base / ._constants / ._obstore -> geotoolz.readers._src.base / .constants / .obstore.
* **geotoolz:** DespeckleLee / DespeckleRefinedLee / DespeckleFrost, DestripeColumn(method="moment_matching") / MomentMatching and MNF / InverseMNF produce different (correct) outputs; MNF.snr_ is now λ − 1 of the noise-whitened eigenproblem; fit_pca's "snr" state key is now "explained_variance".
* **geotoolz:** BAPComposite scores (and hence selected pixels and the returned score) change for raw doy / view_angle / cloud_distance / opacity metadata; mixing raw and precomputed cloud-distance metadata no longer raises. wind_advection_cone / WindAdvectionCone with half_angle_deg > 90 now include pixels behind the crosswind line.
* **geotoolz:** default "ledoit_wolf" (and "oas") covariances change numerically everywhere (e.g. n=30, p=6 LW intensity 0.089 -> 0.214), so MatchedFilter / ColumnEnhancement / ApplyClusterMF scores, SNRs and thresholds change. shrink_covariance(method="ledoit_wolf") now requires fourth_moment=; WelfordAccumulator has new required fields m3 and m4.

### Bug Fixes

* **geotoolz:** Griffiths BAP scores, full-range wind cone, Otsu nbins threading ([#318](https://github.com/jejjohnson/geotoolz/issues/318)) ([3b3d822](https://github.com/jejjohnson/geotoolz/commit/3b3d82241666e28f87dbf5e9d1bb01c1c0fcaec3)), closes [#161](https://github.com/jejjohnson/geotoolz/issues/161)
* **geotoolz:** implement Lee, Frost, moment-matching destriping and MNF per their references ([#320](https://github.com/jejjohnson/geotoolz/issues/320)) ([ed6d8ed](https://github.com/jejjohnson/geotoolz/commit/ed6d8ed3ac8e2da9ea971fefaa676f7e20892ef0)), closes [#158](https://github.com/jejjohnson/geotoolz/issues/158)
* **geotoolz:** match sklearn Ledoit-Wolf/OAS shrinkage, fix GMM restart, add ApplyAdaptiveMF ([#317](https://github.com/jejjohnson/geotoolz/issues/317)) ([3691fc9](https://github.com/jejjohnson/geotoolz/commit/3691fc95ff85a644fa406a800f7370c09bce38e1)), closes [#159](https://github.com/jejjohnson/geotoolz/issues/159)


### Code Refactoring

* **geotoolz:** canonical two-tier layout for every family ([#323](https://github.com/jejjohnson/geotoolz/issues/323)) ([36c963c](https://github.com/jejjohnson/geotoolz/commit/36c963ce82c3dd579634b26ba3bae7c42074193f)), closes [#162](https://github.com/jejjohnson/geotoolz/issues/162)
* **geotoolz:** enforce the export policy, drop the model shim, lazy Natural Earth ([#325](https://github.com/jejjohnson/geotoolz/issues/325)) ([7f95f00](https://github.com/jejjohnson/geotoolz/commit/7f95f0029f4b3929746d1ad3365a255fd79aea87))
* **geotoolz:** keyword-only constructors and one parameter vocabulary ([#324](https://github.com/jejjohnson/geotoolz/issues/324)) ([39626e5](https://github.com/jejjohnson/geotoolz/commit/39626e5266b7e824a2c849f7d34fd384dfbad61f))


### Miscellaneous

* **packaging:** declare rasterio/affine, move matplotlib/sklearn/joblib to extras ([#327](https://github.com/jejjohnson/geotoolz/issues/327)) ([def9896](https://github.com/jejjohnson/geotoolz/commit/def9896fb89af1f5ccbf464e176f2bd56e419413))

## [0.3.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-v0.2.2...geotoolz-v0.3.0) (2026-09-30)


### ⚠ BREAKING CHANGES

* **geotoolz:** geotoolz.patch_ops.Stitch is removed; use geotoolz.patch_ops.MergePatches (the same class as geopatcher.integrations.pipekit.Stitch). The old name collided with geotoolz.geom.Stitch / top-level geotoolz.Stitch. patch_ops.Stitch is now forbid_in_yaml and emits `domain` in get_config(); geopatcher.integrations.pipekit.GridSampler (and patch_ops.GridSampler) is now forbid_in_yaml. SpatialTriangular.weights returns float64 instead of float32.
* **geotoolz:** geom.CropToBounds rounds the pixel window outward instead of flooring it, so pixels the bounds only partially cover are now included: bounds (12.4, 16.4, 15.6, 19.6) on a 1-degree grid now give (1, 4, 4) instead of (1, 3, 3). To keep the old footprint, snap the bounds to pixel edges before cropping.
* **geotoolz:** radiometry.PercentileClip / percentile_clip arguments p_min / p_max are renamed lower / upper; radiometry.Gamma / gamma_correct argument g is renamed gamma; viz.StretchToUint8 / stretch_to_uint8 argument per_band is replaced by axis ((-2, -1) = per band, None = global). Removed: viz.ToDisplayRange and gz.ToDisplayRange (use viz.StretchToUint8), viz.composite (use spectral.select_bands), normalize.percentile_clip (use radiometry.percentile_clip; note its defaults are 2 / 98, not 1 / 99), and the top-level gz.Compose export (use gz.augment.Compose). viz byte outputs are rounded instead of truncated. augment.CutMix rejects donors whose transform differs from the input's (previously only |a|, |e| and CRS were compared).
* **geotoolz:** Geometry/DEM masks now return True = drop; the inside= argument is replaced by keep="inside"|"outside" (default keeps the region, i.e. the old mask is inverted). ApplyMask(invert=) and apply_mask(invert=) are removed (use InvertMask / invert_mask). combine_masks/CombineMasks op="not" is removed (use invert_mask / InvertMask). qa.MaskValid is removed (use qa.MaskInvalid). qa.MaskFromQABits(band_idx=, bits=) and qa.MaskFromSCL(band_idx=, classes=) are removed (use qa.MaskClouds(qa_band=, bits=|values=, invert=) or another _QAMask shortcut). qa.S2SCL(keep=) is replaced by S2SCL(targets=) (names to mask out). qa.reduce_bit_masks is removed. restore.SaturationFlag and restore.saturation_flag are removed (use qa.MaskSaturated(saturation_value=, reduce_bands=False)). The geotoolz.cloud module is removed (import from geotoolz.qa / geotoolz.mask).
* **geotoolz:** spectral.NormalizedDifference is removed; use indices.NormalizedDifference (a=/b= accept names; config keys a_idx/b_idx; default eps 1e-10). spectral.ApplySRF and spectral.GaussianSRF (and the top-level geotoolz.GaussianSRF) are removed; use radiometry.ApplySRF (geotoolz.ApplySRF) with band_names= and source_wavelengths= (omit it to read attrs["wavelengths"]). The spectral.normalized_difference re-export is removed; use indices.normalized_difference. spectral.band_ratio / BandRatio default eps changed from 1e-6 to 1e-10. StackBands now raises on inputs with different fill_value_default. radiometry.ApplySRF raises ValueError for targets outside the source range or with no SRF support on the 1-nm grid.
* **geotoolz:** measure.LabelConnectedComponents(connectivity=) takes 4 | 8 (default 4) instead of skimage's 1 | 2 | None; use 4 for 1 and 8 for 2 / None. plume._src.operators.PLUME_REGIONPROPS is removed; use geotoolz.measure.DEFAULT_REGIONPROPS. The private measure._src.operators._skeleton_diameter_pixels, the private matched_filter._src.array.cube_to_samples(cube, axis=) and plume._src.array.connectivity_structure / _longest_active_pixel_path are removed; use geotoolz.measure.skeleton_length and geotoolz._src.samples.cube_to_samples(arr, band_axis=). measure.SkeletonLength returns Euclidean lengths (a 5-pixel diagonal is 4*sqrt(2), was 4).
* **geotoolz:** named-band lookup now reads attrs "band_names" before "descriptions" (indices/compositing used descriptions first; viz read "bands" first; qa/augment also read "band_descriptions", which is no longer a band-name source). Readers no longer write attrs["bands"] - read attrs["band_names"]. A string band reference on a plain ndarray now raises TypeError in every family (spectral, qa, augment and viz raised ValueError). plume.SBMP no longer maps S2 names on a GeoTensor without band-name attrs (name the bands, or pass integer indices), and its names are case-sensitive. augment.BandJitter no longer treats digit strings / floats as indices (pass ints). The geotoolz.indices._src.bands re-export module is removed - import from geotoolz._src.bands.

### Code Refactoring

* **geotoolz:** delegate CropToBounds to georeader and move coregister maths into array.py ([#157](https://github.com/jejjohnson/geotoolz/issues/157)) ([#311](https://github.com/jejjohnson/geotoolz/issues/311)) ([f6a7b4b](https://github.com/jejjohnson/geotoolz/commit/f6a7b4b620e0ce590314f795fbc13e2cc1333605))
* **geotoolz:** one band-name resolver for every operator family ([#150](https://github.com/jejjohnson/geotoolz/issues/150)) ([#307](https://github.com/jejjohnson/geotoolz/issues/307)) ([0112b65](https://github.com/jejjohnson/geotoolz/commit/0112b65a710aa70788cc3e3d29d9b2b38d81f5c4))
* **geotoolz:** one connected-components / skeleton / regionprops / pixel-flattening implementation ([#152](https://github.com/jejjohnson/geotoolz/issues/152)) ([#309](https://github.com/jejjohnson/geotoolz/issues/309)) ([1c8422f](https://github.com/jejjohnson/geotoolz/commit/1c8422fb59749d59d78d696cdded69e36c02cddb))
* **geotoolz:** one mask polarity, one invalid-mask, one QA decoder, one saturation op ([#154](https://github.com/jejjohnson/geotoolz/issues/154)) ([#312](https://github.com/jejjohnson/geotoolz/issues/312)) ([bd51458](https://github.com/jejjohnson/geotoolz/commit/bd51458deca1845f178da2779ee7b66decd25ca7))
* **geotoolz:** one NormalizedDifference, one ApplySRF, StackBands on grid_matches ([#310](https://github.com/jejjohnson/geotoolz/issues/310)) ([46a9e8a](https://github.com/jejjohnson/geotoolz/commit/46a9e8aa7b07a90864a97b46df9a88ff5c09e74b)), closes [#153](https://github.com/jejjohnson/geotoolz/issues/153)
* **geotoolz:** re-export patch_ops bridge from geopatcher, rename Stitch to MergePatches ([#305](https://github.com/jejjohnson/geotoolz/issues/305)) ([380e115](https://github.com/jejjohnson/geotoolz/commit/380e115f0e5915f4af3c601d76a5fc9b581ae136)), closes [#156](https://github.com/jejjohnson/geotoolz/issues/156)
* **geotoolz:** viz composites on SelectBands, one stretch/gamma vocabulary, Compose as a gated Sequential ([#155](https://github.com/jejjohnson/geotoolz/issues/155)) ([#313](https://github.com/jejjohnson/geotoolz/issues/313)) ([f6014c2](https://github.com/jejjohnson/geotoolz/commit/f6014c252cf98e74dbbc93aa37db1454f4eb2292))

## [0.2.2](https://github.com/jejjohnson/geotoolz/compare/geotoolz-v0.2.1...geotoolz-v0.2.2) (2026-09-29)


### Features

* **geotoolz:** valid_pixels helper and fill-pixel exclusion across families ([#298](https://github.com/jejjohnson/geotoolz/issues/298)) ([47ffe8c](https://github.com/jejjohnson/geotoolz/commit/47ffe8ce415ee7f92e7ed70d9cf7326c34f8df95)), closes [#145](https://github.com/jejjohnson/geotoolz/issues/145)


### Bug Fixes

* **geotoolz:** einx nodata-aware pooling, validation and 4-D presets ([#301](https://github.com/jejjohnson/geotoolz/issues/301)) ([6651a99](https://github.com/jejjohnson/geotoolz/commit/6651a998ba08ddee7894b78aa5a3b10848818cb6)), closes [#149](https://github.com/jejjohnson/geotoolz/issues/149)
* **geotoolz:** learn outputs are channel-first, named, and nodata-aware ([#300](https://github.com/jejjohnson/geotoolz/issues/300)) ([071be0d](https://github.com/jejjohnson/geotoolz/commit/071be0d27e6cfef29f51429a2b72192e50ab6638)), closes [#148](https://github.com/jejjohnson/geotoolz/issues/148)
* **geotoolz:** output fill values follow the output's dtype and meaning ([#299](https://github.com/jejjohnson/geotoolz/issues/299)) ([aaf0911](https://github.com/jejjohnson/geotoolz/commit/aaf0911221728ee73a3225474cf03d10d3d3d921)), closes [#146](https://github.com/jejjohnson/geotoolz/issues/146)
* **geotoolz:** resolve the band axis as -3 and handle 4-D time stacks ([#302](https://github.com/jejjohnson/geotoolz/issues/302)) ([5074bf2](https://github.com/jejjohnson/geotoolz/commit/5074bf2c9f26fa8b0bc3f5c43d25482c71b1fe2a)), closes [#147](https://github.com/jejjohnson/geotoolz/issues/147)
* **geotoolz:** wrap_like copies attrs and rewrites per-band metadata ([#297](https://github.com/jejjohnson/geotoolz/issues/297)) ([881c7f9](https://github.com/jejjohnson/geotoolz/commit/881c7f937adef0e80c85b0ba749b18402189d137)), closes [#144](https://github.com/jejjohnson/geotoolz/issues/144)

## [0.2.1](https://github.com/jejjohnson/geotoolz/compare/geotoolz-v0.2.0...geotoolz-v0.2.1) (2026-09-29)


### Bug Fixes

* **compositing,geom:** implement _apply instead of overriding __call__ ([#280](https://github.com/jejjohnson/geotoolz/issues/280)) ([3f79385](https://github.com/jejjohnson/geotoolz/commit/3f79385528dfc572b21c35bf3752eb989c1a0d3a)), closes [#135](https://github.com/jejjohnson/geotoolz/issues/135)
* **geotoolz:** align geom operators with their documented georeader delegation ([#129](https://github.com/jejjohnson/geotoolz/issues/129)) ([#295](https://github.com/jejjohnson/geotoolz/issues/295)) ([ecdc456](https://github.com/jejjohnson/geotoolz/commit/ecdc4565bcd330f999f93c3ee73e05ed958c0ca7))
* **geotoolz:** align readers with obstore pool keys and georeader ([#132](https://github.com/jejjohnson/geotoolz/issues/132)) ([#288](https://github.com/jejjohnson/geotoolz/issues/288)) ([d994867](https://github.com/jejjohnson/geotoolz/commit/d9948670c43eaeedab89b33568819edf7b7aba15))
* **geotoolz:** correct PhaseAlign georeferencing and parallax pixel-centre sampling ([#118](https://github.com/jejjohnson/geotoolz/issues/118), [#119](https://github.com/jejjohnson/geotoolz/issues/119)) ([#289](https://github.com/jejjohnson/geotoolz/issues/289)) ([2ac1a3e](https://github.com/jejjohnson/geotoolz/commit/2ac1a3e55d582ccf8338feeaacc70ea5f20bb274))
* **geotoolz:** einx input-side survival, hillshade orientation, float ensure_rgba ([#123](https://github.com/jejjohnson/geotoolz/issues/123), [#124](https://github.com/jejjohnson/geotoolz/issues/124), [#125](https://github.com/jejjohnson/geotoolz/issues/125)) ([#290](https://github.com/jejjohnson/geotoolz/issues/290)) ([cc6f97b](https://github.com/jejjohnson/geotoolz/commit/cc6f97bb870f0d5c49bdbecf9672a220e5e24b3d))
* **geotoolz:** exact flip/rot90 transforms and per-call augment streams ([#126](https://github.com/jejjohnson/geotoolz/issues/126), [#134](https://github.com/jejjohnson/geotoolz/issues/134)) ([#291](https://github.com/jejjohnson/geotoolz/issues/291)) ([2263b53](https://github.com/jejjohnson/geotoolz/commit/2263b5320c2d88a0ff0a6688f8bd1b781ba000c7))
* **geotoolz:** io reader/writer correctness fixes ([#127](https://github.com/jejjohnson/geotoolz/issues/127), [#128](https://github.com/jejjohnson/geotoolz/issues/128), [#130](https://github.com/jejjohnson/geotoolz/issues/130), [#131](https://github.com/jejjohnson/geotoolz/issues/131)) ([#294](https://github.com/jejjohnson/geotoolz/issues/294)) ([fc59788](https://github.com/jejjohnson/geotoolz/commit/fc59788b562a17faf6138a986d6f027136c755e3))
* **geotoolz:** promote integer inputs before band arithmetic; Canny keeps skimage thresholds ([#117](https://github.com/jejjohnson/geotoolz/issues/117), [#133](https://github.com/jejjohnson/geotoolz/issues/133)) ([#292](https://github.com/jejjohnson/geotoolz/issues/292)) ([f4a0a88](https://github.com/jejjohnson/geotoolz/commit/f4a0a88af574e10d703f96c7ec1eae2e8d5f8c9a))
* **geotoolz:** segment label/shape fixes; real skeleton plume length; plume ops require metric CRS ([#120](https://github.com/jejjohnson/geotoolz/issues/120), [#121](https://github.com/jejjohnson/geotoolz/issues/121), [#122](https://github.com/jejjohnson/geotoolz/issues/122)) ([#293](https://github.com/jejjohnson/geotoolz/issues/293)) ([54e26e0](https://github.com/jejjohnson/geotoolz/commit/54e26e0c6c26ad0b7b35ee8a40cdca45bc49e682))
* **geotoolz:** set forbid_in_yaml exactly where operators hold runtime state ([#284](https://github.com/jejjohnson/geotoolz/issues/284)) ([c7ce202](https://github.com/jejjohnson/geotoolz/commit/c7ce20297119ab96d74cea3d5e4338a460703e31))
* **geotoolz:** tuple and mapping configs survive a from_state reload ([#282](https://github.com/jejjohnson/geotoolz/issues/282)) ([9a6e9fc](https://github.com/jejjohnson/geotoolz/commit/9a6e9fc07a8c0d4d6e0233925781c99b52c81795)), closes [#139](https://github.com/jejjohnson/geotoolz/issues/139)

## [0.2.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-v0.1.1...geotoolz-v0.2.0) (2026-07-13)


### ⚠ BREAKING CHANGES

* merge geopatcher + geocatalog into a uv workspace monorepo ([#97](https://github.com/jejjohnson/geotoolz/issues/97))

### Features

* merge geopatcher + geocatalog into a uv workspace monorepo ([#97](https://github.com/jejjohnson/geotoolz/issues/97)) ([202286c](https://github.com/jejjohnson/geotoolz/commit/202286c2e3ea976a18142bdbaba906211e17cba1))

## [0.1.1](https://github.com/jejjohnson/geotoolz/compare/v0.1.0...v0.1.1) (2026-07-11)


### Features

* **einx:** universal tensor-notation operators (issue [#69](https://github.com/jejjohnson/geotoolz/issues/69), phases 0-1) ([#93](https://github.com/jejjohnson/geotoolz/issues/93)) ([4697956](https://github.com/jejjohnson/geotoolz/commit/46979569e6e42f5b5f11ac402f5d5ba8061ae7ea))
* **patch_ops:** label-aware chip samplers + train-tile/inference-stitch docs ([#95](https://github.com/jejjohnson/geotoolz/issues/95)) ([7a95e1d](https://github.com/jejjohnson/geotoolz/commit/7a95e1d4bd53c42ec73c909cb18294d708811a01))

## [0.1.0](https://github.com/jejjohnson/geotoolz/compare/v0.0.6...v0.1.0) (2026-06-10)


### ⚠ BREAKING CHANGES

* The composition core (`Operator`, `Sequential`, `Graph`, `Input`, `Node`, `Branch`, `Switch`, `Fanout`, `Identity`, `Const`, `Lambda`, `Sink`, `Tap`, `Snapshot`, `ShapeTrace`, `Carrier`) is now re-exported from the carrier-agnostic `pipekit` framework rather than implemented inside `geotoolz.core`.
* `from geotoolz import GeoSlice` and any `from geotoolz.catalog import ...` now fail. Install `geocatalog` (https://github.com/jejjohnson/geocatalog) and import from there.

### Features

* **compositing:** implement BlendMatched (mean / weighted_mean / ivw) ([#84](https://github.com/jejjohnson/geotoolz/issues/84)) ([b5a8337](https://github.com/jejjohnson/geotoolz/commit/b5a8337568a40a26b03efa2274a5345c0e3408e3))
* **compositing:** median/max-ndvi/cloud-free/bap/min-cloud reduction operators ([#64](https://github.com/jejjohnson/geotoolz/issues/64)) ([d689f72](https://github.com/jejjohnson/geotoolz/commit/d689f72a83c1923e538eaa5e296911898a052906))
* **coregister:** implement RasterToPoints + PointsToRaster ([#82](https://github.com/jejjohnson/geotoolz/issues/82)) ([46fcafb](https://github.com/jejjohnson/geotoolz/commit/46fcafb968f995293272260e60943c25e411e22f))
* **coregister:** implement RasterToRasterLike + StackMatched ([#81](https://github.com/jejjohnson/geotoolz/issues/81)) ([26cc60a](https://github.com/jejjohnson/geotoolz/commit/26cc60a2098e307f5bff4df2c4d1b163e9ce2958))
* **coregister:** point-cloud and vector-aggregation operators ([#83](https://github.com/jejjohnson/geotoolz/issues/83)) ([3faafd8](https://github.com/jejjohnson/geotoolz/commit/3faafd8311be4014538f060b273baceef9482bd2))
* cross-modality coregister operators + matched compositing ([#80](https://github.com/jejjohnson/geotoolz/issues/80)) ([19a7679](https://github.com/jejjohnson/geotoolz/commit/19a7679bd66e354c49a51ec0b2482a87c08a6cce))
* depend on pipekit for Operator / Sequential / Graph composition ([#74](https://github.com/jejjohnson/geotoolz/issues/74)) ([8558d05](https://github.com/jejjohnson/geotoolz/commit/8558d055abf6867a061366d65d5381c0da455ad2))
* **geom:** bowtie/antimeridian/parallax/segment sensor helpers ([#65](https://github.com/jejjohnson/geotoolz/issues/65)) ([621f0d0](https://github.com/jejjohnson/geotoolz/commit/621f0d0544ae270631cceba353aadbc9104529eb))
* **io:** hdf5/hdf4/netcdf-cf multi-format readers ([#63](https://github.com/jejjohnson/geotoolz/issues/63)) ([48f25b1](https://github.com/jejjohnson/geotoolz/commit/48f25b176034ba60af989f686a645c780554396d))
* **learn:** scikit-learn estimator adapter (phase-1, provisional API) ([#67](https://github.com/jejjohnson/geotoolz/issues/67)) ([8f4cea1](https://github.com/jejjohnson/geotoolz/commit/8f4cea190fcd5aadad6aab25aec2ee1f24771229))
* **matched_filter:** pure-numpy hyperspectral matched filter (path B) ([#62](https://github.com/jejjohnson/geotoolz/issues/62)) ([27e2531](https://github.com/jejjohnson/geotoolz/commit/27e253150749c7d5974a1365607967de1aed6acb))
* **measure:** skimage region properties bridge + plume regionprops upgrade ([#75](https://github.com/jejjohnson/geotoolz/issues/75)) ([517a01e](https://github.com/jejjohnson/geotoolz/commit/517a01e5cacd35acb1fb407d157763011536c869))
* **plume:** postprocessing operators from MethaneSAT segmentation paper ([#87](https://github.com/jejjohnson/geotoolz/issues/87)) ([95149da](https://github.com/jejjohnson/geotoolz/commit/95149da80c4632785402646be8497ef222f43876))
* **readers:** optional obstore byte path for SensorReader ([#86](https://github.com/jejjohnson/geotoolz/issues/86)) ([77231df](https://github.com/jejjohnson/geotoolz/commit/77231dfedee98cdd650cadb789b60f4d0d4f54d9))
* **readers:** sensor reader framework + toy_sensor reference reader ([#66](https://github.com/jejjohnson/geotoolz/issues/66)) ([0548f53](https://github.com/jejjohnson/geotoolz/commit/0548f53a57b5dcff9f2edb49c88add2a976e0b97))
* **segment:** skimage segmentation bridge ([#76](https://github.com/jejjohnson/geotoolz/issues/76)) ([08c239d](https://github.com/jejjohnson/geotoolz/commit/08c239d1bd4e96ce5158e4b5d7e3dee84d6336f3))
* **skimage:** feature extraction + CLAHE + geom registration ops ([#77](https://github.com/jejjohnson/geotoolz/issues/77)) ([e355788](https://github.com/jejjohnson/geotoolz/commit/e355788410bf7e38e98a7e9b9f283da1d5c43922))


### Code Refactoring

* extract catalog and GeoSlice into the geocatalog package ([#70](https://github.com/jejjohnson/geotoolz/issues/70)) ([830e842](https://github.com/jejjohnson/geotoolz/commit/830e842c48c0eb3de7672475c9fba41980d01eee))

## [0.0.6](https://github.com/jejjohnson/geotoolz/compare/v0.0.5...v0.0.6) (2026-05-16)


### Features

* **augment:** rs-safe spatial and spectral augmentations ([#43](https://github.com/jejjohnson/geotoolz/issues/43)) ([b9f1fba](https://github.com/jejjohnson/geotoolz/commit/b9f1fbaa3b9386d6a155c3e062637cc9b2a7289d))
* **geom:** geotoolz.geom operator surface ([#35](https://github.com/jejjohnson/geotoolz/issues/35)) ([fdc860a](https://github.com/jejjohnson/geotoolz/commit/fdc860ab15d002f4974af2ee49425e8bdc79b07c))
* **indices:** vegetation/water/snow/burn/mineral indices + named-band resolution ([#38](https://github.com/jejjohnson/geotoolz/issues/38)) ([1f6594b](https://github.com/jejjohnson/geotoolz/commit/1f6594b94a7198705a30a8b874fbcb06124f95f5))
* **io:** reader/writer operators for georeader IO ([#34](https://github.com/jejjohnson/geotoolz/issues/34)) ([530145f](https://github.com/jejjohnson/geotoolz/commit/530145f1e5955f3d414b1019942ab1999a125a06))
* **mask:** geometry rasterization, morphology, and boolean algebra ops ([#40](https://github.com/jejjohnson/geotoolz/issues/40)) ([bb507ef](https://github.com/jejjohnson/geotoolz/commit/bb507ef9804f6ee5ccab7f15c13d63a396fb7643))
* **normalize:** per-band min-max, z-score, robust, and fixed-stats ops ([#41](https://github.com/jejjohnson/geotoolz/issues/41)) ([4e6efad](https://github.com/jejjohnson/geotoolz/commit/4e6efad60accb8b34e2f3805e679e7b3f34dcdc4))
* **plume:** ch4/co2 retrieval, segmentation, and flux operators ([#45](https://github.com/jejjohnson/geotoolz/issues/45)) ([757e26d](https://github.com/jejjohnson/geotoolz/commit/757e26d220b0a8920189a4b37efd953007cadf72))
* **qa:** sensor-specific QA-bit decoders (Landsat, MODIS, S2) ([#39](https://github.com/jejjohnson/geotoolz/issues/39)) ([04eff6f](https://github.com/jejjohnson/geotoolz/commit/04eff6f811c066de2159b531ec4883757917b46d))
* **radiometry:** toa/boa pipeline operators (planck, dos1, srf, sza) ([#37](https://github.com/jejjohnson/geotoolz/issues/37)) ([0d57cff](https://github.com/jejjohnson/geotoolz/commit/0d57cffab9a1061b62d2ff5748e52d502d03a18f))
* **restore:** inpainting, gap-fill, despiking, and smoothing ops ([#42](https://github.com/jejjohnson/geotoolz/issues/42)) ([18524ae](https://github.com/jejjohnson/geotoolz/commit/18524aeece47c31c9bcfeab54731267299173586))
* **spectral:** band-space operators (SelectBands, BandMath, SRF, continuum removal) ([#36](https://github.com/jejjohnson/geotoolz/issues/36)) ([79504ac](https://github.com/jejjohnson/geotoolz/commit/79504acfe503b73e4a95aaa7bf51c922279357df))
* **viz:** rgb/false-color composites, stretches, and colormaps ([#44](https://github.com/jejjohnson/geotoolz/issues/44)) ([293acb5](https://github.com/jejjohnson/geotoolz/commit/293acb5cf5f7fcb733748f3de2d8b93d3b32419e))

## [0.0.5](https://github.com/jejjohnson/geotoolz/compare/v0.0.4...v0.0.5) (2026-05-15)


### Features

* **catalog:** streaming backend="duckdb" builders ([#15](https://github.com/jejjohnson/geotoolz/issues/15)) ([75fdb70](https://github.com/jejjohnson/geotoolz/commit/75fdb701423d132e8703fbc7ff1f3238c3c1d118))
* indices + radiometry + cloud operator stdlib (v0.1) ([#17](https://github.com/jejjohnson/geotoolz/issues/17)) ([48a8bed](https://github.com/jejjohnson/geotoolz/commit/48a8bed005c2eb80bc55ae94ed02fe27f8d7d5e3))

## [0.0.4](https://github.com/jejjohnson/geotoolz/compare/v0.0.3...v0.0.4) (2026-05-15)


### Features

* **catalog:** duckdb-backed GeoCatalog (Phase 2 of geodatabase) ([#14](https://github.com/jejjohnson/geotoolz/issues/14)) ([1a4a31e](https://github.com/jejjohnson/geotoolz/commit/1a4a31e6ccd7df44c93eed02eb5ab18efef74dc1))
* **catalog:** in-memory GeoCatalog + GeoSlice (Phase 1 of geodatabase) ([#12](https://github.com/jejjohnson/geotoolz/issues/12)) ([7b8a7a0](https://github.com/jejjohnson/geotoolz/commit/7b8a7a0d5a19da912b957cfa8dfcff66f165cc84))

## [0.0.3](https://github.com/jejjohnson/geotoolz/compare/v0.0.2...v0.0.3) (2026-05-15)


### Features

* **patch:** four-axis Patcher framework (geopatcher) ([#10](https://github.com/jejjohnson/geotoolz/issues/10)) ([c38e19d](https://github.com/jejjohnson/geotoolz/commit/c38e19d59ee8e0091fdf7af5b7f1de8a5432a1fa))

## [0.0.2](https://github.com/jejjohnson/geotoolz/compare/v0.0.1...v0.0.2) (2026-05-15)


### Features

* composition core — Operator, Sequential, Graph, ModelOp + v0.1 idiom library ([#8](https://github.com/jejjohnson/geotoolz/issues/8)) ([9c669ee](https://github.com/jejjohnson/geotoolz/commit/9c669eeaaab1fbd0fc3868f0a8230fe6e33bb689))

## 0.0.1 (2026-05-14)


### Features

* scaffold geotoolz package from pypackage_template ([4e02738](https://github.com/jejjohnson/geotoolz/commit/4e02738c4501baaea07395c6f794608b37123e36))
* scaffold geotoolz package from pypackage_template ([2d1d6fc](https://github.com/jejjohnson/geotoolz/commit/2d1d6fc97c55157b28be6707248acdcd8fa14783))

## Changelog

All notable changes to this project will be documented in this file.

See [Conventional Commits](https://www.conventionalcommits.org/) for commit guidelines.

## Unreleased
