# Restore

Denoising, despeckling, destriping, gap-fill, and inpainting operators.

- **Denoising:** `BilateralDenoise`, `GaussianDenoise`, `MedianDenoise`, `NLMeans`, `DenoisePCA`,
  `MNF` (minimum noise fraction, Green et al. 1988; `fit` / `transform` / `inverse`)
- **SAR despeckle:** `DespeckleLee` (Lee 1980), `DespeckleRefinedLee`, `DespeckleFrost` (Frost 1982)
- **Destripe:** `DestripeColumn`, `MomentMatching` (per-column gain + offset, Gadallah et al. 2000)
- **Gap fill:** `GapFillNearest`, `GapFillIDW`, `GapFillInpaintBiharmonic`, `GapFillLaplacian`
- **Outlier handling:** `OutlierMask`, `ReplaceOutliers` (saturation flags: `geotoolz.qa.MaskSaturated`)
- **Tier-A primitives:** `bilateral_denoise`, `gaussian_denoise`, `median_denoise`, `nl_means`,
  `pca_denoise`, `despeckle_lee`, `despeckle_refined_lee`, `despeckle_frost`, `destripe_column`,
  `gap_fill_nearest`, `gap_fill_idw`, `gap_fill_biharmonic`, `gap_fill_laplacian`, `outlier_mask`,
  `replace_outliers`

::: geotoolz.restore
