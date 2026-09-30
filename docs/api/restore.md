# Restore

Denoising, despeckling, destriping, gap-fill, and inpainting operators.

- **Denoising:** `BilateralDenoise`, `GaussianDenoise`, `MedianDenoise`, `NLMeans`, `DenoisePCA`,
  `MNF` / `InverseMNF` (minimum noise fraction, Green et al. 1988)
- **SAR despeckle:** `DespeckleLee` (Lee 1980), `DespeckleRefinedLee`, `DespeckleFrost` (Frost 1982)
- **Destripe:** `DestripeColumn`, `MomentMatching` (per-column gain + offset, Gadallah et al. 2000)
- **Gap fill:** `GapFillNearest`, `GapFillIDW`, `GapFillInpaintBiharmonic`, `GapFillLaplacian`
- **Outlier handling:** `OutlierMask`, `ReplaceOutliers` (saturation flags: `geotoolz.qa.MaskSaturated`)

::: geotoolz.restore
