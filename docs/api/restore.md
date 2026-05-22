# Restore

Denoising, despeckling, destriping, gap-fill, and inpainting operators.

- **Denoising:** `BilateralDenoise`, `GaussianDenoise`, `MedianDenoise`, `NLMeans`, `DenoisePCA`,
  `MNF` / `InverseMNF` (minimum noise fraction)
- **SAR despeckle:** `DespeckleLee`, `DespeckleRefinedLee`, `DespeckleFrost`
- **Destripe:** `DestripeColumn`
- **Gap fill:** `GapFillNearest`, `GapFillIDW`, `GapFillInpaintBiharmonic`, `GapFillLaplacian`
- **Outlier handling:** `OutlierMask`, `ReplaceOutliers`, `SaturationFlag`
- **Histogram matching:** `MomentMatching`

::: geotoolz.restore
