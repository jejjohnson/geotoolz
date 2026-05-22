# Mask

Geometry-based masks, morphological mask ops, and boolean mask algebra. Convention: `True` means
"mask this pixel out" (consistent with `geotoolz.cloud` / `geotoolz.qa`).

- **Rasterise vector geometry:** `MaskFromGeometry`, `MaskFromGeoDataFrame`
- **Morphology:** `Dilate`, `Erode`, `Open`, `Close`, `RemoveSmallObjects`, `RemoveSmallHoles`
- **Algebra:** `MaskAnd`, `MaskOr`, `MaskNot`, `MaskXor`
- **Apply:** `ApplyMask` (fill values where mask is `True`)

::: geotoolz.mask
