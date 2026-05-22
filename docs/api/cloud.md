# Cloud

`geotoolz.cloud` is the generic primitive layer for cloud / shadow / invalid-pixel masking. Pair these
with the sensor-specific decoders in [`geotoolz.qa`](qa.md) when working with a known sensor product.

- `MaskFromQABits` — decode bit-flag QA bands (Landsat-style, OR of bits).
- `MaskFromSCL` — categorical scene-classification membership (Sentinel-2 SCL).
- `MaskValid` / `ApplyMask` — apply a boolean mask to a `GeoTensor`.
- `SCL`, `SCL_CLOUDS`, `SCL_INVALID` — Sentinel-2 SCL class registries.

The convention is **`True` means "mask this pixel out"**.

::: geotoolz.cloud
