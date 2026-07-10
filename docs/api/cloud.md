# Cloud (deprecated alias)

!!! warning "Deprecated"
    `geotoolz.cloud` is now a compatibility alias. Its contents moved:
    mask **extraction** (`MaskFromQABits`, `MaskFromSCL`, `MaskValid`,
    the `SCL` enum + convenience sets, `mask_from_qa_bits`,
    `mask_from_scl`) lives in [`geotoolz.qa`](qa.md); mask
    **application** (`ApplyMask`, `apply_mask`) lives in
    [`geotoolz.mask`](mask.md). The alias re-exports the original names
    unchanged and will be removed in a future release.

The convention is unchanged everywhere: **`True` means "mask this pixel out"**.
