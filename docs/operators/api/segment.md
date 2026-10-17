# Segment

`geoops.segment` wraps `skimage.segmentation`. Outputs are integer-labelled `GeoArray`s with the
input CRS / transform preserved.

- **Thresholding:** `Threshold` — absolute, Otsu or `"percentile:<p>"` global threshold to a
  boolean mask (nodata excluded from the statistic and always `False`)
- **Superpixels:** `SLIC`, `Quickshift`, `Felzenszwalb`
- **Region-based:** `Watershed`, `RandomWalker`, `ChanVese`
- **Post-processing:** `ExpandLabels`, `MarkBoundaries`, `MergeNearbyInstances`, `MaskNMS`
- **Tier-A primitives:** `threshold_mask`, `otsu_threshold`, `resolve_threshold`, `fill_invalid`,
  `merge_nearby_instances`, `mask_nms` (`geoops.plume.PlumeMask` uses the same thresholding)

Operators that accept non-JSON-safe carrier params (`mask`, `markers`, `label_img`) set
`forbid_in_yaml=True` so hydra-zen won't try to round-trip them.

::: geoops.segment
