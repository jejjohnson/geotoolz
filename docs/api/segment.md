# Segment

`geotoolz.segment` wraps `skimage.segmentation`. Outputs are integer-labelled `GeoTensor`s with the
input CRS / transform preserved.

- **Superpixels:** `SLIC`, `Quickshift`, `Felzenszwalb`
- **Region-based:** `Watershed`, `RandomWalker`, `ChanVese`
- **Post-processing:** `ExpandLabels`, `MarkBoundaries`

Operators that accept non-JSON-safe carrier params (`mask`, `markers`, `label_img`) set
`forbid_in_yaml=True` so hydra-zen won't try to round-trip them.

::: geotoolz.segment
