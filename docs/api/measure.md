# Measure

`geotoolz.measure` wraps `skimage.measure` for region properties and contour extraction. Outputs are
either label `GeoTensor`s or `GeoDataFrame`s with geometries in the input CRS.

- `LabelConnectedComponents` — component labelling with `connectivity=4|8` (skimage's `1|2`
  spelling is not accepted) and an optional `min_area` filter
- `RegionProps` — per-component property table; carries `forbid_in_yaml=True` when given an
  `intensity_image`. Property columns are in **pixel units** (areas in pixels, lengths in pixel
  widths, positions as `(row, col)`); `scale_to_crs=True` converts areas / lengths / second moments
  to CRS units. The `geometry` column is always in CRS units.
- `FindContours` — marching-squares iso-contours emitted as `LineString` features (CRS units)
- `SkeletonLength` — longest Euclidean path through the mask skeleton, in pixel widths
  (`scale_to_crs=True` for CRS units)
- `ProfileLine` — sample values along a line between two points
- `RANSAC` — generic model fitting on `(N, 2)` coordinate pairs
- `ShannonEntropy` — single-band scalar entropy

Shared primitives (also used by `geotoolz.mask` and `geotoolz.plume`): `label_components`,
`skeleton_length`, `regionprops_frame` and `DEFAULT_REGIONPROPS`, plus
`regionprops_to_crs_units` (the `scale_to_crs=True` conversion).

::: geotoolz.measure
