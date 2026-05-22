# Measure

`geotoolz.measure` wraps `skimage.measure` for region properties and contour extraction. Outputs are
either label `GeoTensor`s or `GeoDataFrame`s with geometries in the input CRS.

- `LabelConnectedComponents` — 4/8-connectivity component labelling
- `RegionProps` — per-component property table; carries `forbid_in_yaml=True` when given an
  `intensity_image`
- `FindContours` — marching-squares iso-contours emitted as `LineString` features
- `ProfileLine` — sample values along a line between two points
- `RANSAC` — generic model fitting on `(N, 2)` coordinate pairs
- `ShannonEntropy` — single-band scalar entropy

::: geotoolz.measure
