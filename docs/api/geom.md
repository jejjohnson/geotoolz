# Geometry

::: geotoolz.geom

## Projection operator choices

- `Reproject` changes CRS and optionally resolution.
- `ReprojectLike` matches another `GeoTensor`'s CRS, transform, and shape.
- `ResampleLike` is the same grid-matching surface for resolution/alignment work where the CRS is already compatible.
- `CropToBounds` delegates to `georeader.read.read_from_bounds(..., boundless=False)`: the pixel window is rounded outward, so every pixel the bounds touch (even partially) is kept, and the result is clipped to the carrier.

