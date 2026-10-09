# Geometry

Grid, projection and tiling operators, plus swath / sensor-geometry
corrections and image registration. Coregistration operators for
multi-source matchups live in the `geom.coregister` subnamespace
([below](#coregistration)).

- **Projection / resampling:** `Reproject`, `ReprojectLike`, `Resample`, `ResampleLike`, `Resize`
- **Shape:** `CropTo`, `CropToBounds`, `PadTo`
- **Tiling:** `Tile`, `SlidingWindow`, `Stitch`, `Mosaic`
- **Vector ↔ raster:** `Rasterize`, `RasterizeLike`, `Vectorize`
- **Swath / sensor geometry:** `Georeference` (GLT), `SegmentStitch`, `BowtieCorrection`,
  `AntimeridianSplit`, `GeostationaryParallaxCorrect`
- **Registration:** `PhaseAlign` (phase cross-correlation), `OpticalFlowILK`, `OpticalFlowTVL1`

## Projection operator choices

- `Reproject` changes CRS and optionally resolution.
- `ReprojectLike` matches another `GeoTensor`'s CRS, transform, and shape.
- `ResampleLike` is the same grid-matching surface for resolution/alignment work where the CRS is already compatible.
- `CropToBounds` delegates to `georeader.read.read_from_bounds(..., boundless=False)`: the pixel window is rounded outward, so every pixel the bounds touch (even partially) is kept, and the result is clipped to the carrier.

::: geotoolz.geom

## Coregistration

`geom.coregister` aligns one source onto another's support — the
coregistration callables for `geopatcher.matched.MatchedField`:
`RasterToRasterLike`, `RasterToPoints` / `PointsToRaster` (`[vector-cube]`
extra), `RasterToPointCloud` / `PointCloudToRaster`, `VectorToRasterAgg`.

::: geotoolz.geom.coregister
