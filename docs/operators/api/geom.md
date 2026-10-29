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
- **Diagnostic calculus:** `Gradient`, `Divergence`, `Curl`, `Laplacian` (operators); `gradient`, `divergence`, `curl`, `laplacian` and their `adaptive_*` twins (arrays); `difference`, `adaptive_difference`, `difference_offsets`, `difference_coeffs` (stencils)

## Projection operator choices

- `Reproject` changes CRS and optionally resolution.
- `ReprojectLike` matches another `GeoArray`'s CRS, transform, and shape.
- `ResampleLike` is the same grid-matching surface for resolution/alignment work where the CRS is already compatible.
- `CropToBounds` delegates to `georeader.read.read_from_bounds(..., boundless=False)`: the pixel window is rounded outward, so every pixel the bounds touch (even partially) is kept, and the result is clipped to the carrier.

## Diagnostic calculus

The calculus family has three levels, and each level is built from the one below.
The operators (`Gradient`, `Divergence`, `Curl`, `Laplacian`) read the geometry from the carrier: they map pixel derivatives to the CRS's axes through the transform, and on a geographic CRS they use the CRS ellipsoid's radii, so results are per CRS unit (per metre on a geographic CRS).
The array primitives (`gradient`, `divergence`, `curl`, `laplacian` and their `adaptive_*` twins) take the row and column steps `dy`, `dx` instead, so the same maths runs on bare NumPy or JAX arrays inside a jitted cost.
Both rest on `difference` and `adaptive_difference`, whose stencils come from `difference_offsets` and `difference_coeffs`.

Pick `nan_policy="adaptive"` when the field has a mask that matters, such as a coastline: each derivative then uses the widest stencil that fits on valid data, so the cells next to land keep a value and only the masked cells are `NaN`.
The default, `"propagate"`, uses the plain stencil, and a masked cell costs a halo of about `accuracy` cells.

```python
import numpy as np
from geoarray import GeoArray

import geoops as gz

rows, cols = np.mgrid[0:7, 0:9] + 0.5
x, y = 10.0 * cols, -10.0 * rows                      # (7, 9) metres
transform = (10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
u = GeoArray(-y, transform, "EPSG:32613", dims=("y", "x"))  # (7, 9) float64
v = GeoArray(x, transform, "EPSG:32613", dims=("y", "x"))   # (7, 9) float64
land = np.zeros((7, 9), dtype=bool)                   # (7, 9) bool, True drops
land[:, 4] = True

zeta = gz.Curl(nan_policy="adaptive")(u, v, land)     # (7, 9), 2.0 off the wall
div = gz.Divergence()(u, v)                           # (7, 9), 0.0 everywhere
same = gz.geom.adaptive_curl(-y, x, -10.0, 10.0, mask=land)  # (7, 9) ndarray
```

::: geoops.geom

## Coregistration

`geom.coregister` aligns one source onto another's support — the
coregistration callables for `geopatcher.matched.MatchedField`:
`RasterToRasterLike`, `RasterToPoints` / `PointsToRaster` (`[vector-cube]`
extra), `RasterToPointCloud` / `PointCloudToRaster`, `VectorToRasterAgg`.

::: geoops.geom.coregister
