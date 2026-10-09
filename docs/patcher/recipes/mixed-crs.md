# Patch across coordinate systems

Patch a field whose CRS differs from your anchors or from the grid you
want. There are two levels: reproject the anchor coordinates (cheap), or
reproject the pixels of every chip (heavy).

| You have | Use |
|---|---|
| Imagery on the right grid; event or track coordinates in lon/lat | `spatial.sampler.ExplicitCoords(crs=...)` or `spatial.sampler.AlongTrack(crs=...)` |
| Imagery on the wrong grid for your pipeline | `geopatcher.fields.ReprojectingRasterField(reader, dst_crs=...)` |

## Centre patches on lon/lat points

`ExplicitCoords` reprojects each coordinate to the domain's CRS and
centres a patch on it. `AlongTrack(track, spacing, crs=...)` does the same
for points spaced along a track, with `spacing` in domain units.

```python
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp

field: gp.RasterField = gp.RasterField(
    GeoTensor(
        np.ones((1, 256, 256), dtype=np.float32),       # (1, 256, 256) float32, UTM 11N
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
events: list[tuple[float, float]] = [(-116.9926, 38.8431), (-116.9779, 38.8315)]  # lon, lat

patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(64, 64)),
    sampler=gp.spatial.sampler.ExplicitCoords(coords=events, crs="EPSG:4326"),  # None = domain CRS
    window=gp.spatial.window.Boxcar(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
chips: list[gp.Patch] = list(patcher.split(field))    # 2 × (1, 64, 64) float32
print([c.anchor for c in chips])                      # pixel anchors in the UTM grid
```

`polar_guard="warn"` (the default; also `"raise"` or `"ignore"`) flags
unreliable reprojection near the poles (`|lat| > 80°`) or across the
antimeridian when the source CRS is geographic.

## Patch on a different grid

`ReprojectingRasterField` presents the destination grid as its domain, so
every sampler, geometry and aggregation works on that grid. Each chip is
warped from a crop of the source around its footprint.

```python
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp

source: GeoTensor = GeoTensor(
    np.ones((1, 256, 256), dtype=np.float32),           # (1, 256, 256) float32, UTM 11N at 10 m
    transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
    crs="EPSG:32611",
)
field: gp.fields.ReprojectingRasterField = gp.fields.ReprojectingRasterField(
    source, dst_crs="EPSG:3857", resolution=30.0
)
print(field.domain.crs, field.domain.shape)            # EPSG:3857 (1, 111, 110)

patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(64, 64)),
    sampler=gp.spatial.sampler.RegularStride(step=(32, 32)),
    window=gp.spatial.window.Hann(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
chips: list[gp.Patch] = list(patcher.split(field))    # 4 × (1, 64, 64) float32 in EPSG:3857
mosaic: np.ndarray = patcher.merge(chips, field.domain)  # (1, 111, 110) float64 · NaN past row / col 96
```

- The domain keeps the source's leading dims (`(bands, H, W)`), so chips
  merge back without a rank mismatch.
- Per-chip cost scales with the chip, not the scene.
- Chips are warped independently, so a stitched mosaic can differ
  slightly from one full-scene warp. Call `georeader.read.read_reproject`
  once when you need that exactly.
- `resampling=` defaults to `"bilinear"`; it is part of the
  [PatchCache](patch-cache.md) key.
