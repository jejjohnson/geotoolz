# Patch a catalog with geopatcher

`geocatalog.patch.field_for` mosaics the rows a `GeoSlice` selects into one
`geopatcher.RasterField`, which a `SpatialPatcher` splits into patches and
merges back. It needs the `[patch]` extra; how the patcher and catalog fit
together is on [How the packages interlock](../../geostack.md).

## Tile a mosaic and stitch it back

```python
# Shapes: the slice grid is H × W = 300 × 400 pixels, C = 4 bands.
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

import geocatalog as gc
import geopatcher as gp
from geocatalog.backends import InMemoryGeoCatalog

# Two adjacent 3 km tiles, 4 bands at 10 m, UTM 11N.
root: Path = Path(tempfile.mkdtemp())
rng: np.random.Generator = np.random.default_rng(0)
for i, x0 in enumerate((500_000, 503_000)):
    with rasterio.open(
        root / f"tile{i}_20240605.tif", "w", driver="GTiff", width=300, height=300, count=4,
        dtype="uint16", crs="EPSG:32611", transform=from_origin(x0, 4_303_000, 10, 10),
    ) as dst:
        dst.write(rng.integers(0, 10_000, (4, 300, 300), dtype=np.uint16))  # (4, 300, 300) uint16

catalog: InMemoryGeoCatalog = gc.build.build_raster_catalog(
    sorted(root.glob("*.tif")), filename_regex=r"_(?P<date>\d{8})\.tif", crs="EPSG:32611"
)                                                                    # 2 rows
aoi: gc.GeoSlice = gc.GeoSlice(
    bounds=(501_000, 4_300_000, 505_000, 4_303_000),                 # straddles both tiles
    interval=pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both"),
    resolution=(10.0, 10.0),
    crs="EPSG:32611",
)

field: gp.RasterField = gc.patch.field_for(catalog, aoi)              # domain (C, H, W) uint16
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(100, 100)),
    sampler=gp.spatial.sampler.RegularStride(step=(100, 100)),
    window=gp.spatial.window.Boxcar(),
    aggregation=gp.spatial.aggregation.Mean(),
)
patches: list[gp.Patch] = list(patcher.split(field))                  # 12 patches of (C, 100, 100)
scaled: list[gp.Patch] = [
    p.with_data(np.asarray(p.data) * 1e-4)                           # (C, 100, 100) uint16 → float64
    for p in patches
]
reflectance: np.ndarray = patcher.merge(scaled, field.domain)         # (C, H, W) float64
```

`field_for` reads through `load_raster`, so files in other CRSs are warped
onto the slice grid and mosaicked. Its options:

| Argument | Effect |
| --- | --- |
| `slice_=None` | cover the whole catalog in its CRS, on the first file's grid |
| `asset="B04"` | read that key of each row's `assets` map (bundle or staged rows) instead of `filepath` |
| `band_indexes=[1, 2]` | keep these 1-indexed bands |
| `materialize=False` | read nothing; return one lazy `RasterField` per row, each in its file's own CRS and grid |

For remote rows, [stage](staging.md) the catalog first so the patcher
reads local files. Choosing the patcher's four axes is covered in the
[patcher concepts](../../patcher/concepts.md).

## Walk a catalog file by file

`geocatalog.patch.CatalogDomain` yields one `GeoSlice` per row, for code
that loads and processes each file itself:

```python
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.transform import from_origin

import geocatalog as gc
from geocatalog.backends import InMemoryGeoCatalog

root: Path = Path(tempfile.mkdtemp())
for i, x0 in enumerate((500_000, 503_000)):
    with rasterio.open(
        root / f"tile{i}_20240605.tif", "w", driver="GTiff", width=300, height=300, count=4,
        dtype="uint16", crs="EPSG:32611", transform=from_origin(x0, 4_303_000, 10, 10),
    ) as dst:
        dst.write(np.full((4, 300, 300), 1_000, dtype=np.uint16))   # (4, 300, 300) uint16

catalog: InMemoryGeoCatalog = gc.build.build_raster_catalog(
    sorted(root.glob("*.tif")), filename_regex=r"_(?P<date>\d{8})\.tif", crs="EPSG:32611"
)
domain: gc.patch.CatalogDomain = gc.patch.CatalogDomain(catalog=catalog, resolution=(10.0, 10.0))
means: list[float] = []
for slice_ in domain.slices():                                       # one 3 km slice per row
    nir: GeoTensor = gc.load.load_raster(catalog, slice_, band_indexes=[4])  # (1, 300, 300) uint16
    means.append(float(np.asarray(nir).mean()))
```

`CatalogDomain` accepts either backend. Point and line footprints become
one-pixel slices, and rows without a footprint are skipped with a warning.
No geopatcher class consumes it; use `field_for` to feed a patcher.
