"""`geocloud.cog` — Cloud-Optimized GeoTIFFs: reads from object storage, and writes.

Every read returns a `geoarray.GeoArray`.

- `CogSource` — an opened, tiled COG (async-geotiff over the shared
  obstore pool): ``read_windows`` fetches every tile the windows touch
  once, in concurrent groups, sync or ``await``-ed. A
  `geoarray.RasterSource` and `geoarray.AsyncRasterSource` (``grid``,
  ``shape``, ``dtype``, ``fill_value``, ``attrs``, ``read_window`` /
  ``load`` and their ``a``-twins). Pickles by URL.
- `CogDomain` — its I/O-free metadata (``crs``, ``transform``, ``grid``,
  ``shape``, ``bounds``, ``res``, ``nodata``, ``fill_value``).
- `AsyncCogReader` — the ``await``-native face of a source: a
  `geoarray.AsyncRasterSource` that keeps georeader's metadata surface,
  lazy window views and an async ``load``. The ``read_*`` coroutines
  mirror ``georeader.read``: they fetch only the source pixels a request
  needs, asynchronously, then reuse the sync georeader function on that
  chunk (georeader is the engine; results come back as `GeoArray`).

- `write_cog` — write a `GeoArray` (or any `RasterSource`, strip by strip)
  as a validated COG to a local path or any object store: explicit
  compression / predictor / tiling / overview / nodata choices, built in
  a temporary directory and only then renamed or uploaded into place.

The readers need the ``[cog]`` extra (``pip install
'geotoolz-cloud[cog]'``); `write_cog` runs on the base install (GDAL's
``COG`` driver through rasterio).

```python
from geoarray import GeoArray, PixelWindow

from geocloud.cog import AsyncCogReader, CogSource, read_from_tile, write_cog

src: CogSource = CogSource.open("s3://bucket/scene.tif")
window = PixelWindow(col_off=0, row_off=0, width=256, height=256)
chip: GeoArray = src.read_window(window)        # (C, 256, 256)
chips: list[GeoArray] = src.read_windows([window, window])  # tiles fetched once

# in a notebook / async function:
reader: AsyncCogReader = await AsyncCogReader.open("s3://bucket/scene.tif")
tile: GeoArray = await read_from_tile(reader, x=163, y=395, z=10)  # (C, 256, 256)

write_cog(chip, "s3://bucket/products/chip.tif", compress="zstd")
```
"""

from __future__ import annotations

from geocloud._src.cog_reader import (
    AsyncCogReader,
    read_from_bounds,
    read_from_center_coords,
    read_from_polygon,
    read_from_tile,
    read_from_window,
    read_reproject,
    read_reproject_like,
    read_to_crs,
)
from geocloud._src.cog_source import CogDomain, CogSource
from geocloud._src.cog_write import write_cog


__all__ = [
    "AsyncCogReader",
    "CogDomain",
    "CogSource",
    "read_from_bounds",
    "read_from_center_coords",
    "read_from_polygon",
    "read_from_tile",
    "read_from_window",
    "read_reproject",
    "read_reproject_like",
    "read_to_crs",
    "write_cog",
]
