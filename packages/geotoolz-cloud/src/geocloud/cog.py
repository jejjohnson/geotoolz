"""`geocloud.cog` — Cloud-Optimized GeoTIFFs: reads from object storage, and writes.

- `CogSource` — an opened, tiled COG (async-geotiff over the shared
  obstore pool): ``read_windows`` fetches every tile the windows touch
  once, in concurrent groups, sync or ``await``-ed. Pickles by URL.
- `CogDomain` — its I/O-free metadata (``crs``, ``transform``,
  ``shape``, ``bounds``, ``res``, ``nodata``).
- `AsyncCogReader` — the ``await``-native, georeader-shaped face of a
  source: georeader's `GeoData` metadata, lazy window views and an async
  ``load``. The ``read_*`` coroutines mirror ``georeader.read``: they
  fetch only the source pixels a request needs, asynchronously, then
  reuse the sync georeader function on that chunk.

- `write_cog` — write a `GeoTensor` (or a lazy reader, strip by strip) as
  a validated COG to a local path or any object store: explicit
  compression / predictor / tiling / overview / nodata choices, built in
  a temporary directory and only then renamed or uploaded into place.

The readers need the ``[cog]`` extra (``pip install
'geotoolz-cloud[cog]'``); `write_cog` runs on the base install (GDAL's
``COG`` driver through rasterio).

```python
from geocloud.cog import AsyncCogReader, CogSource, read_from_tile

src = CogSource.open("s3://bucket/scene.tif")
chips = src.read_windows(windows)          # shared tiles fetched once

reader = await AsyncCogReader.open("s3://bucket/scene.tif")
tile = await read_from_tile(reader, x=163, y=395, z=10)

write_cog(ndvi, "s3://bucket/products/ndvi.tif", compress="zstd")
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
