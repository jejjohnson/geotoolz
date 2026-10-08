"""`geocloud.cog` — Cloud-Optimized GeoTIFF reads straight from object storage.

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

Needs the ``[cog]`` extra: ``pip install 'geotoolz-cloud[cog]'``.

```python
from geocloud.cog import AsyncCogReader, CogSource, read_from_tile

src = CogSource.open("s3://bucket/scene.tif")
chips = src.read_windows(windows)          # shared tiles fetched once

reader = await AsyncCogReader.open("s3://bucket/scene.tif")
tile = await read_from_tile(reader, x=163, y=395, z=10)
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
]
