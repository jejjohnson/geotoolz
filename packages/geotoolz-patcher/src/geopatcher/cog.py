"""Public alias for `geopatcher._src.cog` — async COG reading.

`AsyncCogReader` is the ``await``-native face of `ObstoreCogField`'s
engine (async-geotiff over the shared obstore pool, shared tiles
fetched once, concurrent fetch groups), with georeader's `GeoData`
metadata surface. The ``read_*`` coroutines mirror ``georeader.read``:
they fetch only the source pixels a request needs, asynchronously, then
reuse the sync georeader function on that chunk. Needs the
``[obstore-cog]`` extra: ``pip install 'geotoolz-patcher[obstore-cog]'``.

```python
from geopatcher.cog import AsyncCogReader, read_from_tile, read_reproject_like

reader = await AsyncCogReader.open("s3://bucket/scene.tif")
tile = await read_from_tile(reader, x=163, y=395, z=10)
chips = await reader.load_many(windows)
```
"""

from __future__ import annotations

from geopatcher._src.cog import (
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


__all__ = [
    "AsyncCogReader",
    "read_from_bounds",
    "read_from_center_coords",
    "read_from_polygon",
    "read_from_tile",
    "read_from_window",
    "read_reproject",
    "read_reproject_like",
    "read_to_crs",
]
