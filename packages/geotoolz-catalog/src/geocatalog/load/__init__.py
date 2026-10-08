"""`geocatalog.load` — read the pixels a catalog covers on a `GeoSlice`.

Each loader takes a catalog and a `geocatalog.GeoSlice`, and returns
the data resampled onto that slice's grid (its bounds, resolution and
CRS):

- `load_raster` / `aload_raster` — one mosaicked ``(C, H, W)``
  `GeoTensor` (``aload_raster`` is the async twin).
- `load_raster_timeseries` — one ``(T, C, H, W)`` `GeoTensor`, a time
  step per day in the slice's interval.
- `load_xarray` — an xarray Dataset (``[xarray-raster]`` extra; lazy).
- `load_vector` — a GeoDataFrame (``[vector]`` extra; lazy).

```python
import pandas as pd

from geocatalog import GeoSlice, open_catalog
from geocatalog.load import load_raster

catalog = open_catalog("scenes.parquet", engine="memory")
aoi = GeoSlice(
    bounds=(500_000.0, 4_000_000.0, 510_000.0, 4_010_000.0),
    interval=pd.Interval(pd.Timestamp("2024-06-01"),
                         pd.Timestamp("2024-06-30"), closed="both"),
    resolution=(10.0, 10.0),
    crs="EPSG:32630",
)
tensor = load_raster(catalog, aoi)   # (C, 1000, 1000)
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from geocatalog._src._lazy import lazy_getattr
from geocatalog._src.formats.raster import (
    aload_raster,
    load_raster,
    load_raster_timeseries,
)


if TYPE_CHECKING:
    from geocatalog._src.formats.vector import load_vector
    from geocatalog._src.formats.xarray_backend import load_xarray


__all__ = [
    "aload_raster",
    "load_raster",
    "load_raster_timeseries",
    "load_vector",
    "load_xarray",
]


# Extras-gated names resolve lazily (`geocatalog._src._lazy.LAZY`).
__getattr__ = lazy_getattr(
    globals(),
    [
        "load_vector",
        "load_xarray",
    ],
)
