"""`geocatalog` — find, index, join, load and stage geospatial files.

The root holds what every catalog workflow touches:

- `GeoCatalog` — the catalog protocol every backend satisfies.
- `GeoSlice` — bounds + time interval + resolution + CRS: what catalogs
  are queried with and loaders read on.
- `open_catalog` — open a saved catalog with the right backend.
- `query` / `intersect` / `union` — select rows, and combine two
  catalogs.

Everything else lives in one sub-namespace per step of the workflow:

- discover — `geocatalog.sources`: `STACSource`, `CMRSource`,
  `EarthAccessSource`, `GEESource`.
- index — `geocatalog.build`: `build_raster_catalog`,
  `build_xarray_catalog`, `build_vector_catalog`.
- hold — `geocatalog.backends`: `InMemoryGeoCatalog`, `DuckDBGeoCatalog`,
  the errors.
- join — `geocatalog.matchup`: `matchup` and its spatial / temporal
  strategies.
- read — `geocatalog.load`: `load_raster`, `load_raster_timeseries`,
  `load_xarray`.
- save / share — `geocatalog.storage`: GeoParquet, STAC export,
  `CatalogBundle`.
- stage — `geocatalog.staging`: `stage` (into a `geocloud.cache.LocalCache`).
- patch — `geocatalog.patch`: `field_for`, `CatalogDomain` (the geopatcher
  bridge).
- grids — `geocatalog.grid`: `slice_to_window`, `is_grid_aligned`,
  `count_steps`.
- helpers — `geocatalog.utils`: `parse_uri`, `retry_transient_io`, UTC
  time helpers.

Each public name has exactly one home. Extras-gated names (DuckDB,
xarray, vector, STAC, the source adapters) load lazily, so importing
any namespace never needs an optional dependency; using a name whose
extra is missing raises its install hint.

```python
import pandas as pd

from geocatalog import GeoSlice, open_catalog
from geocatalog.load import load_raster

catalog = open_catalog("scenes.parquet")
aoi = GeoSlice(
    bounds=(500_000.0, 4_000_000.0, 510_000.0, 4_010_000.0),
    interval=pd.Interval(pd.Timestamp("2024-06-01"),
                         pd.Timestamp("2024-06-30"), closed="both"),
    resolution=(10.0, 10.0),
    crs="EPSG:32630",
)
june = catalog.query(aoi)         # the rows overlapping the slice
tensor = load_raster(june, aoi)   # (C, 1000, 1000) GeoTensor
```
"""

from __future__ import annotations

from loguru import logger as _logger

from geocatalog import (
    backends,
    build,
    grid,
    load,
    matchup,
    patch,
    sources,
    staging,
    storage,
    utils,
)
from geocatalog._src.base import GeoCatalog
from geocatalog._src.factory import open_catalog
from geocatalog._src.geoslice import GeoSlice
from geocatalog._src.ops import intersect, query, union


# Library hygiene: loguru's recommended pattern is to disable the
# library's own logger at import time so consumers don't see output by
# default. Opt in from a consumer app with `logger.enable("geocatalog")`.
_logger.disable("geocatalog")


__version__ = "0.5.0"  # x-release-please-version

__all__ = [
    "GeoCatalog",
    "GeoSlice",
    "__version__",
    "backends",
    "build",
    "grid",
    "intersect",
    "load",
    "matchup",
    "open_catalog",
    "patch",
    "query",
    "sources",
    "staging",
    "storage",
    "union",
    "utils",
]
