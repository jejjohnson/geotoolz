"""`geocatalog.build` — index raster, array and vector files into a catalog.

- `build_raster_catalog` — GeoTIFF / COG footprints, CRS and resolution,
  with the time parsed from the filename (``filename_regex`` +
  ``date_format``).
- `build_xarray_catalog` — NetCDF / Zarr variables (``[xarray-raster]``
  extra; loads lazily).
- `build_vector_catalog` — vector files (``[vector]`` extra; loads
  lazily).
- `append_files` — add files to an existing GeoParquet catalog without
  rebuilding it.

```python
from glob import glob

from geocatalog.build import build_raster_catalog

catalog = build_raster_catalog(
    sorted(glob("scenes/*.tif")),        # e.g. scenes/S2_20240601.tif
    filename_regex=r"_(?P<date>\\d{8})",
    date_format="%Y%m%d",
)
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from geocatalog._src._lazy import lazy_getattr
from geocatalog._src.raster import build_raster_catalog
from geocatalog._src.streaming import append_files


if TYPE_CHECKING:
    from geocatalog._src.vector import build_vector_catalog
    from geocatalog._src.xarray_backend import build_xarray_catalog


__all__ = [
    "append_files",
    "build_raster_catalog",
    "build_vector_catalog",
    "build_xarray_catalog",
]


# Extras-gated names resolve lazily (`geocatalog._src._lazy.LAZY`).
__getattr__ = lazy_getattr(
    globals(),
    [
        "build_vector_catalog",
        "build_xarray_catalog",
    ],
)
