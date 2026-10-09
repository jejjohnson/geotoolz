"""`geoproducts` — readers for Earth-observation data products.

Each reader turns a mission's or provider's product into a georeader
`GeoData` / `GeoTensor`, so the geotoolz operators, the geopatcher
patchers and the geocatalog loaders consume it unchanged. The package
depends on georeader only: it installs without the operator library.

Public surface:

- `ProductReader` — the ABC for product readers (lazy ``_read_window``,
  band names, track, optional pooled object-store byte reads via the
  ``[obstore]`` extra).
- `stack` — several readers' bands on one reference grid.
- Per-sensor / per-provider subpackages, each exposing ``Reader``,
  ``BANDS`` / ``CONSTANTS`` and, with the ``[operators]`` extra,
  ``presets`` (geotoolz operators bound to the product's band names):
  ``toy_sensor`` is the in-memory reference reader; ``goes`` reads
  GOES-R ABI L1b radiances and L2 products (``[goes]`` extra) and
  ``himawari`` Himawari-8 / -9 AHI HSD segments and L2 cloud products,
  both finding them in NOAA's public buckets.
"""

from __future__ import annotations

from geoproducts import goes, himawari, toy_sensor
from geoproducts._src.base import ProductReader
from geoproducts._src.stack import stack


__version__ = "0.3.0"  # x-release-please-version

__all__ = [
    "ProductReader",
    "__version__",
    "goes",
    "himawari",
    "stack",
    "toy_sensor",
]
