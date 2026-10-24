"""`geoproducts` — readers for Earth-observation data products.

Each reader turns a mission's or provider's product into a lazy
`geoarray.RasterSource` whose reads return a `geoarray.GeoArray`, so the
geotoolz operators and the geopatcher patchers consume it unchanged. The
package installs without the operator library; georeader is only its
internal engine (map-coordinate windows, reprojection), never in its API.

Public surface:

- `ProductReader` — the ABC for product readers: a `geoarray.RasterSource`
  (``grid``, ``read_window``, ``load``) plus ``read_from_bounds`` /
  ``read_from_center_coords``, band names and track; subclasses implement
  ``_read_window(PixelWindow)``. File formats and object storage come from
  geotoolz-cloud, through each reader's extra.
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


__version__ = "0.7.0"  # x-release-please-version

__all__ = [
    "ProductReader",
    "__version__",
    "goes",
    "himawari",
    "stack",
    "toy_sensor",
]
