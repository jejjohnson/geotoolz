"""`geoproducts` — readers for Earth-observation data products.

Each reader turns a mission's or provider's product into a georeader
`GeoData` / `GeoTensor`, so the geotoolz operators, the geopatcher
patchers and the geocatalog loaders consume it unchanged. The package
depends on georeader only: it installs without the operator library.

Public surface:

- `ProductReader` — the ABC for product readers (lazy ``_read_window``,
  band names, track, optional pooled object-store byte reads via the
  ``[obstore]`` extra).
- Per-sensor / per-provider subpackages, each exposing ``Reader``,
  ``BANDS`` / ``CONSTANTS`` and, with the ``[operators]`` extra,
  ``presets`` (geotoolz operators bound to the product's band names):
  ``toy_sensor`` is the in-memory reference reader.
"""

from __future__ import annotations

from geoproducts import toy_sensor
from geoproducts._src.base import ProductReader


__version__ = "0.0.0"  # x-release-please-version

__all__ = ["ProductReader", "__version__", "toy_sensor"]
