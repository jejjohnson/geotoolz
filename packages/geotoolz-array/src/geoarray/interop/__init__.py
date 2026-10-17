"""Bridges between `GeoArray` and other libraries, one module per library.

- `geoarray.interop.georeader` — ``from_georeader`` / ``to_georeader``
  (``geotoolz-array[georeader]``).

Each module imports its library only when a function runs, so importing
this package needs none of them.
"""

from __future__ import annotations
