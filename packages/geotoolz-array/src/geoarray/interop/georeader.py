"""georeader's ``GeoTensor`` ↔ `GeoArray`, sharing memory (NumPy only).

Needs the ``[georeader]`` extra: ``pip install 'geotoolz-array[georeader]'``.

- `from_georeader` — a ``GeoTensor`` → a NumPy `GeoArray`.
- `to_georeader` — a NumPy `GeoArray` → a ``GeoTensor``.
- `source_from_georeader` — a lazy georeader ``GeoData`` → a `RasterSource`.
- `source_to_georeader` — a `RasterSource` → a ``GeoData``, so georeader's
  ``read`` functions can read it.
"""

from __future__ import annotations

from geoarray._src.georeader import (
    from_georeader,
    source_from_georeader,
    source_to_georeader,
    to_georeader,
)


__all__ = [
    "from_georeader",
    "source_from_georeader",
    "source_to_georeader",
    "to_georeader",
]
