"""georeader's ``GeoTensor`` ↔ `GeoArray`, sharing memory (NumPy only).

Needs the ``[georeader]`` extra: ``pip install 'geotoolz-array[georeader]'``.

- `from_georeader` — a ``GeoTensor`` → a NumPy `GeoArray`.
- `to_georeader` — a NumPy `GeoArray` → a ``GeoTensor``.
"""

from __future__ import annotations

from geoarray._src.georeader import from_georeader, to_georeader


__all__ = ["from_georeader", "to_georeader"]
