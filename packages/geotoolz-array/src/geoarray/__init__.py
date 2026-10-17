"""`geoarray` — one georeferenced array for the geotoolz stack.

A `GeoArray` holds a NumPy or a JAX array with its affine transform, CRS,
fill value and hashable attrs. It is a JAX pytree whose leaves are the data
and a float64 transform, so ``jit`` reuses compiled code across scenes and
``grad`` reaches the georeferencing.

- `GeoArray` — the carrier: dims ``("time", "band", "y", "x")[-ndim:]``,
  `GeoArray.grid`, `GeoArray.to_jax` / `GeoArray.to_numpy`,
  `GeoArray.with_data` to rewrap a result.
- `Grid` — a pixel grid with no data: transform, CRS and ``(H, W)``;
  hashable, so it can be a warp target or static config.
- `PixelWindow` — a rectangle of pixels (``row_off``, ``col_off``,
  ``height``, ``width``): rasterio's ``Window`` without rasterio.
- `FrozenAttrs` — the immutable, hashable metadata mapping.
- `as_transform` / `as_crs` — coerce an ``affine.Affine``, six numbers, an
  EPSG code or WKT.
- Errors: `GeoArrayError` and its subclasses `X64Error` and
  `GeoArrayTraceError`.

JAX is optional (``geotoolz-array[jax]``); without it, geoarray never
imports JAX.
"""

from __future__ import annotations

from geoarray._src.array import DIMS, GeoArray
from geoarray._src.attrs import FrozenAttrs
from geoarray._src.errors import GeoArrayError, GeoArrayTraceError, X64Error
from geoarray._src.grid import Grid, PixelWindow, as_crs, as_transform


__version__ = "0.0.0"  # x-release-please-version

__all__ = [
    "DIMS",
    "FrozenAttrs",
    "GeoArray",
    "GeoArrayError",
    "GeoArrayTraceError",
    "Grid",
    "PixelWindow",
    "X64Error",
    "__version__",
    "as_crs",
    "as_transform",
]
