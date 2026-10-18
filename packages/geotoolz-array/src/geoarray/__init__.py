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
- `PixelWindow` — a rectangle of pixels (``col_off``, ``row_off``,
  ``width``, ``height``): rasterio's ``Window`` without rasterio.
- `FrozenAttrs` — the immutable, hashable metadata mapping;
  `PER_BAND_KEYS` names the keys that hold one entry per band.
- Geometry (re-index or re-describe pixels, never compute new values):
  `GeoArray.__getitem__` / `GeoArray.isel` (slices move the transform),
  `crop` (traced offsets), `pad`, `read_window` / `write_window`,
  `pixel_coords`, `GeoArray.valid_mask`, `GeoArray.map`, `combine`,
  `stack` / `concatenate`, `footprint` (``[vector]``).
- `RasterSource` / `AsyncRasterSource` — what a lazy reader provides:
  its `Grid`, shape, dtype, fill and attrs, and reads that return a
  `GeoArray` (``read_window``, ``load``). A `GeoArray` is one too.
- `as_transform` / `as_crs` — coerce an ``affine.Affine``, six numbers, an
  EPSG code or WKT.
- Backends (C1): NumPy and JAX carriers mix eagerly. ``np.asarray(g)``
  copies JAX data to the host (and raises `GeoArrayTraceError` under a
  trace); joins and writes promote NumPy inputs to JAX. `set_options`
  (``warn_on_transfer``) / `get_options`, `HostTransferWarning`.
- `geoarray.interop.georeader` — ``from_georeader`` / ``to_georeader``,
  zero-copy bridges to georeader's ``GeoTensor``, and
  ``source_from_georeader`` / ``source_to_georeader`` for its lazy readers
  (``[georeader]``).
- Errors: `GeoArrayError` and its subclasses `X64Error`,
  `GeoArrayTraceError` and `GridMismatchError`.

JAX is optional (``geotoolz-array[jax]``); without it, geoarray never
imports JAX.
"""

from __future__ import annotations

from geoarray._src.array import DIMS, GeoArray
from geoarray._src.attrs import PER_BAND_KEYS, FrozenAttrs
from geoarray._src.errors import (
    GeoArrayError,
    GeoArrayTraceError,
    GridMismatchError,
    X64Error,
)
from geoarray._src.geometry import (
    combine,
    concatenate,
    crop,
    footprint,
    pad,
    pixel_coords,
    read_window,
    stack,
    write_window,
)
from geoarray._src.grid import Grid, PixelWindow, as_crs, as_transform
from geoarray._src.options import HostTransferWarning, get_options, set_options
from geoarray._src.source import AsyncRasterSource, RasterSource


__version__ = "0.1.0"  # x-release-please-version

__all__ = [
    "DIMS",
    "PER_BAND_KEYS",
    "AsyncRasterSource",
    "FrozenAttrs",
    "GeoArray",
    "GeoArrayError",
    "GeoArrayTraceError",
    "Grid",
    "GridMismatchError",
    "HostTransferWarning",
    "PixelWindow",
    "RasterSource",
    "X64Error",
    "__version__",
    "as_crs",
    "as_transform",
    "combine",
    "concatenate",
    "crop",
    "footprint",
    "get_options",
    "pad",
    "pixel_coords",
    "read_window",
    "set_options",
    "stack",
    "write_window",
]
