"""`geoarray` — one georeferenced array for the geotoolz stack.

A `GeoArray` holds a NumPy or a JAX array with its affine transform, CRS,
fill value and hashable attrs. It is a JAX pytree whose leaves are the data
and a float64 transform, so ``jit`` reuses compiled code across scenes and
``grad`` reaches the georeferencing.

- `GeoArray` — the carrier: stored dims ending in ``("y", "x")``
  (``GeoArray(..., dims="band y x")``; read from the rank with a
  ``DeprecationWarning`` when omitted), coordinates per dim
  (`geoarray.coords`), `GeoArray.grid`, `GeoArray.sel` / `GeoArray.isel` by
  name, `GeoArray.to_jax` / `GeoArray.to_numpy`, `GeoArray.with_data` to
  rewrap a result, `GeoArray.from_coords`.
- The dim-level functions, shared with every named carrier (they are
  pipekit-array's: ``geoarray.reduce is pipekit_array.reduce``): `isel`,
  `sel`, `transpose`, `reduce`, `argreduce`, `take_along`, `map_over`,
  `concat`, `combine`. geoarray's `GeoArrayAdapter` is registered with
  pipekit at import, so they, and pipekit-array's ``KernelOp``, run on
  carriers; a result that consumed ``y`` or ``x`` is a
  ``pipekit_array.DimArray`` (`to_dimarray`).
- `GeoSwath` (per-pixel lat/lon on ``row col``) and `GeoPoints` (xy, and
  optional z, time and valid on ``obs``): the carriers whose georeference
  is not an affine grid; `pad_points` pads a point set to a fixed N.
- `Grid` — a pixel grid with no data: transform, CRS and ``(H, W)``;
  hashable, so it can be a warp target or static config.
- `PixelWindow` — a rectangle of pixels (``col_off``, ``row_off``,
  ``width``, ``height``): rasterio's ``Window`` without rasterio.
- `FrozenAttrs` — the immutable, hashable metadata mapping;
  `PER_BAND_KEYS` names the keys that hold one entry per band.
- Geometry: operations that read only the georeference, in three families
  (anything that writes a new grid — warp, resample, reproject — is
  geoops):

  - **re-index** — `GeoArray.__getitem__` / `GeoArray.isel` (slices move
    the transform), `crop` (traced offsets), `pad`, `read_window` /
    `write_window`, and the joins `combine`, `stack` / `concat`
    (`concatenate`, by position);
  - **re-describe** — coordinates and dims, `pixel_coords`,
    `GeoArray.valid_mask`, `GeoArray.map`, `footprint` (``[vector]``);
  - **read at** — `to_pixel` (map coordinates → fractional pixel centres)
    and `interp_at` (H(x): a grid read at `GeoPoints`, bilinear or
    nearest, padding-safe): points in, values out.
- `RasterSource` / `AsyncRasterSource` — what a lazy reader provides:
  its `Grid`, shape, dtype, fill and attrs, and reads that return a
  `GeoArray` (``read_window``, ``load``). A `GeoArray` is one too.
- `as_transform` / `as_crs` — coerce an ``affine.Affine``, six numbers, an
  EPSG code or WKT.
- Backends (C1): NumPy and JAX carriers mix eagerly. ``np.asarray(g)``
  copies JAX data to the host (and raises `GeoArrayTraceError` under a
  trace); joins and writes promote NumPy inputs to JAX. `set_options`
  (``warn_on_transfer``) / `get_options`, `HostTransferWarning`.
- `geoarray.coords` — the standard coordinates of named dims:
  `SpectralAxis`, `VariableAxis`, `VerticalAxis` / `Vertical`, `TimeAxis`,
  and the grids `PixelGrid`, `SwathGrid`, `CurvilinearGrid`, `PointAxis`.
- `geoarray.types` — readable names for the common shapes (`GeoImage`,
  `GeoMapSeries`, `GeoTracks`, …): a dims pattern and a geometry kind any
  carrier can match; `check_dims` checks them at the call, and
  ``GeoArray["band y x"]`` annotates a pattern inline.
- `geoarray.interop.georeader` — ``from_georeader`` / ``to_georeader``,
  zero-copy bridges to georeader's ``GeoTensor``, and
  ``source_from_georeader`` / ``source_to_georeader`` for its lazy readers
  (``[georeader]``).
- Errors: `GeoArrayError` and its subclasses `X64Error`,
  `GeoArrayTraceError`, `GridMismatchError` and `LabelError`.

JAX is optional (``geotoolz-array[jax]``); without it, geoarray never
imports JAX.
"""

from __future__ import annotations

from pipekit_array import (
    argreduce,
    combine,
    concat,
    isel,
    map_over,
    reduce,
    sel,
    take_along,
    transpose,
)

from geoarray._src.adapter import (
    GeoArrayAdapter,
    GeoPointsAdapter,
    GeoSwathAdapter,
    to_dimarray,
)
from geoarray._src.array import DIMS, SPATIAL_DIMS, GeoArray
from geoarray._src.attrs import PER_BAND_KEYS, FrozenAttrs
from geoarray._src.carriers import GeoPoints, GeoSwath, pad_points
from geoarray._src.errors import (
    GeoArrayError,
    GeoArrayTraceError,
    GridMismatchError,
    LabelError,
    X64Error,
)
from geoarray._src.geometry import (
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
from geoarray._src.sampling import interp_at, to_pixel
from geoarray._src.source import AsyncRasterSource, RasterSource


__version__ = "0.2.1"  # x-release-please-version

__all__ = [
    "DIMS",
    "PER_BAND_KEYS",
    "SPATIAL_DIMS",
    "AsyncRasterSource",
    "FrozenAttrs",
    "GeoArray",
    "GeoArrayAdapter",
    "GeoArrayError",
    "GeoArrayTraceError",
    "GeoPoints",
    "GeoPointsAdapter",
    "GeoSwath",
    "GeoSwathAdapter",
    "Grid",
    "GridMismatchError",
    "HostTransferWarning",
    "LabelError",
    "PixelWindow",
    "RasterSource",
    "X64Error",
    "__version__",
    "argreduce",
    "as_crs",
    "as_transform",
    "combine",
    "concat",
    "concatenate",
    "crop",
    "footprint",
    "get_options",
    "interp_at",
    "isel",
    "map_over",
    "pad",
    "pad_points",
    "pixel_coords",
    "read_window",
    "reduce",
    "sel",
    "set_options",
    "stack",
    "take_along",
    "to_dimarray",
    "to_pixel",
    "transpose",
    "write_window",
]
