"""`geocatalog.grid` — pixel grids: windows, alignment and even division.

- `slice_to_window` / `window_to_slice` — convert between a
  `geocatalog.GeoSlice` and a rasterio pixel window on a grid
  (`PIXEL_PRECISION` is the snapping tolerance).
- `Align`, `is_grid_aligned`, `GridAlignmentWarning` — check and snap a
  slice to a raster's pixel grid.
- `divide_evenly` — split a slice into equal, grid-aligned tiles.
"""

from __future__ import annotations

from geocatalog._src._align import (
    Align,
    GridAlignmentWarning,
    divide_evenly,
    is_grid_aligned,
)
from geocatalog._src.geoslice import (
    PIXEL_PRECISION,
    slice_to_window,
    window_to_slice,
)


__all__ = [
    "PIXEL_PRECISION",
    "Align",
    "GridAlignmentWarning",
    "divide_evenly",
    "is_grid_aligned",
    "slice_to_window",
    "window_to_slice",
]
