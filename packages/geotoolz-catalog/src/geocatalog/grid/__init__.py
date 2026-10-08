"""`geocatalog.grid` — pixel grids: windows, alignment and exact pixel counts.

- `slice_to_window` / `window_to_slice` — convert between a
  `geocatalog.GeoSlice` and a rasterio pixel window on a grid
  (`PIXEL_PRECISION` is the snapping tolerance).
- `Align`, `is_grid_aligned`, `GridAlignmentWarning` — check and snap a
  slice to a raster's pixel grid.
- `count_steps` — how many whole steps (pixels) of a resolution fit in a
  length, raising when the division is off by more than a fraction of a
  pixel.
"""

from __future__ import annotations

from geocatalog._src.geoslice import (
    PIXEL_PRECISION,
    slice_to_window,
    window_to_slice,
)
from geocatalog._src.grid import (
    Align,
    GridAlignmentWarning,
    count_steps,
    is_grid_aligned,
)


__all__ = [
    "PIXEL_PRECISION",
    "Align",
    "GridAlignmentWarning",
    "count_steps",
    "is_grid_aligned",
    "slice_to_window",
    "window_to_slice",
]
