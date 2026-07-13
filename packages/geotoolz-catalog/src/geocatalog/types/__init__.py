"""`geocatalog.types` — cross-cutting wire-format dataclasses.

Hybrid-layout sub-namespace. Re-exports `GeoSlice` and its helpers
from the flat top-level surface, so both
``from geocatalog import GeoSlice`` and
``from geocatalog.types import GeoSlice`` work.
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
    GeoSlice,
    slice_to_window,
    window_to_slice,
)


__all__ = [
    "PIXEL_PRECISION",
    "Align",
    "GeoSlice",
    "GridAlignmentWarning",
    "divide_evenly",
    "is_grid_aligned",
    "slice_to_window",
    "window_to_slice",
]
