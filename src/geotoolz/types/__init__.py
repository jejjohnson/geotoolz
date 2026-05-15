"""`geotoolz.types` — cross-cutting wire-format dataclasses.

Currently exposes `GeoSlice`, the unit of work that flows between the
catalog, the patcher, and the readers. New cross-cutting types (e.g.
`GeoCoverage` for vector AOIs) will join here as they land.
"""

from __future__ import annotations

from geotoolz.types._src.geoslice import (
    PIXEL_PRECISION,
    GeoSlice,
    slice_to_window,
    window_to_slice,
)


__all__ = [
    "PIXEL_PRECISION",
    "GeoSlice",
    "slice_to_window",
    "window_to_slice",
]
