"""Compatibility re-export for mask morphology primitives."""

from __future__ import annotations

from xr_toolz.transforms._src.morphology import (
    Footprint,
    _resolve_footprint,
    binary_closing_2d,
    binary_opening_2d,
    clean_mask,
    remove_small_holes_2d,
    remove_small_objects_2d,
)


__all__ = [
    "Footprint",
    "_resolve_footprint",
    "binary_closing_2d",
    "binary_opening_2d",
    "clean_mask",
    "remove_small_holes_2d",
    "remove_small_objects_2d",
]
