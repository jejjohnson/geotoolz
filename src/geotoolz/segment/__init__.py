"""Segmentation operators backed by :mod:`skimage.segmentation`."""

from __future__ import annotations

from geotoolz.segment._src.operators import (
    SLIC,
    ChanVese,
    ExpandLabels,
    Felzenszwalb,
    MarkBoundaries,
    Quickshift,
    RandomWalker,
    Watershed,
)


__all__ = [
    "SLIC",
    "ChanVese",
    "ExpandLabels",
    "Felzenszwalb",
    "MarkBoundaries",
    "Quickshift",
    "RandomWalker",
    "Watershed",
]
