"""Segmentation operators backed by :mod:`skimage.segmentation`."""

from __future__ import annotations

from geotoolz.segment._src.array import (
    fill_invalid,
    mask_nms,
    merge_nearby_instances,
    otsu_threshold,
    resolve_threshold,
    threshold_mask,
)
from geotoolz.segment._src.operators import (
    SLIC,
    ChanVese,
    ExpandLabels,
    Felzenszwalb,
    MarkBoundaries,
    MaskNMS,
    MergeNearbyInstances,
    Quickshift,
    RandomWalker,
    Threshold,
    Watershed,
)


__all__ = [
    "SLIC",
    "ChanVese",
    "ExpandLabels",
    "Felzenszwalb",
    "MarkBoundaries",
    "MaskNMS",
    "MergeNearbyInstances",
    "Quickshift",
    "RandomWalker",
    "Threshold",
    "Watershed",
    "fill_invalid",
    "mask_nms",
    "merge_nearby_instances",
    "otsu_threshold",
    "resolve_threshold",
    "threshold_mask",
]
