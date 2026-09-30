"""Explicit compositing operators for co-registered GeoTensor stacks."""

from __future__ import annotations

from geotoolz.compositing._src.array import (
    bap_scores,
    blend_weighted,
    cloud_distance_score,
    doy_distance,
    doy_score,
    mask_frames,
    mean_composite,
    median_composite,
    opacity_score,
    take_by_spatial_index,
    view_angle_score,
)
from geotoolz.compositing._src.operators import (
    BAPComposite,
    BlendMatched,
    CloudFreeComposite,
    MaxNDVIComposite,
    MedianComposite,
    MinCloudComposite,
    StackMatched,
)


__all__ = [
    "BAPComposite",
    "BlendMatched",
    "CloudFreeComposite",
    "MaxNDVIComposite",
    "MedianComposite",
    "MinCloudComposite",
    "StackMatched",
    "bap_scores",
    "blend_weighted",
    "cloud_distance_score",
    "doy_distance",
    "doy_score",
    "mask_frames",
    "mean_composite",
    "median_composite",
    "opacity_score",
    "take_by_spatial_index",
    "view_angle_score",
]
