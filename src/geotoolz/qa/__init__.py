"""QA bit decoding and sensor mask presets.

The operators in this module return boolean ``GeoTensor`` masks where
``True`` means "mask this pixel out". Use them with ``geotoolz.cloud.ApplyMask``
or downstream mask-combination operators.
"""

from __future__ import annotations

from geotoolz.qa._src.operators import (
    S2QA60,
    S2SCL,
    SENSOR_QA_REGISTRY,
    CloudSEN12,
    DecodeBitmask,
    LandsatQA_PIXEL,
    MaskCirrus,
    MaskClouds,
    MaskCloudShadow,
    MaskNoData,
    MaskSaturated,
    MaskSnow,
    MaskWater,
    MODISStateQA,
    OmniCloudMask,
    S2Cloudless,
)


__all__ = [
    "S2QA60",
    "S2SCL",
    "SENSOR_QA_REGISTRY",
    "CloudSEN12",
    "DecodeBitmask",
    "LandsatQA_PIXEL",
    "MODISStateQA",
    "MaskCirrus",
    "MaskCloudShadow",
    "MaskClouds",
    "MaskNoData",
    "MaskSaturated",
    "MaskSnow",
    "MaskWater",
    "OmniCloudMask",
    "S2Cloudless",
]
