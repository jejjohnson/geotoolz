"""QA bit decoding, SCL classes, and sensor mask presets.

The operators in this module return boolean masks where ``True`` means
"mask this pixel out". Use them with ``geotoolz.mask.ApplyMask`` or
downstream mask-combination operators.

Two operator families:

- Generic extraction (`MaskFromQABits`, `MaskFromSCL`, `MaskValid`,
  `DecodeBitmask`) — pass explicit bits / SCL classes.
- Sensor presets (`S2QA60`, `S2SCL`, `LandsatQA_PIXEL`, `MODISStateQA`,
  `MaskClouds`, ...) — published bit/class layouts baked in.
"""

from __future__ import annotations

from geotoolz.qa._src.array import (
    mask_from_bit_field,
    mask_from_qa_bits,
    mask_from_scl,
    reduce_bit_masks,
)
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
    MaskFromQABits,
    MaskFromSCL,
    MaskNoData,
    MaskSaturated,
    MaskSnow,
    MaskValid,
    MaskWater,
    MODISStateQA,
    OmniCloudMask,
    S2Cloudless,
)
from geotoolz.qa._src.scl import (
    SCL,
    SCL_CLOUDS,
    SCL_CLOUDS_AND_INVALID,
    SCL_INVALID,
    SCL_LAND,
    SCL_WATER,
)


__all__ = [
    "S2QA60",
    "S2SCL",
    "SCL",
    "SCL_CLOUDS",
    "SCL_CLOUDS_AND_INVALID",
    "SCL_INVALID",
    "SCL_LAND",
    "SCL_WATER",
    "SENSOR_QA_REGISTRY",
    "CloudSEN12",
    "DecodeBitmask",
    "LandsatQA_PIXEL",
    "MODISStateQA",
    "MaskCirrus",
    "MaskCloudShadow",
    "MaskClouds",
    "MaskFromQABits",
    "MaskFromSCL",
    "MaskNoData",
    "MaskSaturated",
    "MaskSnow",
    "MaskValid",
    "MaskWater",
    "OmniCloudMask",
    "S2Cloudless",
    "mask_from_bit_field",
    "mask_from_qa_bits",
    "mask_from_scl",
    "reduce_bit_masks",
]
