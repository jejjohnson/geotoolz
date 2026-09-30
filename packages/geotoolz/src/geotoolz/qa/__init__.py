"""QA bit decoding, SCL classes, and sensor mask presets.

The operators in this module return boolean masks where ``True`` means
"mask this pixel out". Use them with ``geotoolz.mask.ApplyMask`` or
downstream mask-combination operators.

Two operator families:

- Generic extraction (`MaskClouds`, `MaskCloudShadow`, `MaskCirrus`,
  `MaskSnow`, `MaskWater`, `MaskNoData`, `MaskInvalid`, `MaskSaturated`,
  `DecodeBitmask`) — pass explicit ``bits`` / ``values`` with an optional
  ``qa_band`` (``None`` = the carrier is the QA band).
- Sensor presets (`S2QA60`, `S2SCL`, `LandsatQA_PIXEL`, `MODISStateQA`)
  — published bit/class layouts baked in; pick ``targets`` by name.
"""

from __future__ import annotations

from geotoolz.qa._src.array import (
    mask_from_bit_field,
    mask_from_qa_bits,
    mask_from_scl,
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
    MaskInvalid,
    MaskNoData,
    MaskSaturated,
    MaskSnow,
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
    "MaskInvalid",
    "MaskNoData",
    "MaskSaturated",
    "MaskSnow",
    "MaskWater",
    "OmniCloudMask",
    "S2Cloudless",
    "mask_from_bit_field",
    "mask_from_qa_bits",
    "mask_from_scl",
]
