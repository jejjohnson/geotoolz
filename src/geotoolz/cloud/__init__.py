"""DEPRECATED compatibility alias — the cloud module's contents moved.

The mask-*extraction* half (``MaskFromQABits``, ``MaskFromSCL``,
``MaskValid``, the ``SCL`` enum + convenience sets, and the
``mask_from_qa_bits`` / ``mask_from_scl`` primitives) now lives in
`geotoolz.qa` alongside the sensor presets; the mask-*application* half
(``ApplyMask``, ``apply_mask``) lives in `geotoolz.mask` with the rest
of the mask algebra. Import from those modules going forward — this
alias re-exports the original names unchanged and will be removed in a
future release.
"""

from __future__ import annotations

from geotoolz.mask import ApplyMask, apply_mask
from geotoolz.qa import (
    SCL,
    SCL_CLOUDS,
    SCL_CLOUDS_AND_INVALID,
    SCL_INVALID,
    SCL_LAND,
    SCL_WATER,
    MaskFromQABits,
    MaskFromSCL,
    MaskValid,
    mask_from_qa_bits,
    mask_from_scl,
)


__all__ = [
    "SCL",
    "SCL_CLOUDS",
    "SCL_CLOUDS_AND_INVALID",
    "SCL_INVALID",
    "SCL_LAND",
    "SCL_WATER",
    "ApplyMask",
    "MaskFromQABits",
    "MaskFromSCL",
    "MaskValid",
    "apply_mask",
    "mask_from_qa_bits",
    "mask_from_scl",
]
