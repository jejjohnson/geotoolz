"""Cloud / mask helpers for Sentinel-2 SCL and Landsat-style QA layers.

Mask extraction (`MaskFromQABits`, `MaskFromSCL`, `MaskValid`) plus
generic mask application (`ApplyMask`). The `SCL` IntEnum and a few
convenience sets (``SCL_CLOUDS``, ``SCL_INVALID``, ``SCL_LAND``,
``SCL_WATER``) cover the standard Sentinel-2 L2A scene-classification
groupings.

This module deliberately stays narrow: ML-based detectors
(``s2cloudless``, FMask, learned per-pixel classifiers) belong in a
later ``cloud-ml`` extra. What's here covers ≥ 80 % of "drop cloudy
pixels before computing an index" workflows.

Examples:
    Standard S2 L2A cloud-clean NDVI::

        import geotoolz as gz

        pipeline = (
            gz.radiometry.DNToReflectance(scale=1e-4)
            | gz.cloud.ApplyMask(
                mask=gz.cloud.MaskFromSCL(
                    band_idx=-1, classes=gz.cloud.SCL_CLOUDS,
                ),
            )
            | gz.indices.NDVI(nir_idx=7, red_idx=3)
        )
        ndvi = pipeline(s2_dn_geotensor)

    Landsat-8 QA_PIXEL clear mask::

        clear = gz.cloud.MaskFromQABits(
            band_idx=-1, bits=[2, 3, 4], invert=True,  # keep where cloud bits OFF
        )
        is_clear = clear(landsat_stack)
"""

from __future__ import annotations

from geotoolz.cloud._src.array import (
    apply_mask,
    mask_from_qa_bits,
    mask_from_scl,
)
from geotoolz.cloud._src.operators import (
    ApplyMask,
    MaskFromQABits,
    MaskFromSCL,
    MaskValid,
)
from geotoolz.cloud._src.scl import (
    SCL,
    SCL_CLOUDS,
    SCL_CLOUDS_AND_INVALID,
    SCL_INVALID,
    SCL_LAND,
    SCL_WATER,
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
