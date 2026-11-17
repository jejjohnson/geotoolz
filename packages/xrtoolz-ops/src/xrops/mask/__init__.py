"""Masks — land/ocean/country masks, mask application, and canonical regimes.

Layer-1 ``Operator`` wrappers live in :mod:`xrops.mask.operators`.
"""

from __future__ import annotations

from xrops.mask._src.masks import (
    add_country_mask,
    add_land_mask,
    add_ocean_mask,
    apply_mask,
)
from xrops.mask.operators import (
    AddCountryMask,
    AddLandMask,
    AddOceanMask,
    ApplyMask,
)


__all__ = [
    "AddCountryMask",
    "AddLandMask",
    "AddOceanMask",
    "ApplyMask",
    "add_country_mask",
    "add_land_mask",
    "add_ocean_mask",
    "apply_mask",
]
