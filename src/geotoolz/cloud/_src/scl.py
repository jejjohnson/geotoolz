"""Sentinel-2 Scene Classification Layer (SCL) class enum + convenience sets.

The SCL band is part of the Sentinel-2 *Level-2A* product (Sen2Cor's
output) and assigns each pixel one of 12 categorical labels. It's the
canonical first cut for cloud / shadow / snow / water masking when
working with L2A; the more sophisticated alternatives (s2cloudless,
FMask, ML-based detectors) live downstream of this.

Per Sen2Cor's product specification (ESA, S2-PDGS-MPC-L2A-DPM):

============  ===========================================
SCL Value     Class
============  ===========================================
``0``         NO_DATA
``1``         SATURATED_OR_DEFECTIVE
``2``         DARK_AREA_PIXELS / cast shadows
``3``         CLOUD_SHADOWS
``4``         VEGETATION
``5``         NOT_VEGETATED (bare soil / desert)
``6``         WATER
``7``         UNCLASSIFIED
``8``         CLOUD_MEDIUM_PROBABILITY
``9``         CLOUD_HIGH_PROBABILITY
``10``        THIN_CIRRUS
``11``        SNOW / ICE
============  ===========================================

The convenience frozensets at the bottom of this module bundle the
common groupings: ``SCL_CLOUDS`` for everything cloud-like (medium,
high, cirrus), ``SCL_INVALID`` for pixels you'd want to drop entirely
(no-data, defective, shadows), and ``SCL_LAND`` / ``SCL_WATER`` for
crude land-water separation.
"""

from __future__ import annotations

from enum import IntEnum


class SCL(IntEnum):
    """Sentinel-2 L2A Scene Classification Layer class IDs."""

    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    DARK_AREA_PIXELS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW = 11


#: Cloud-like classes — what you'd typically mask out before NDVI etc.
SCL_CLOUDS: frozenset[int] = frozenset(
    {SCL.CLOUD_MEDIUM_PROBABILITY, SCL.CLOUD_HIGH_PROBABILITY, SCL.THIN_CIRRUS}
)

#: Pixels you'd want to drop entirely (no-data + defective + shadows).
SCL_INVALID: frozenset[int] = frozenset(
    {SCL.NO_DATA, SCL.SATURATED_OR_DEFECTIVE, SCL.CLOUD_SHADOWS}
)

#: Cloud + invalid combined — "everything I don't want for vegetation analysis".
SCL_CLOUDS_AND_INVALID: frozenset[int] = SCL_CLOUDS | SCL_INVALID

#: Land-cover classes (vegetation + bare soil).
SCL_LAND: frozenset[int] = frozenset({SCL.VEGETATION, SCL.NOT_VEGETATED})

#: Water class (singleton — included for symmetry).
SCL_WATER: frozenset[int] = frozenset({SCL.WATER})
