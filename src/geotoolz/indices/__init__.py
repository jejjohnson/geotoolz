"""Spectral indices for remote sensing.

The classics — NDVI, NDWI, NDBI, NBR, SAVI, EVI — plus the generic
``NormalizedDifference`` they all specialise, and ``AppendIndex`` for
stacking an index back as a new channel of the carrier.

Each Operator wraps a pure-numpy primitive in
:mod:`geotoolz.indices._src.array` so users who want the math without
the carrier can call the primitives directly. The Operators round-trip
``get_config()`` for Hydra-zen ``builds()``.

Examples:
    Eager mode::

        import geotoolz as gz
        ndvi = gz.indices.NDVI(nir_idx=7, red_idx=3)  # S2 band order
        v = ndvi(reflectance_geotensor)               # GeoTensor (H, W)

    Composition::

        pipeline = (
            gz.radiometry.DNToReflectance(scale=1e-4)
            | gz.indices.NDVI(nir_idx=7, red_idx=3)
        )
        v = pipeline(dn_geotensor)

    Stack the index back as an extra channel::

        op = gz.indices.AppendIndex(index_op=gz.indices.NDVI())
        stacked = op(reflectance_geotensor)  # shape (C+1, H, W)
"""

from __future__ import annotations

from geotoolz.indices._src.array import (
    arvi,
    bais2,
    bsi,
    ciri,
    clay_minerals,
    evi,
    evi2,
    gci,
    iron_oxide,
    kndvi,
    mndwi,
    nbr,
    nbr2,
    ndbi,
    ndmi,
    ndsi,
    ndvi,
    ndwi_mcfeeters,
    normalized_difference,
    savi,
)
from geotoolz.indices._src.operators import (
    ARVI,
    BAIS2,
    BSI,
    CIRI,
    EVI,
    EVI2,
    GCI,
    MNDWI,
    NBR,
    NBR2,
    NDBI,
    NDMI,
    NDSI,
    NDVI,
    NDWI,
    SAVI,
    AppendIndex,
    ClayMinerals,
    IronOxide,
    NormalizedDifference,
    dNBR,
    kNDVI,
)


__all__ = [
    "ARVI",
    "BAIS2",
    "BSI",
    "CIRI",
    "EVI",
    "EVI2",
    "GCI",
    "MNDWI",
    "NBR",
    "NBR2",
    "NDBI",
    "NDMI",
    "NDSI",
    "NDVI",
    "NDWI",
    "SAVI",
    "AppendIndex",
    "ClayMinerals",
    "IronOxide",
    "NormalizedDifference",
    "arvi",
    "bais2",
    "bsi",
    "ciri",
    "clay_minerals",
    "dNBR",
    "evi",
    "evi2",
    "gci",
    "iron_oxide",
    "kNDVI",
    "kndvi",
    "mndwi",
    "nbr",
    "nbr2",
    "ndbi",
    "ndmi",
    "ndsi",
    "ndvi",
    "ndwi_mcfeeters",
    "normalized_difference",
    "savi",
]
