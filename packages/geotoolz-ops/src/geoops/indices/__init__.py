"""Spectral indices for remote sensing.

The classics — NDVI, NDWI, NDBI, NBR, SAVI, EVI — plus the generic
``NormalizedDifference`` they all specialise, and ``AppendIndex`` for
stacking an index back as a new channel of the carrier.

Each Operator runs an element-wise kernel from
:mod:`geoops.indices._src.array` through pipekit-array's ``KernelOp``,
so the same operator works on a NumPy or JAX ``GeoArray`` (under
``jax.jit`` too), on a lazy dask-backed ``xr.DataArray`` and on a plain
array. Users who want the maths without a carrier call the kernels
directly, one array per band. The Operators round-trip ``get_config()``
for Hydra-zen ``builds()``.

Examples:
    Eager mode::

        import geoops as gz
        ndvi = gz.indices.NDVI(nir=7, red=3)  # S2 band order
        v = ndvi(reflectance_geoarray)               # GeoArray (H, W)

    Composition::

        pipeline = (
            gz.radiometry.DNToReflectance(scale=1e-4)
            | gz.indices.NDVI(nir=7, red=3)
        )
        v = pipeline(dn_geoarray)

    The same operator on a lazy cube and under ``jax.jit``::

        cube = xr.open_zarr(store)["reflectance"]  # (time, band, y, x), dask
        ndvi = gz.indices.NDVI(nir="B8", red="B4")(cube)  # lazy (time, y, x)
        step = jax.jit(gz.indices.NDVI(nir=7, red=3))
        v = step(reflectance_geoarray.to_jax())   # GeoArray (H, W), JAX

    Stack the index back as an extra channel::

        op = gz.indices.AppendIndex(index_op=gz.indices.NDVI())
        stacked = op(reflectance_geoarray)  # shape (C+1, H, W)
"""

from __future__ import annotations

from geoops.indices._src.array import (
    arvi,
    bais2,
    bsi,
    ciri,
    clay_minerals,
    dnbr,
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
from geoops.indices._src.operators import (
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
    "dnbr",
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
