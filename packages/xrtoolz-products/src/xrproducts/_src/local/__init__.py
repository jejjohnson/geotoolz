"""Local-file adapter for satellite L2 products already on disk."""

from xrproducts._src.local.catalog import LOCAL_DATASETS
from xrproducts._src.local.errors import MissingPropertyError, ProductGridError
from xrproducts._src.local.openers import (
    open_av3_ch4_l2b,
    open_ch4_inventory,
    open_ch4_plume_complexes,
    open_emit_ch4_l2b,
    open_ghgsat_ch4_l2,
    open_tropomi_ch4_l2,
    plume_outline,
)
from xrproducts._src.local.source import LocalL2Source


__all__ = [
    "LOCAL_DATASETS",
    "LocalL2Source",
    "MissingPropertyError",
    "ProductGridError",
    "open_av3_ch4_l2b",
    "open_ch4_inventory",
    "open_ch4_plume_complexes",
    "open_emit_ch4_l2b",
    "open_ghgsat_ch4_l2",
    "open_tropomi_ch4_l2",
    "plume_outline",
]
