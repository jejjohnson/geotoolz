"""Local-file adapter for satellite L2 products already on disk."""

from xrreader._src.local.catalog import LOCAL_DATASETS
from xrreader._src.local.openers import (
    open_emit_ch4_l2b,
    open_ghgsat_ch4_l2,
    open_tropomi_ch4_l2,
)
from xrreader._src.local.source import LocalL2Source


__all__ = [
    "LOCAL_DATASETS",
    "LocalL2Source",
    "open_emit_ch4_l2b",
    "open_ghgsat_ch4_l2",
    "open_tropomi_ch4_l2",
]
