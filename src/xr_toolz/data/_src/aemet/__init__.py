"""AEMET OpenData adapter (Spanish national meteorological service)."""

from xr_toolz.data._src.aemet.archive import AemetArchive, ArchiveCoverage
from xr_toolz.data._src.aemet.catalog import AEMET_DATASETS
from xr_toolz.data._src.aemet.source import (
    AemetAuthError,
    AemetError,
    AemetRateLimitError,
    AemetSource,
)


__all__ = [
    "AEMET_DATASETS",
    "AemetArchive",
    "AemetAuthError",
    "AemetError",
    "AemetRateLimitError",
    "AemetSource",
    "ArchiveCoverage",
]
