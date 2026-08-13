"""AEMET OpenData adapter (Spanish national meteorological service)."""

from xrreader._src.aemet.archive import AemetArchive, ArchiveCoverage
from xrreader._src.aemet.catalog import AEMET_DATASETS
from xrreader._src.aemet.source import (
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
