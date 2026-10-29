"""Climate Data Store (CDS) adapter."""

from xrproducts._src.cds.archive import (
    PRESET_TO_DATASET,
    ArchiveCoverage,
    CDSInsituArchive,
)
from xrproducts._src.cds.profiles import INSITU, REANALYSIS, CDSFormProfile
from xrproducts._src.cds.source import CDSSource


__all__ = [
    "INSITU",
    "PRESET_TO_DATASET",
    "REANALYSIS",
    "ArchiveCoverage",
    "CDSFormProfile",
    "CDSInsituArchive",
    "CDSSource",
]
