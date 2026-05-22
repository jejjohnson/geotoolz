"""Climate Data Store (CDS) adapter."""

from xrtoolz.data._src.cds.archive import (
    PRESET_TO_DATASET,
    ArchiveCoverage,
    CDSInsituArchive,
)
from xrtoolz.data._src.cds.profiles import INSITU, REANALYSIS, CDSFormProfile
from xrtoolz.data._src.cds.source import CDSSource


__all__ = [
    "INSITU",
    "PRESET_TO_DATASET",
    "REANALYSIS",
    "ArchiveCoverage",
    "CDSFormProfile",
    "CDSInsituArchive",
    "CDSSource",
]
