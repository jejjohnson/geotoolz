"""Climate Data Store (CDS) adapter."""

from xr_toolz.data._src.cds.archive import (
    PRESET_TO_DATASET,
    ArchiveCoverage,
    CDSInsituArchive,
)
from xr_toolz.data._src.cds.profiles import INSITU, REANALYSIS, CDSFormProfile
from xr_toolz.data._src.cds.source import CDSSource


__all__ = [
    "INSITU",
    "PRESET_TO_DATASET",
    "REANALYSIS",
    "ArchiveCoverage",
    "CDSFormProfile",
    "CDSInsituArchive",
    "CDSSource",
]
