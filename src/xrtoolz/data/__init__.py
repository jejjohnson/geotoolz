"""Data downloaders for external geoscience archives.

Data adapters here translate typed ``xrtoolz`` requests
(:class:`~xrtoolz.types.BBox`, :class:`~xrtoolz.types.TimeRange`,
:class:`~xrtoolz.types.Variable`, ...) into source-specific payloads,
so all the type work lives in :mod:`xrtoolz.types` and this module
just speaks the language of each underlying service.

Exports:

- :class:`DataSource`, :class:`DatasetInfo`, :class:`DatasetKind`
- Adapters: :class:`CMEMSSource`, :class:`CDSSource`, :class:`AemetSource`
- Credentials: :class:`CMEMSCredentials`, :class:`CDSCredentials`,
  :class:`AEMETCredentials`, :func:`load_cmems`, :func:`load_cds`,
  :func:`load_aemet`.
- Catalog: :data:`CATALOG`, :class:`CatalogEntry`, :func:`all_entries`,
  :func:`describe`.
- AEMET extras: :class:`AemetArchive`, :class:`AemetError`,
  :class:`AemetAuthError`, :class:`AemetRateLimitError`.
- CDS extras: :class:`CDSInsituArchive`, :class:`CDSFormProfile`,
  :data:`INSITU`, :data:`REANALYSIS`.
"""

from xrtoolz.data._src.aemet import (
    AemetArchive,
    AemetAuthError,
    AemetError,
    AemetRateLimitError,
    AemetSource,
    ArchiveCoverage,
)
from xrtoolz.data._src.base import DatasetInfo, DatasetKind, DataSource
from xrtoolz.data._src.catalog import CATALOG, CatalogEntry, all_entries, describe
from xrtoolz.data._src.cds import (
    INSITU,
    REANALYSIS,
    CDSFormProfile,
    CDSInsituArchive,
    CDSSource,
)
from xrtoolz.data._src.cmems import CMEMSSource
from xrtoolz.data._src.credentials import (
    AEMETCredentials,
    CDSCredentials,
    CMEMSCredentials,
    load_aemet,
    load_cds,
    load_cmems,
)


__all__ = [
    "CATALOG",
    "INSITU",
    "REANALYSIS",
    "AEMETCredentials",
    "AemetArchive",
    "AemetAuthError",
    "AemetError",
    "AemetRateLimitError",
    "AemetSource",
    "ArchiveCoverage",
    "CDSCredentials",
    "CDSFormProfile",
    "CDSInsituArchive",
    "CDSSource",
    "CMEMSCredentials",
    "CMEMSSource",
    "CatalogEntry",
    "DataSource",
    "DatasetInfo",
    "DatasetKind",
    "all_entries",
    "describe",
    "load_aemet",
    "load_cds",
    "load_cmems",
]
