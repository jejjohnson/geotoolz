"""Data downloaders for external geoscience archives.

Data adapters here translate typed ``xr_toolz`` requests
(:class:`~xr_toolz.types.BBox`, :class:`~xr_toolz.types.TimeRange`,
:class:`~xr_toolz.types.Variable`, ...) into source-specific payloads,
so all the type work lives in :mod:`xr_toolz.types` and this module
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

from xr_toolz.data._src.aemet import (
    AemetArchive,
    AemetAuthError,
    AemetError,
    AemetRateLimitError,
    AemetSource,
    ArchiveCoverage,
)
from xr_toolz.data._src.base import DatasetInfo, DatasetKind, DataSource
from xr_toolz.data._src.catalog import CATALOG, CatalogEntry, all_entries, describe
from xr_toolz.data._src.cds import (
    INSITU,
    REANALYSIS,
    CDSFormProfile,
    CDSInsituArchive,
    CDSSource,
)
from xr_toolz.data._src.cmems import CMEMSSource
from xr_toolz.data._src.credentials import (
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
