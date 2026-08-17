"""xrreader — authenticated archive readers for xarray geodata.

``xrreader`` turns a typed :class:`Request` into a service-specific payload
and downloads or opens an :class:`xarray.Dataset` from external geoscience
archives (CMEMS, CDS, AEMET). The typed request vocabulary
(:class:`BBox`, :class:`TimeRange`, :class:`Variable`, ...) lives in
:mod:`xrreader.types`; this top-level namespace re-exports the reader API
and the common request types flat.

Reader API:

- :class:`DataSource`, :class:`DatasetInfo`, :class:`DatasetKind`
- Adapters: :class:`CMEMSSource`, :class:`CDSSource`, :class:`AemetSource`
- Credentials: :class:`CMEMSCredentials`, :class:`CDSCredentials`,
  :class:`AEMETCredentials`, :func:`load_cmems`, :func:`load_cds`,
  :func:`load_aemet`.
- Catalog: :data:`CATALOG`, :class:`CatalogEntry`, :func:`all_entries`,
  :func:`describe`.
- AEMET extras: :class:`AemetArchive`, :class:`AemetError`,
  :class:`AemetAuthError`, :class:`AemetRateLimitError`,
  :class:`ArchiveCoverage`.
- CDS extras: :class:`CDSInsituArchive`, :class:`CDSFormProfile`,
  :data:`INSITU`, :data:`REANALYSIS`.

Request vocabulary (also available via :mod:`xrreader.types`):

- Spatial / temporal / vertical: :class:`BBox`, :class:`TimeRange`,
  :class:`DepthRange`, :class:`PressureLevels`.
- Composition: :class:`Request`.
- Stations: :class:`Station`, :class:`StationCollection`.
- Variables: :class:`Variable`, :data:`REGISTRY`, :func:`resolve`,
  :func:`register`.
- Validation: :func:`validate_variable`, :func:`validate_dataset`,
  :func:`apply_cf_attrs`, :class:`ValidationReport`, :class:`Issue`,
  :class:`Severity`.
"""

from xrreader._src.aemet import (
    AemetArchive,
    AemetAuthError,
    AemetError,
    AemetRateLimitError,
    AemetSource,
    ArchiveCoverage,
)
from xrreader._src.base import DatasetInfo, DatasetKind, DataSource
from xrreader._src.catalog import CATALOG, CatalogEntry, all_entries, describe
from xrreader._src.cds import (
    INSITU,
    REANALYSIS,
    CDSFormProfile,
    CDSInsituArchive,
    CDSSource,
)
from xrreader._src.cmems import CMEMSSource
from xrreader._src.credentials import (
    AEMETCredentials,
    CDSCredentials,
    CMEMSCredentials,
    load_aemet,
    load_cds,
    load_cmems,
)
from xrreader.types import (
    REGISTRY,
    BBox,
    DepthRange,
    Issue,
    PressureLevels,
    Request,
    Severity,
    Station,
    StationCollection,
    TimeRange,
    ValidationReport,
    Variable,
    apply_cf_attrs,
    register,
    resolve,
    validate_dataset,
    validate_variable,
)


__version__ = "0.0.3"  # x-release-please-version

__all__ = [
    "CATALOG",
    "INSITU",
    "REANALYSIS",
    "REGISTRY",
    "AEMETCredentials",
    "AemetArchive",
    "AemetAuthError",
    "AemetError",
    "AemetRateLimitError",
    "AemetSource",
    "ArchiveCoverage",
    "BBox",
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
    "DepthRange",
    "Issue",
    "PressureLevels",
    "Request",
    "Severity",
    "Station",
    "StationCollection",
    "TimeRange",
    "ValidationReport",
    "Variable",
    "__version__",
    "all_entries",
    "apply_cf_attrs",
    "describe",
    "load_aemet",
    "load_cds",
    "load_cmems",
    "register",
    "resolve",
    "validate_dataset",
    "validate_variable",
]
