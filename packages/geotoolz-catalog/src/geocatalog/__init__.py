"""`geocatalog` — spatiotemporal index over geospatial files.

Surface (every public name is available flat at the top level):

- **Catalogs** — `GeoCatalog` (Protocol), `InMemoryGeoCatalog`,
  `DuckDBGeoCatalog` (``[duckdb]`` extra), `open_catalog` factory,
  `CatalogRow`, and the `query` / `intersect` / `union` set algebra.
- **Types** — `GeoSlice` (bbox + interval + CRS + resolution; catalogs
  produce them, loaders consume them), `slice_to_window` /
  `window_to_slice`, and the grid-alignment helpers (`Align`,
  `divide_evenly`, `is_grid_aligned`, `GridAlignmentWarning`).
- **Builders / loaders** — `build_raster_catalog` /
  `build_xarray_catalog` / `build_vector_catalog` and `load_raster` /
  `load_raster_timeseries` / `load_xarray` / `load_vector` (the
  xarray / vector pairs are extras-gated).
- **Persistence** — `to_geoparquet` / `from_geoparquet`,
  `migrate_geoparquet`, `append_files`, `StreamingParquetWriter`,
  `sort_geoparquet`, `SCHEMA_VERSION_CURRENT`, and the
  `GeoCatalogError` hierarchy (`CatalogMetadataError`,
  `CatalogSchemaError`, `CatalogClosedError`).
- **I/O helpers** — `parse_uri` / `ParsedURI`, `retry_transient_io` and
  the UTC time helpers (`to_utc_ts`, `to_naive_utc`, `to_rfc3339`,
  `is_time_invariant`, `TIME_INVARIANT_START` / `TIME_INVARIANT_END`).
- **Discovery** — the `Source` Protocol, `SourceRow` carrier,
  `AuthStatus`, and the adapters `STACSource` / `CMRSource` /
  `EarthAccessSource` / `GEESource` (extras-gated), plus the STAC
  conversion trio `from_stac_items` / `from_stac_search` /
  `to_stac_collection`.
- **Matchup** — the `matchup` engine, `MatchupRow`, and the spatial
  (`Intersects`, `IouAtLeast`, `CentroidWithin`, `Contains`) and
  temporal (`NearestInTime`, `WithinWindow`, `Synchronous`)
  strategies.
- **Bundle / staging** — `CatalogBundle` / `QueryRecord` provenance
  persistence, `stage` / `LocalCache` remote-asset staging, and the
  `field_for` geopatcher bridge.
- **Domain bridge** — `CatalogDomain`, so a downstream
  `SpatialPatcher` (geopatcher) can iterate a catalog's rows.

Every public name is at the top level, and each is also exported by
exactly one thematic sub-namespace — `geocatalog.catalog` (catalogs,
builders, loaders, persistence), `geocatalog.types`, `geocatalog.sources`,
`geocatalog.matchup`, `geocatalog.bundle`, `geocatalog.staging` and
`geocatalog.io` — as the same object. Extras-gated names resolve
lazily, so importing `geocatalog` (or ``from geocatalog import *``)
never requires an optional dependency; using one whose extra is missing
raises its install hint.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from loguru import logger as _logger

from geocatalog._src._align import (
    Align,
    GridAlignmentWarning,
    divide_evenly,
    is_grid_aligned,
)
from geocatalog._src._lazy import lazy_getattr
from geocatalog._src._timeutil import (
    TIME_INVARIANT_END,
    TIME_INVARIANT_START,
    is_time_invariant,
    to_naive_utc,
    to_rfc3339,
    to_utc_ts,
)
from geocatalog._src.base import (
    CatalogClosedError,
    CatalogMetadataError,
    CatalogRow,
    CatalogSchemaError,
    GeoCatalog,
    GeoCatalogError,
)
from geocatalog._src.bundle import CatalogBundle, QueryRecord, source_row_to_gdf_row
from geocatalog._src.domain import CatalogDomain
from geocatalog._src.factory import open_catalog
from geocatalog._src.geoslice import (
    PIXEL_PRECISION,
    GeoSlice,
    slice_to_window,
    window_to_slice,
)
from geocatalog._src.matchup import (
    CentroidWithin,
    Contains,
    Intersects,
    IouAtLeast,
    MatchupRow,
    NearestInTime,
    SpatialStrategy,
    Synchronous,
    TemporalStrategy,
    WithinWindow,
    matchup,
)
from geocatalog._src.memory import InMemoryGeoCatalog
from geocatalog._src.ops import intersect, query, union
from geocatalog._src.parquet import (
    SCHEMA_VERSION_CURRENT,
    from_geoparquet,
    migrate_geoparquet,
    to_geoparquet,
)
from geocatalog._src.raster import (
    aload_raster,
    build_raster_catalog,
    load_raster,
    load_raster_timeseries,
)
from geocatalog._src.retry import retry_transient_io
from geocatalog._src.sources import AuthStatus, Source, SourceRow
from geocatalog._src.staging import LocalCache, field_for, stage
from geocatalog._src.streaming import (
    StreamingParquetWriter,
    append_files,
    sort_geoparquet,
)
from geocatalog._src.uri import ParsedURI, parse_uri


# Library hygiene: loguru's recommended pattern is to disable the
# library's own logger at import time so consumers don't see output by
# default. Opt in from a consumer app with `logger.enable("geocatalog")`.
_logger.disable("geocatalog")


if TYPE_CHECKING:
    from geocatalog._src.duckdb_backend import DuckDBGeoCatalog
    from geocatalog._src.sources.cmr import CMRSource
    from geocatalog._src.sources.earthaccess import EarthAccessSource
    from geocatalog._src.sources.gee import GEESource
    from geocatalog._src.sources.stac import STACSource
    from geocatalog._src.stac import (
        from_stac_items,
        from_stac_search,
        to_stac_collection,
    )
    from geocatalog._src.vector import build_vector_catalog, load_vector
    from geocatalog._src.xarray_backend import (
        build_xarray_catalog,
        load_xarray,
    )


__version__ = "0.2.3"  # x-release-please-version

__all__ = [
    "PIXEL_PRECISION",
    "SCHEMA_VERSION_CURRENT",
    "TIME_INVARIANT_END",
    "TIME_INVARIANT_START",
    "Align",
    "AuthStatus",
    "CMRSource",
    "CatalogBundle",
    "CatalogClosedError",
    "CatalogDomain",
    "CatalogMetadataError",
    "CatalogRow",
    "CatalogSchemaError",
    "CentroidWithin",
    "Contains",
    "DuckDBGeoCatalog",
    "EarthAccessSource",
    "GEESource",
    "GeoCatalog",
    "GeoCatalogError",
    "GeoSlice",
    "GridAlignmentWarning",
    "InMemoryGeoCatalog",
    "Intersects",
    "IouAtLeast",
    "LocalCache",
    "MatchupRow",
    "NearestInTime",
    "ParsedURI",
    "QueryRecord",
    "STACSource",
    "Source",
    "SourceRow",
    "SpatialStrategy",
    "StreamingParquetWriter",
    "Synchronous",
    "TemporalStrategy",
    "WithinWindow",
    "aload_raster",
    "append_files",
    "build_raster_catalog",
    "build_vector_catalog",
    "build_xarray_catalog",
    "divide_evenly",
    "field_for",
    "from_geoparquet",
    "from_stac_items",
    "from_stac_search",
    "intersect",
    "is_grid_aligned",
    "is_time_invariant",
    "load_raster",
    "load_raster_timeseries",
    "load_vector",
    "load_xarray",
    "matchup",
    "migrate_geoparquet",
    "open_catalog",
    "parse_uri",
    "query",
    "retry_transient_io",
    "slice_to_window",
    "sort_geoparquet",
    "source_row_to_gdf_row",
    "stage",
    "to_geoparquet",
    "to_naive_utc",
    "to_rfc3339",
    "to_stac_collection",
    "to_utc_ts",
    "union",
    "window_to_slice",
]


# Load the sub-namespaces now and bind them explicitly, so attribute
# access never depends on import order (or on `importlib.reload`, which
# re-runs the `matchup` function import above while the cached
# sub-modules are not re-imported). `geocatalog.matchup` is both a
# function and a sub-namespace: the module is callable (it forwards to
# the `matchup` function), so `geocatalog.matchup(...)`,
# `geocatalog.matchup.MatchupRow` and `import geocatalog.matchup as m`
# all work.
for _namespace in ("bundle", "catalog", "io", "matchup", "sources", "staging", "types"):
    globals()[_namespace] = importlib.import_module(f"geocatalog.{_namespace}")
del _namespace


# Extras-gated names resolve lazily (`geocatalog._src._lazy.LAZY`).
__getattr__ = lazy_getattr(
    globals(),
    [
        "CMRSource",
        "DuckDBGeoCatalog",
        "EarthAccessSource",
        "GEESource",
        "STACSource",
        "build_vector_catalog",
        "build_xarray_catalog",
        "from_stac_items",
        "from_stac_search",
        "load_vector",
        "load_xarray",
        "to_stac_collection",
    ],
)
