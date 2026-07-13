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
  `migrate_geoparquet`, `append_files`, `SCHEMA_VERSION_CURRENT`.
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

The hybrid layout exposes the same surface at three paths: flat top
level (`geocatalog.GeoSlice`), thematic sub-namespaces
(`geocatalog.types.GeoSlice`, `geocatalog.sources.STACSource`,
`geocatalog.matchup.matchup`, ...), and `geocatalog.catalog.*`. Pick
whichever reads best where you import. Extras-gated names resolve
lazily — importing `geocatalog` never requires an optional dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger as _logger

from geocatalog._src._align import (
    Align,
    GridAlignmentWarning,
    divide_evenly,
    is_grid_aligned,
)
from geocatalog._src.base import (
    CatalogMetadataError,
    CatalogRow,
    CatalogSchemaError,
    GeoCatalog,
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
from geocatalog._src.sources import AuthStatus, Source, SourceRow
from geocatalog._src.staging import LocalCache, field_for, stage
from geocatalog._src.streaming import append_files


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


__version__ = "0.0.3"

__all__ = [
    "PIXEL_PRECISION",
    "SCHEMA_VERSION_CURRENT",
    "Align",
    "AuthStatus",
    "CMRSource",
    "CatalogBundle",
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
    "GeoSlice",
    "GridAlignmentWarning",
    "InMemoryGeoCatalog",
    "Intersects",
    "IouAtLeast",
    "LocalCache",
    "MatchupRow",
    "NearestInTime",
    "QueryRecord",
    "STACSource",
    "Source",
    "SourceRow",
    "SpatialStrategy",
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
    "load_raster",
    "load_raster_timeseries",
    "load_vector",
    "load_xarray",
    "matchup",
    "migrate_geoparquet",
    "open_catalog",
    "query",
    "slice_to_window",
    "source_row_to_gdf_row",
    "stage",
    "to_geoparquet",
    "to_stac_collection",
    "union",
    "window_to_slice",
]


# Extras-gated names resolve lazily: mapping of public name to
# (module, extra) — `extra` is the install hint used in the error
# message, or None when the module has no dedicated extra.
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    "build_xarray_catalog": ("geocatalog._src.xarray_backend", "xarray-raster"),
    "load_xarray": ("geocatalog._src.xarray_backend", "xarray-raster"),
    "build_vector_catalog": ("geocatalog._src.vector", None),
    "load_vector": ("geocatalog._src.vector", None),
    "DuckDBGeoCatalog": ("geocatalog._src.duckdb_backend", "duckdb"),
    "from_stac_items": ("geocatalog._src.stac", "stac"),
    "from_stac_search": ("geocatalog._src.stac", "stac"),
    "to_stac_collection": ("geocatalog._src.stac", "stac"),
    "CMRSource": ("geocatalog._src.sources.cmr", None),
    "EarthAccessSource": ("geocatalog._src.sources.earthaccess", "earthaccess"),
    "STACSource": ("geocatalog._src.sources.stac", "stac"),
    "GEESource": ("geocatalog._src.sources.gee", "gee"),
}


def __getattr__(name: str) -> Any:
    """Lazy import for the extras-gated backends and source adapters.

    Importing `geocatalog` at top level should not fail just because an
    optional dep (`xarray`, `duckdb`, `pystac-client`, ...) is missing.
    Resolving e.g. `load_xarray` or `STACSource` finally triggers the
    import and raises a friendly `ImportError` naming the extra to
    install if the corresponding dependency isn't there.
    """
    if name in _LAZY_ATTRS:
        import importlib

        module_name, extra = _LAZY_ATTRS[name]
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            if extra is None:
                raise
            raise ImportError(
                f"`geocatalog.{name}` requires the [{extra}] extra; "
                f"install via `pip install 'geocatalog[{extra}]'`."
            ) from exc
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'geocatalog' has no attribute {name!r}")
