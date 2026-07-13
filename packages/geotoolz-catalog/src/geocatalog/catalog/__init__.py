"""`geocatalog.catalog` — backends, builders, loaders, set algebra.

Hybrid-layout sub-namespace. Re-exports everything in the flat
top-level surface except `GeoSlice` (which lives in `geocatalog.types`).
Both ``from geocatalog import InMemoryGeoCatalog`` and
``from geocatalog.catalog import InMemoryGeoCatalog`` work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geocatalog import open_catalog
from geocatalog._src.base import CatalogRow, GeoCatalog
from geocatalog._src.domain import CatalogDomain
from geocatalog._src.memory import InMemoryGeoCatalog
from geocatalog._src.ops import intersect, query, union
from geocatalog._src.parquet import from_geoparquet, to_geoparquet
from geocatalog._src.raster import (
    build_raster_catalog,
    load_raster,
    load_raster_timeseries,
)
from geocatalog._src.streaming import append_files


if TYPE_CHECKING:
    from geocatalog._src.duckdb_backend import DuckDBGeoCatalog
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


__all__ = [
    "CatalogDomain",
    "CatalogRow",
    "DuckDBGeoCatalog",
    "GeoCatalog",
    "InMemoryGeoCatalog",
    "append_files",
    "build_raster_catalog",
    "build_vector_catalog",
    "build_xarray_catalog",
    "from_geoparquet",
    "from_stac_items",
    "from_stac_search",
    "intersect",
    "load_raster",
    "load_raster_timeseries",
    "load_vector",
    "load_xarray",
    "open_catalog",
    "query",
    "to_geoparquet",
    "to_stac_collection",
    "union",
]


def __getattr__(name: str) -> Any:
    """Defer to the top-level lazy loader for extras-gated backends."""
    import geocatalog as _gc

    return getattr(_gc, name)
