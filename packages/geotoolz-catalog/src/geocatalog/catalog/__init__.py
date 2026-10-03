"""`geocatalog.catalog` — catalogs, builders, loaders and persistence.

Thematic sub-namespace: the catalog backends and their set algebra, the
builders and loaders, GeoParquet persistence (including the streaming
writer and the schema-version errors) and the patcher domain bridge.
Every name here is also at the top level, as the same object —
``from geocatalog import InMemoryGeoCatalog`` and
``from geocatalog.catalog import InMemoryGeoCatalog`` are equivalent.
`GeoSlice` and the grid helpers live in `geocatalog.types`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from geocatalog._src._lazy import lazy_getattr
from geocatalog._src.base import (
    CatalogMetadataError,
    CatalogRow,
    CatalogSchemaError,
    GeoCatalog,
)
from geocatalog._src.domain import CatalogDomain
from geocatalog._src.factory import open_catalog
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
from geocatalog._src.streaming import (
    StreamingParquetWriter,
    append_files,
    sort_geoparquet,
)


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
    "SCHEMA_VERSION_CURRENT",
    "CatalogDomain",
    "CatalogMetadataError",
    "CatalogRow",
    "CatalogSchemaError",
    "DuckDBGeoCatalog",
    "GeoCatalog",
    "InMemoryGeoCatalog",
    "StreamingParquetWriter",
    "aload_raster",
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
    "migrate_geoparquet",
    "open_catalog",
    "query",
    "sort_geoparquet",
    "to_geoparquet",
    "to_stac_collection",
    "union",
]


__getattr__ = lazy_getattr(
    globals(),
    [
        "DuckDBGeoCatalog",
        "build_vector_catalog",
        "build_xarray_catalog",
        "from_stac_items",
        "from_stac_search",
        "load_vector",
        "load_xarray",
        "to_stac_collection",
    ],
)
