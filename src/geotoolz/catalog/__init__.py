"""`geotoolz.catalog` — spatiotemporal index over geospatial files.

Phase 1 of the geodatabase design — see
``research_journal_v2/notes/geotoolz/plans/geodatabase/geocatalog.md``.

Surface:

- `GeoCatalog`: Protocol; one shape across backends.
- `InMemoryGeoCatalog`: GeoDataFrame-backed Phase 1 implementation.
- `build_raster_catalog` / `build_xarray_catalog` / `build_vector_catalog`:
  builders, the latter two extras-gated.
- `load_raster` / `load_raster_timeseries` / `load_xarray` / `load_vector`:
  per-backend loaders that consume a `GeoSlice` and return `GeoTensor`
  (raster, vector) or `xr.Dataset` (xarray).
- `query` / `intersect` / `union`: set algebra over catalogs.
- `to_geoparquet` / `from_geoparquet`: portable artifact round-trip.
- `CatalogDomain`: adapter so a `geotoolz.patch.SpatialPatcher` can
  iterate a catalog's rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geotoolz.catalog._src.base import GeoCatalog
from geotoolz.catalog._src.domain import CatalogDomain
from geotoolz.catalog._src.memory import InMemoryGeoCatalog
from geotoolz.catalog._src.ops import intersect, query, union
from geotoolz.catalog._src.parquet import from_geoparquet, to_geoparquet
from geotoolz.catalog._src.raster import (
    build_raster_catalog,
    load_raster,
    load_raster_timeseries,
)


if TYPE_CHECKING:
    from geotoolz.catalog._src.vector import build_vector_catalog, load_vector
    from geotoolz.catalog._src.xarray_backend import (
        build_xarray_catalog,
        load_xarray,
    )


__all__ = [
    "CatalogDomain",
    "GeoCatalog",
    "InMemoryGeoCatalog",
    "build_raster_catalog",
    "build_vector_catalog",
    "build_xarray_catalog",
    "from_geoparquet",
    "intersect",
    "load_raster",
    "load_raster_timeseries",
    "load_vector",
    "load_xarray",
    "query",
    "to_geoparquet",
    "union",
]


_LAZY_ATTRS = {
    "build_xarray_catalog": ("geotoolz.catalog._src.xarray_backend",),
    "load_xarray": ("geotoolz.catalog._src.xarray_backend",),
    "build_vector_catalog": ("geotoolz.catalog._src.vector",),
    "load_vector": ("geotoolz.catalog._src.vector",),
}


def __getattr__(name: str) -> Any:
    """Lazy import for the extras-gated backends.

    Importing `geotoolz.catalog` at top level should not fail just because
    `xarray` or `geopandas` is missing. Resolving e.g. `load_xarray`
    finally triggers the import and raises a friendly `ImportError` if
    the corresponding extra isn't installed.
    """
    if name in _LAZY_ATTRS:
        import importlib

        module_name = _LAZY_ATTRS[name][0]
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            extra = "xarray-raster" if "xarray" in module_name else "vector"
            raise ImportError(
                f"`geotoolz.catalog.{name}` requires the [{extra}] extra; "
                f"install via `pip install 'geotoolz[{extra}]'`."
            ) from exc
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'geotoolz.catalog' has no attribute {name!r}")
