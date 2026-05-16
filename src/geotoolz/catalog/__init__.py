"""`geotoolz.catalog` — spatiotemporal index over geospatial files.

Phase 1 + Phase 2 of the geodatabase design — see
``research_journal_v2/notes/geotoolz/plans/geodatabase/``.

Surface:

- `GeoCatalog`: Protocol; one shape across backends.
- `CatalogRow`: backend-neutral row view yielded by `iter_rows`.
- `InMemoryGeoCatalog`: GeoDataFrame-backed Phase 1 implementation.
- `DuckDBGeoCatalog`: SQL-backed Phase 2 implementation (extras-gated
  via `[duckdb]`).
- `open_catalog`: factory that picks a backend for a GeoParquet artifact.
- `build_raster_catalog` / `build_xarray_catalog` / `build_vector_catalog`:
  builders, the latter two extras-gated.
- `load_raster` / `load_raster_timeseries` / `load_xarray` / `load_vector`:
  per-backend loaders that consume a `GeoSlice` and return `GeoTensor`
  (raster, vector) or `xr.Dataset` (xarray).
- `query` / `intersect` / `union`: set algebra over catalogs.
- `to_geoparquet` / `from_geoparquet`: portable artifact round-trip.
- `CatalogDomain`: adapter so a `geopatcher.SpatialPatcher` can
  iterate a catalog's rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from geotoolz.catalog._src.base import CatalogRow, GeoCatalog
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
    from geotoolz.catalog._src.duckdb_backend import DuckDBGeoCatalog
    from geotoolz.catalog._src.vector import build_vector_catalog, load_vector
    from geotoolz.catalog._src.xarray_backend import (
        build_xarray_catalog,
        load_xarray,
    )


__all__ = [
    "CatalogDomain",
    "CatalogRow",
    "DuckDBGeoCatalog",
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
    "open_catalog",
    "query",
    "to_geoparquet",
    "union",
]


_BACKEND_T = Literal["raster", "xarray", "vector"]


def open_catalog(
    source: str | Path,
    *,
    backend: _BACKEND_T | None = None,
    engine: Literal["auto", "memory", "duckdb"] = "auto",
    crs: Any | None = None,
) -> GeoCatalog:
    """Open a GeoParquet artifact as a `GeoCatalog`.

    The factory picks an engine. ``"auto"`` prefers the DuckDB backend
    (lazy, scales) when the ``[duckdb]`` extra is installed; otherwise it
    falls back to the in-memory backend via `from_geoparquet`. Pass
    ``engine="memory"`` to force the eager path even with DuckDB present.

    The artifact's stored backend tag (the ``_backend`` column written
    by `to_geoparquet`) is honoured by default — pass ``backend=...``
    only to override a wrong tag or to tag an externally produced
    artifact that lacks the column. Forcing a default would silently
    miscategorise xarray / vector catalogs and break loader dispatch.

    Args:
        source: Path or URI to a GeoParquet file. A directory of shards
            (``shards/``) or a glob (``shards/*.parquet``) is read as one
            virtual table by the DuckDB engine; the in-memory engine
            requires a single file.
        backend: Loader dispatch tag (``"raster"`` / ``"xarray"`` /
            ``"vector"``). ``None`` reads the ``_backend`` column from
            the artifact (default ``"raster"`` if missing).
        engine: ``"auto"`` (DuckDB if available, else memory),
            ``"memory"`` (force `from_geoparquet`), or ``"duckdb"``
            (force `DuckDBGeoCatalog.open`; raises if the extra is not
            installed).
        crs: Optional CRS override; only consulted by the DuckDB engine
            when the artifact doesn't carry one.

    Returns:
        A `GeoCatalog` over ``source``. The concrete class is either
        `DuckDBGeoCatalog` or `InMemoryGeoCatalog` depending on the
        resolved engine.

    Raises:
        ImportError: ``engine="duckdb"`` with the extra missing.
    """
    if engine == "memory":
        cat = from_geoparquet(source)
        if backend is not None:
            cat.backend = backend
        return cat
    if engine == "duckdb":
        from geotoolz.catalog._src.duckdb_backend import DuckDBGeoCatalog

        return DuckDBGeoCatalog.open(source, backend=backend, crs=crs)
    # engine == "auto"
    try:
        from geotoolz.catalog._src.duckdb_backend import DuckDBGeoCatalog
    except ImportError:
        cat = from_geoparquet(source)
        if backend is not None:
            cat.backend = backend
        return cat
    return DuckDBGeoCatalog.open(source, backend=backend, crs=crs)


_LAZY_ATTRS = {
    "build_xarray_catalog": ("geotoolz.catalog._src.xarray_backend",),
    "load_xarray": ("geotoolz.catalog._src.xarray_backend",),
    "build_vector_catalog": ("geotoolz.catalog._src.vector",),
    "load_vector": ("geotoolz.catalog._src.vector",),
    "DuckDBGeoCatalog": ("geotoolz.catalog._src.duckdb_backend",),
}


def __getattr__(name: str) -> Any:
    """Lazy import for the extras-gated backends.

    Importing `geotoolz.catalog` at top level should not fail just
    because an optional dep (`xarray`, `duckdb`) is missing. Resolving
    e.g. `load_xarray` or `DuckDBGeoCatalog` finally triggers the import
    and raises a friendly `ImportError` if the corresponding extra
    isn't installed.
    """
    if name in _LAZY_ATTRS:
        import importlib

        module_name = _LAZY_ATTRS[name][0]
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            if "xarray" in module_name:
                extra = "xarray-raster"
            elif "duckdb" in module_name:
                extra = "duckdb"
            else:
                extra = "vector"
            raise ImportError(
                f"`geotoolz.catalog.{name}` requires the [{extra}] extra; "
                f"install via `pip install 'geotoolz[{extra}]'`."
            ) from exc
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'geotoolz.catalog' has no attribute {name!r}")
