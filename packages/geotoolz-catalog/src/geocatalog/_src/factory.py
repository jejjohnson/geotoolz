"""Backend-agnostic catalog factory.

`open_catalog` is the one entry point that picks a concrete
`GeoCatalog` implementation for a GeoParquet artifact — the DuckDB
backend when the ``[duckdb]`` extra is installed (lazy, scales to
sharded artifacts), else the in-memory GeoDataFrame backend. It lives
in ``_src`` so the package ``__init__`` stays a pure re-export facade.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal, get_args

from geocatalog._src._deprecation import renamed_kwargs
from geocatalog._src._schema import CatalogKind
from geocatalog._src.base import GeoCatalog
from geocatalog._src.memory import InMemoryGeoCatalog
from geocatalog._src.parquet import from_geoparquet


_ENGINE_T = Literal["auto", "memory", "duckdb"]


@renamed_kwargs(backend="kind")
def open_catalog(
    source: str | Path,
    *,
    kind: CatalogKind | None = None,
    engine: _ENGINE_T = "auto",
    crs: Any | None = None,
    storage_options: dict[str, Any] | None = None,
    strict: bool = False,
) -> GeoCatalog:
    """Open a GeoParquet artifact as a `GeoCatalog`.

    The factory picks an engine. ``"auto"`` prefers the DuckDB backend
    (lazy, scales) when the ``[duckdb]`` extra is installed; otherwise it
    falls back to the in-memory backend via `from_geoparquet`. Pass
    ``engine="memory"`` to force the eager path even with DuckDB present.

    The artifact's stored kind (the ``_backend`` column written
    by `to_geoparquet`) is honoured by default — pass ``kind=...``
    only to override a wrong tag or to tag an externally produced
    artifact that lacks the column. Forcing a default would silently
    miscategorise xarray / vector catalogs and break loader dispatch.

    Args:
        source: Path or URI to a GeoParquet file. A directory of shards
            (``shards/``) or a glob (``shards/*.parquet``) is read as one
            virtual table by the DuckDB engine; the in-memory engine
            requires a single file.
        kind: Loader dispatch tag (``"raster"`` / ``"xarray"`` /
            ``"vector"``). ``None`` reads the ``_backend`` column from
            the artifact (default ``"raster"`` if missing).
        engine: ``"auto"`` (DuckDB if available, else memory),
            ``"memory"`` (force `from_geoparquet`), or ``"duckdb"``
            (force `DuckDBGeoCatalog.open`; raises if the extra is not
            installed).
        crs: Optional CRS override for either engine. It relabels the
            footprints (no reprojection) — use it for artifacts whose
            GeoParquet metadata lacks or misstates the CRS.
        storage_options: Options forwarded to fsspec when reading cloud
            URIs through the in-memory engine. A non-empty mapping makes
            ``"auto"`` pick that engine; ``{}`` counts as no options.
        strict: If ``True``, raise `CatalogMetadataError` instead of
            warning-and-falling-back when the artifact is missing the
            ``_backend`` column (and ``kind=`` was not passed) or —
            on the DuckDB engine — its ``geo`` metadata is unreadable
            (and ``crs=`` was not passed).

    Returns:
        A `GeoCatalog` over ``source``. The concrete class is either
        `DuckDBGeoCatalog` or `InMemoryGeoCatalog` depending on the
        resolved engine.

    ``"auto"`` also falls back to the in-memory engine, with a warning,
    when DuckDB is installed but its ``spatial`` extension cannot be
    loaded (e.g. offline), since every spatial query would fail.

    Raises:
        ImportError: ``engine="duckdb"`` with the extra missing.
        ValueError: ``engine`` is not one of ``"auto"``, ``"memory"``,
            ``"duckdb"``.
    """
    if engine not in get_args(_ENGINE_T):
        raise ValueError(
            f"open_catalog engine must be one of {list(get_args(_ENGINE_T))}; "
            f"got {engine!r}."
        )
    # `{}` carries no options; don't let it switch engines (#225).
    storage_options = storage_options or None
    if engine == "memory":
        return _memory_engine(
            source, kind, crs=crs, storage_options=storage_options, strict=strict
        )
    if engine == "duckdb":
        from geocatalog._src.duckdb_backend import DuckDBGeoCatalog

        return DuckDBGeoCatalog.open(
            source,
            kind=kind,
            crs=crs,
            storage_options=storage_options,
            strict=strict,
        )
    # engine == "auto"
    if storage_options is not None:
        return _memory_engine(
            source, kind, crs=crs, storage_options=storage_options, strict=strict
        )
    try:
        from geocatalog._src.duckdb_backend import DuckDBGeoCatalog
    except ImportError:
        return _memory_engine(source, kind, crs=crs, strict=strict)
    try:
        catalog = DuckDBGeoCatalog.open(
            source,
            kind=kind,
            crs=crs,
            storage_options=storage_options,
            strict=strict,
        )
    except ImportError:
        return _memory_engine(source, kind, crs=crs, strict=strict)
    if not catalog._spatial_available:
        catalog.close()
        warnings.warn(
            f"open_catalog({str(source)!r}): DuckDB's spatial extension is "
            "unavailable, so spatial queries would fail; using the in-memory "
            "engine instead. Pass engine='duckdb' to keep DuckDB anyway.",
            UserWarning,
            stacklevel=2,
        )
        return _memory_engine(source, kind, crs=crs, strict=strict)
    return catalog


def _memory_engine(
    source: str | Path,
    kind: CatalogKind | None,
    *,
    crs: Any | None = None,
    storage_options: dict[str, Any] | None = None,
    strict: bool = False,
) -> InMemoryGeoCatalog:
    """Open ``source`` as an `InMemoryGeoCatalog`, applying overrides.

    The kind override is forwarded into `from_geoparquet` so an
    explicit ``kind=`` skips tag recovery entirely — no missing-column
    warning, no strict-mode raise. A ``crs=`` override relabels the
    footprints, as the DuckDB engine does (it was silently ignored here).
    """
    catalog = from_geoparquet(
        source, kind=kind, strict=strict, storage_options=storage_options
    )
    if crs is None:
        return catalog
    gdf = catalog.gdf.set_crs(crs, allow_override=True)
    return InMemoryGeoCatalog(gdf, kind=catalog.kind)
