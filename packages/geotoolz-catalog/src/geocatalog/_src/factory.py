"""Backend-agnostic catalog factory.

`open_catalog` is the one entry point that picks a concrete
`GeoCatalog` implementation for a GeoParquet artifact — the DuckDB
backend when the ``[duckdb]`` extra is installed (lazy, scales to
sharded artifacts), else the in-memory GeoDataFrame backend. It lives
in ``_src`` so the package ``__init__`` stays a pure re-export facade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from geocatalog._src.base import GeoCatalog
from geocatalog._src.memory import InMemoryGeoCatalog
from geocatalog._src.parquet import from_geoparquet


_BACKEND_T = Literal["raster", "xarray", "vector"]


def open_catalog(
    source: str | Path,
    *,
    backend: _BACKEND_T | None = None,
    engine: Literal["auto", "memory", "duckdb"] = "auto",
    crs: Any | None = None,
    storage_options: dict[str, Any] | None = None,
    strict: bool = False,
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
        storage_options: Options forwarded to fsspec when reading cloud
            URIs through the in-memory engine.
        strict: If ``True``, raise `CatalogMetadataError` instead of
            warning-and-falling-back when the artifact is missing the
            ``_backend`` column (and ``backend=`` was not passed) or —
            on the DuckDB engine — its ``geo`` metadata is unreadable
            (and ``crs=`` was not passed).

    Returns:
        A `GeoCatalog` over ``source``. The concrete class is either
        `DuckDBGeoCatalog` or `InMemoryGeoCatalog` depending on the
        resolved engine.

    Raises:
        ImportError: ``engine="duckdb"`` with the extra missing.
    """
    if engine == "memory":
        return _memory_engine(
            source, backend, storage_options=storage_options, strict=strict
        )
    if engine == "duckdb":
        from geocatalog._src.duckdb_backend import DuckDBGeoCatalog

        return DuckDBGeoCatalog.open(
            source,
            backend=backend,
            crs=crs,
            storage_options=storage_options,
            strict=strict,
        )
    # engine == "auto"
    if storage_options is not None:
        return _memory_engine(
            source, backend, storage_options=storage_options, strict=strict
        )
    try:
        from geocatalog._src.duckdb_backend import DuckDBGeoCatalog
    except ImportError:
        return _memory_engine(
            source, backend, storage_options=storage_options, strict=strict
        )
    try:
        return DuckDBGeoCatalog.open(
            source,
            backend=backend,
            crs=crs,
            storage_options=storage_options,
            strict=strict,
        )
    except ImportError:
        return _memory_engine(
            source, backend, storage_options=storage_options, strict=strict
        )


def _memory_engine(
    source: str | Path,
    backend: _BACKEND_T | None,
    *,
    storage_options: dict[str, Any] | None = None,
    strict: bool = False,
) -> InMemoryGeoCatalog:
    """Open ``source`` as an `InMemoryGeoCatalog`, applying a backend override.

    The override is forwarded into `from_geoparquet` so an explicit
    ``backend=`` skips tag recovery entirely — no missing-column
    warning, no strict-mode raise.
    """
    return from_geoparquet(
        source, backend=backend, strict=strict, storage_options=storage_options
    )
