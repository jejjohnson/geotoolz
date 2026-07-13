"""`to_geoparquet` / `from_geoparquet` — catalog ↔ portable artifact.

Round-trips an `InMemoryGeoCatalog` through GeoParquet 1.1 (geopandas
writes the bbox covering struct when ``write_covering_bbox=True``). The
artifact is then queryable from pandas / DuckDB / GDAL without
ceremony — no pickle-version fragility.

Phase 2 (DuckDB backend) reads the *same* GeoParquet file, so
``to_geoparquet`` writes the canonical interchange format for both
backends.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger as log

from geocatalog._src.base import CatalogMetadataError, CatalogSchemaError
from geocatalog._src.io import _close_resolved_uri, _resolve_uri
from geocatalog._src.memory import InMemoryGeoCatalog
from geocatalog._src.retry import retry_transient_io


_BACKEND_T = Literal["raster", "xarray", "vector"]


# The reader's current schema version. Bump on every substantive schema
# change and add an entry to `_MIGRATIONS` for ``previous → this``.
SCHEMA_VERSION_CURRENT: int = 0


# Artifacts written before `_schema_version` existed as a reserved
# column are treated as v0 — the schema that was current when the
# column was introduced. Pinning to a constant (not
# `SCHEMA_VERSION_CURRENT`) means the next schema bump will still
# trigger a v0 -> v1 migration on those legacy files, instead of
# silently skipping it.
_LEGACY_UNVERSIONED: int = 0


# Forward migrations keyed by *source* version. `_MIGRATIONS[k]` takes a
# v_k gdf and returns a v_(k+1) gdf. The chain
# ``_MIGRATIONS[v_artifact] ∘ … ∘ _MIGRATIONS[v_current - 1]`` brings an
# old artifact up to the current version. Empty today (current schema
# is v0); populate when shipping v1.
#
# Input shape contract for migration authors: migrations run inside
# `from_geoparquet` AFTER the internal `_backend` / `_schema_version`
# columns are dropped and BEFORE the (start_time, end_time)
# IntervalIndex is rebuilt — a migration sees plain `start_time` /
# `end_time` columns, never the interval index.
_MIGRATIONS: dict[int, Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame]] = {}


def _read_schema_version(
    path: str | Path,
    *,
    storage_options: dict[str, Any] | None = None,
) -> int:
    """Return the `_schema_version` recorded in ``path``, without loading rows.

    Reads only the `_schema_version` column from the parquet file
    via pyarrow's selective-column reader — avoids loading the
    geometry / time columns just to inspect the version. Used by
    `migrate_geoparquet` to cheaply decide whether a rewrite is
    necessary, and importable for tests.

    Returns:
        - The unique value in the column when present and consistent
          across all rows.
        - `_LEGACY_UNVERSIONED` (0) when the column is absent (file
          predates the column's introduction).

    Raises:
        CatalogSchemaError: If the column exists but holds null/NaN,
            or holds different versions across rows (mixed-shard
            artifact — the reader can't decide which is canonical).
    """
    # `_schema_version` may not exist in pre-versioning artifacts.
    # pyarrow raises either ArrowInvalid (selective read) or KeyError
    # (column lookup) — catch both so the absence is transparently
    # treated as "legacy v0".
    import pyarrow

    try:
        resolved = _resolve_uri(path, storage_options=storage_options)
        try:
            table = pq.read_table(resolved, columns=["_schema_version"])
        finally:
            _close_resolved_uri(resolved)
        version_series = pd.Series(table.column("_schema_version").to_pandas())
    except (KeyError, pyarrow.lib.ArrowInvalid):
        return _LEGACY_UNVERSIONED
    if len(version_series) == 0:
        return _LEGACY_UNVERSIONED
    if version_series.isna().any():
        raise CatalogSchemaError(
            f"artifact {path} has null values in `_schema_version`; "
            "the column must be populated on every row."
        )
    unique = version_series.unique()
    if len(unique) > 1:
        raise CatalogSchemaError(
            f"artifact {path} has mixed `_schema_version` values "
            f"{sorted(map(int, unique))}; the reader can't open a "
            "multi-version source. Migrate each shard separately, "
            "or rewrite into one file at a single version."
        )
    return int(unique[0])


def _apply_migrations(gdf: gpd.GeoDataFrame, *, from_version: int) -> gpd.GeoDataFrame:
    """Chain forward migrations from `from_version` to `SCHEMA_VERSION_CURRENT`.

    Raises:
        CatalogSchemaError: If a migration is missing for any version in
            the chain — that's a library bug (someone bumped
            `SCHEMA_VERSION_CURRENT` without registering the migration).
    """
    for v in range(from_version, SCHEMA_VERSION_CURRENT):
        migration = _MIGRATIONS.get(v)
        if migration is None:
            raise CatalogSchemaError(
                f"missing migration v{v} -> v{v + 1}; this is a library bug — "
                "`SCHEMA_VERSION_CURRENT` was bumped without registering "
                "the corresponding entry in `_MIGRATIONS`."
            )
        gdf = migration(gdf)
    return gdf


def to_geoparquet(
    catalog: InMemoryGeoCatalog,
    path: str | Path,
    *,
    schema_version: int = SCHEMA_VERSION_CURRENT,
    write_covering_bbox: bool = True,
    partition_by: tuple[str, ...] | None = None,
) -> None:
    """Persist ``catalog`` as a GeoParquet file on disk.

    The result is a single Parquet file readable by any GeoParquet-aware
    tool (DuckDB, GDAL, pandas, geopandas) — not a pickle, so it
    survives version bumps and crosses Python / language boundaries.
    The Phase 2 ``DuckDBGeoCatalog`` reads the *same* artifact, so this
    is the canonical interchange format for both backends.

    Two columns are added on write and stripped on load:

    - ``_backend``: round-trips the backend tag so `from_geoparquet`
      restores the right loader dispatch.
    - ``_schema_version``: reserved for forward-compat (§10.4 of the
      design plan); bump on first substantive schema change.

    The row-level ``pd.IntervalIndex`` is unpacked into ``start_time`` /
    ``end_time`` columns (Parquet has no native IntervalIndex type);
    `from_geoparquet` rebuilds the index from those columns.

    Args:
        catalog: An `InMemoryGeoCatalog` to serialise. The catalog's
            ``gdf.crs`` is written into the GeoParquet metadata.
        path: Destination path. The ``.parquet`` extension is
            conventional. Any parent directory must exist.
        schema_version: Value written to the reserved
            ``_schema_version`` column. Defaults to
            `SCHEMA_VERSION_CURRENT` (today: 0).
        write_covering_bbox: Emit the per-row ``bbox`` covering struct
            that GeoParquet 1.1 readers (DuckDB, geopandas ≥0.14) use
            for predicate pushdown. Default True; turn off only if a
            downstream consumer chokes on 1.1.
        partition_by: Optional Hive partition columns for directory
            output. Built-in ``"year"``, ``"month"``, and ``"day"`` are
            derived from ``start_time``. Rows are streamed via
            `gdf.itertuples` (much faster than `iterrows` on large
            catalogs) and routed through `write_partitioned_rows`.
    """
    gdf = catalog.gdf.copy()
    if isinstance(gdf.index, pd.IntervalIndex):
        gdf["start_time"] = gdf.index.left
        gdf["end_time"] = gdf.index.right
        gdf = gdf.reset_index(drop=True)
    if partition_by is not None:
        from geocatalog._src.streaming import write_partitioned_rows

        # `itertuples(index=False)` is ~10-50x faster than `iterrows()` on
        # wide catalogs (`iterrows` materialises a `pd.Series` per row).
        # Access columns by positional index — attribute access via
        # `row.<colname>` chokes on names like ``eo:cloud_cover`` that
        # aren't valid Python identifiers (pandas would silently rename
        # them to ``_0``/etc.).
        columns = list(gdf.columns)
        rows = (
            dict(zip(columns, tup, strict=True))
            for tup in gdf.itertuples(index=False, name=None)
        )
        write_partitioned_rows(
            rows,
            out_path=path,
            crs=gdf.crs,
            backend=catalog.backend,
            partition_by=partition_by,
            schema_version=schema_version,
            write_bbox=write_covering_bbox,
            replace=True,
        )
        return
    gdf["_backend"] = catalog.backend
    gdf["_schema_version"] = schema_version
    gdf.to_parquet(
        Path(path),
        write_covering_bbox=write_covering_bbox,
    )


def from_geoparquet(
    path: str | Path,
    *,
    backend: _BACKEND_T | None = None,
    strict: bool = False,
    retries: int = 3,
    storage_options: dict[str, Any] | None = None,
) -> InMemoryGeoCatalog:
    """Load a GeoParquet file into an `InMemoryGeoCatalog`.

    Inverse of `to_geoparquet`: rebuilds the `IntervalIndex` from
    ``start_time`` / ``end_time`` columns and recovers the backend tag
    from the reserved ``_backend`` column. Externally produced files
    (no ``_backend`` column) default to backend ``"raster"`` — adjust
    on the returned catalog if that's wrong.

    The reserved ``_schema_version`` column drives forward migration
    (see `SCHEMA_VERSION_CURRENT` and `_MIGRATIONS`):

    - ``v_artifact == v_current``: load directly.
    - ``v_artifact <  v_current``: chain forward migrations transparently.
    - ``v_artifact >  v_current``: raise `CatalogSchemaError` — the
      reader is older than the writer and needs upgrading.

    Args:
        path: Path to a GeoParquet file produced by `to_geoparquet`,
            DuckDB's ``COPY ... TO``, or any GeoParquet 1.x writer.
        backend: Loader dispatch tag override. ``None`` reads the
            reserved ``_backend`` column; an explicit value skips tag
            recovery entirely (no warning, no strict raise).
        strict: If ``True`` and neither ``backend=`` nor a ``_backend``
            column is available, raise `CatalogMetadataError` instead of
            defaulting to ``"raster"`` with a warning.
        retries: Number of retries for transient remote I/O failures.
            ``0`` disables retry/backoff.
        storage_options: Options forwarded to fsspec for cloud/HTTP URIs
            (e.g. ``{"anon": True}`` for public S3 buckets). ``None``
            uses fsspec defaults — set explicitly to override credentials.

    Returns:
        An `InMemoryGeoCatalog` with the same rows, CRS, and (where
        recoverable) backend tag as the source.

    Raises:
        CatalogSchemaError: If the artifact's `_schema_version` exceeds
            `SCHEMA_VERSION_CURRENT`.
        CatalogMetadataError: ``strict=True`` and the artifact has no
            ``_backend`` column (and no ``backend=`` override).
    """
    # Read the version *first* via a column-selective parquet load so
    # we can reject a v_future / multi-version artifact before paying
    # for the full read.
    v_artifact = retry_transient_io(
        _read_schema_version,
        path,
        storage_options=storage_options,
        retries=retries,
    )
    resolved = _resolve_uri(path, storage_options=storage_options)
    try:
        gdf = retry_transient_io(gpd.read_parquet, resolved, retries=retries)
    finally:
        _close_resolved_uri(resolved)
    backend_col = gdf.pop("_backend") if "_backend" in gdf.columns else None
    if "_schema_version" in gdf.columns:
        gdf = gdf.drop(columns=["_schema_version"])
    if v_artifact > SCHEMA_VERSION_CURRENT:
        raise CatalogSchemaError(
            f"artifact {Path(path)} has _schema_version={v_artifact}, "
            f"exceeds reader v{SCHEMA_VERSION_CURRENT}. "
            "Upgrade `geocatalog` to read this artifact."
        )
    if v_artifact < SCHEMA_VERSION_CURRENT:
        gdf = _apply_migrations(gdf, from_version=v_artifact)
    if "start_time" in gdf.columns and "end_time" in gdf.columns:
        idx = pd.IntervalIndex.from_arrays(
            gdf.pop("start_time"),
            gdf.pop("end_time"),
            closed="both",
            name="datetime",
        )
        gdf = gdf.set_index(idx)
    if backend is None:
        if backend_col is not None and len(backend_col) > 0:
            backend = backend_col.iloc[0]
        elif strict:
            raise CatalogMetadataError(
                f"{Path(path)} is missing the reserved '_backend' column. "
                "Pass backend=... explicitly, or write the catalog via "
                "geocatalog's to_geoparquet first."
            )
        else:
            log.warning(
                "opened {!r}: no _backend column found; defaulting to "
                "backend='raster'. Pass backend=... explicitly to silence.",
                str(path),
            )
            backend = "raster"
    return InMemoryGeoCatalog(gdf, backend=backend)


def migrate_geoparquet(source: str | Path, *, to_version: int) -> int:
    """Read ``source``, migrate it to ``to_version``, write back in-place.

    A thin file-level wrapper over `from_geoparquet` + `to_geoparquet`
    used by the ``geocatalog migrate`` CLI. The artifact is rewritten
    only if the migration actually changed the version, so calling
    twice is idempotent.

    Args:
        source: GeoParquet file to migrate. Rewritten in-place.
        to_version: Target version. Must equal `SCHEMA_VERSION_CURRENT`
            today (forward-only migrations); kept as an explicit
            parameter so future versions can target a pinned schema.

    Returns:
        The artifact's `_schema_version` *before* the migration. Equal
        to ``to_version`` for already-current files.

    Raises:
        CatalogSchemaError: If ``to_version`` differs from
            `SCHEMA_VERSION_CURRENT`.
    """
    if to_version != SCHEMA_VERSION_CURRENT:
        raise CatalogSchemaError(
            f"migrate target v{to_version} differs from reader "
            f"v{SCHEMA_VERSION_CURRENT}; only forward migrations to the "
            "current version are supported."
        )
    path = Path(source)
    # Cheap version probe via column-selective parquet read — avoids
    # materialising the full gdf just to decide we don't need to.
    from_version = _read_schema_version(path)
    if from_version == to_version:
        return from_version
    cat = from_geoparquet(path)
    to_geoparquet(cat, path, schema_version=to_version)
    return from_version
