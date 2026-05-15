"""`to_geoparquet` / `from_geoparquet` — catalog ↔ portable artifact.

Round-trips an `InMemoryGeoCatalog` through GeoParquet 1.1 (geopandas
writes the bbox covering struct when ``write_covering_bbox=True``). The
artifact is then queryable from pandas / DuckDB / GDAL without
ceremony — no pickle-version fragility.

Phase 2 (DuckDB backend) reads the *same* GeoParquet file, so
``to_geoparquet`` writes the canonical interchange format for both
backends. See ``research_journal_v2/notes/geotoolz/plans/geodatabase/
geocatalog.md`` §10.2 for the GeoParquet 1.1 adoption caveat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd

from geotoolz.catalog._src.memory import InMemoryGeoCatalog


_BACKEND_T = Literal["raster", "xarray", "vector"]


def to_geoparquet(
    catalog: InMemoryGeoCatalog,
    path: str | Path,
    *,
    schema_version: int = 0,
    write_covering_bbox: bool = True,
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
            ``_schema_version`` column. Default 0.
        write_covering_bbox: Emit the per-row ``bbox`` covering struct
            that GeoParquet 1.1 readers (DuckDB, geopandas ≥0.14) use
            for predicate pushdown. Default True; turn off only if a
            downstream consumer chokes on 1.1.
    """
    gdf = catalog.gdf.copy()
    if isinstance(gdf.index, pd.IntervalIndex):
        gdf["start_time"] = gdf.index.left
        gdf["end_time"] = gdf.index.right
        gdf = gdf.reset_index(drop=True)
    gdf["_backend"] = catalog.backend
    gdf["_schema_version"] = schema_version
    gdf.to_parquet(
        Path(path),
        write_covering_bbox=write_covering_bbox,
    )


def from_geoparquet(path: str | Path) -> InMemoryGeoCatalog:
    """Load a GeoParquet file into an `InMemoryGeoCatalog`.

    Inverse of `to_geoparquet`: rebuilds the `IntervalIndex` from
    ``start_time`` / ``end_time`` columns and recovers the backend tag
    from the reserved ``_backend`` column. Externally produced files
    (no ``_backend`` column) default to backend ``"raster"`` — adjust
    on the returned catalog if that's wrong.

    Args:
        path: Path to a GeoParquet file produced by `to_geoparquet`,
            DuckDB's ``COPY ... TO``, or any GeoParquet 1.x writer.

    Returns:
        An `InMemoryGeoCatalog` with the same rows, CRS, and (where
        recoverable) backend tag as the source.
    """
    gdf = gpd.read_parquet(Path(path))
    backend_col = gdf.pop("_backend") if "_backend" in gdf.columns else None
    if "_schema_version" in gdf.columns:
        gdf = gdf.drop(columns=["_schema_version"])
    if "start_time" in gdf.columns and "end_time" in gdf.columns:
        idx = pd.IntervalIndex.from_arrays(
            gdf.pop("start_time"),
            gdf.pop("end_time"),
            closed="both",
            name="datetime",
        )
        gdf = gdf.set_index(idx)
    if backend_col is not None and len(backend_col) > 0:
        backend: _BACKEND_T = backend_col.iloc[0]
    else:
        backend = "raster"
    return InMemoryGeoCatalog(gdf, backend=backend)
