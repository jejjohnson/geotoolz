"""The catalog's on-disk schema, spelled once.

Reserved column names, the backend tags, the schema version and the
messages for artifacts the reader cannot load live here so the
in-memory and DuckDB backends, the writers and the builders cannot
drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, get_args


if TYPE_CHECKING:
    import geopandas as gpd


RESERVED_COLUMNS: frozenset[str] = frozenset(
    {"filepath", "start_time", "end_time", "geometry", "bbox"}
)
"""Column names reserved by the catalog schema itself.

Every GeoParquet artifact carries these as its required layout
(``bbox`` only when the writer emits the GeoParquet 1.1 covering
struct); backend row iterators and schema builders treat anything
else as user extras.
"""

INTERNAL_COLUMNS: frozenset[str] = frozenset({"_backend", "_schema_version"})
"""Writer-managed metadata columns of the on-disk schema.

Appended by `StreamingParquetWriter` (and `to_geoparquet`) to every
artifact; they are not user-visible row metadata, so readers filter
them out of ``extras`` and rewrite passes drop them before re-encoding
(the writer re-emits its own).
"""

CatalogKind = Literal["raster", "xarray", "vector"]
"""The kind of data a catalog indexes: `.kind` (stored as the ``_backend`` column)."""

BACKEND_TAGS: tuple[str, ...] = get_args(CatalogKind)


def check_kind(kind: object, where: str) -> None:
    """Raise `ValueError` unless ``kind`` is a `CatalogKind`."""
    if kind not in BACKEND_TAGS:
        raise ValueError(
            f"{where} kind must be one of {list(BACKEND_TAGS)}; got {kind!r}."
        )


StorageEngine = Literal["memory", "duckdb"]
"""Where a built or opened catalog lives: in RAM, or a DuckDB relation."""

SCHEMA_VERSION_CURRENT: int = 0
"""The reader's schema version. Bump on every substantive schema change
and register the ``previous → this`` migration in `parquet._MIGRATIONS`."""

LEGACY_UNVERSIONED: int = 0
"""Version assumed for artifacts written before ``_schema_version`` existed.

Pinned (not `SCHEMA_VERSION_CURRENT`) so the next schema bump still
migrates those legacy files instead of silently skipping them.
"""


def check_schema_versions(
    source: object, lo: int, hi: int, *, reader: int, can_migrate: bool
) -> None:
    """Raise `CatalogSchemaError` for versions this reader cannot open.

    Args:
        source: The artifact, for the message.
        lo: Smallest ``_schema_version`` in the artifact.
        hi: Largest ``_schema_version`` in the artifact.
        reader: The reader's version (`SCHEMA_VERSION_CURRENT`; passed in
            so a caller's module-level value is the one checked).
        can_migrate: Whether the caller runs forward migrations itself
            (the in-memory reader does; DuckDB does not, so an older
            artifact is rejected with a pointer to ``geocatalog migrate``).
    """
    from geocatalog._src.base import CatalogSchemaError

    if lo != hi:
        raise CatalogSchemaError(
            f"artifact {source} has mixed `_schema_version` values "
            f"(min={lo}, max={hi}); the reader can't open a multi-version "
            "source. Migrate each shard separately, or rewrite into one "
            "file at a single version."
        )
    if hi > reader:
        raise CatalogSchemaError(
            f"artifact {source} has _schema_version={hi}, exceeds reader "
            f"v{reader}. Upgrade `geotoolz-catalog` to read "
            "this artifact."
        )
    if hi < reader and not can_migrate:
        raise CatalogSchemaError(
            f"artifact {source} has _schema_version={hi} < reader "
            f"v{reader}. The DuckDB backend does not run "
            "forward migrations in-place; bring the artifact up to date with "
            f"`geocatalog migrate {source} --to-version {reader}`."
        )


def crs_config_string(crs: Any) -> str:
    """The CRS as ``get_config()`` reports it: one string per CRS, stable.

    ``"AUTH:CODE"`` when the CRS carries an authority identifier
    (``"EPSG:32629"``, ``"OGC:CRS84"``), else its WKT2. The PROJJSON a
    GeoParquet artifact stores round-trips to the same string, so the
    in-memory and DuckDB backends, and a catalog before and after a
    write, report the same value. (``CRS.to_string()`` would print a
    PROJ string for one and PROJJSON for the other.)
    """
    import pyproj

    parsed = pyproj.CRS.from_user_input(crs)
    ident = parsed.to_json_dict().get("id")
    if isinstance(ident, dict) and "authority" in ident and "code" in ident:
        return f"{ident['authority']}:{ident['code']}"
    return parsed.to_wkt()


def empty_frame(crs: Any, columns: Mapping[str, str] | None = None) -> gpd.GeoDataFrame:
    """A zero-row catalog frame: typed ``columns``, geometry, interval index.

    Args:
        crs: CRS of the geometry column.
        columns: Extra column name → dtype (``"object"``, ``"bool"``, …),
            in order, before the geometry.

    Returns:
        A `GeoDataFrame` with a ``closed="both"`` `IntervalIndex` named
        ``datetime`` — the layout every catalog constructor expects.
    """
    import geopandas as gpd
    import pandas as pd

    gdf = gpd.GeoDataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in (columns or {}).items()},
        geometry=gpd.GeoSeries([], crs=crs),
    )
    gdf.index = pd.IntervalIndex.from_arrays(
        pd.to_datetime([]), pd.to_datetime([]), closed="both", name="datetime"
    )
    return gdf


__all__ = [
    "BACKEND_TAGS",
    "INTERNAL_COLUMNS",
    "LEGACY_UNVERSIONED",
    "RESERVED_COLUMNS",
    "SCHEMA_VERSION_CURRENT",
    "CatalogKind",
    "StorageEngine",
    "check_kind",
    "check_schema_versions",
    "crs_config_string",
    "empty_frame",
]
