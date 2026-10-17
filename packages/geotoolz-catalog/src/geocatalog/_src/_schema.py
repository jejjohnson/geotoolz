"""The catalog's on-disk schema, spelled once.

Reserved column names, the backend tags, the schema version and the
messages for artifacts the reader cannot load live here so the
in-memory and DuckDB backends, the writers and the builders cannot
drift apart.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
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

StateKind = Literal["analysis", "forecast", "reanalysis"]
"""What a model state is: the reserved ``kind`` column of a state catalog."""

STATE_KINDS: tuple[str, ...] = get_args(StateKind)

STATE_COLUMNS: tuple[str, ...] = ("kind", "run_time", "member", "producer", "parents")
"""Row columns reserved, with fixed types, for model states (schema v1).

``kind`` (string, a `StateKind`), ``run_time`` (``datetime64[us]``, naive
UTC: the run's initialisation time), ``member`` (string: a CF realization
number written as text, or a CMIP variant label such as ``r1i1p1f1``),
``producer`` (string) and ``parents`` (``list<string>``: the URIs the
state was computed from). A catalog carries all five or none of them;
lead time is not stored, it is the row's valid time minus ``run_time``.
They stay ordinary row columns (and `CatalogRow.extras` keys), unlike
`RESERVED_COLUMNS`, which every artifact carries.
"""

SCHEMA_VERSION_CURRENT: int = 1
"""The reader's schema version. Bump on every substantive schema change
and register the ``previous → this`` migration in `parquet._MIGRATIONS`.

v1 reserves `STATE_COLUMNS`."""

LEGACY_UNVERSIONED: int = 0
"""Version assumed for artifacts written before ``_schema_version`` existed.

Pinned (not `SCHEMA_VERSION_CURRENT`) so the next schema bump still
migrates those legacy files instead of silently skipping them.
"""


MIGRATION_TOUCHES: dict[int, frozenset[str]] = {0: frozenset(STATE_COLUMNS)}
"""The columns the ``v → v+1`` migration reads, by source version ``v``.

An artifact with none of them is already at ``v+1`` in all but its
stamp, so a reader that cannot migrate (DuckDB) may open it as is.
v0 → v1 only types the state columns: every v0 raster, vector or
xarray catalog without them, and every external GeoParquet, opens.
"""


def migration_is_noop(lo: int, reader: int, columns: Collection[str]) -> bool:
    """True when migrating an artifact with ``columns`` from ``lo`` changes nothing."""
    present = set(columns)
    for version in range(lo, reader):
        touched = MIGRATION_TOUCHES.get(version)
        if touched is None or touched & present:
            return False
    return True


def check_schema_versions(
    source: object,
    lo: int,
    hi: int,
    *,
    reader: int,
    can_migrate: bool,
    columns: Collection[str] | None = None,
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
        columns: The artifact's column names. When given, an older
            artifact that the migrations would leave unchanged
            (`migration_is_noop`) is accepted even without
            ``can_migrate``.
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
    if (
        hi < reader
        and not can_migrate
        and not (columns is not None and migration_is_noop(lo, reader, columns))
    ):
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


def state_arrow_types() -> dict[str, Any]:
    """The Arrow type of each of `STATE_COLUMNS`, for the streaming writer."""
    import pyarrow as pa

    return {
        "kind": pa.string(),
        "run_time": pa.timestamp("us"),
        "member": pa.string(),
        "producer": pa.string(),
        "parents": pa.list_(pa.string()),
    }


def _missing(value: Any) -> bool:
    """True for ``None`` and scalar NA (``NaN``, ``NaT``, ``pd.NA``)."""
    import pandas as pd

    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # arrays and other non-scalars
        return False


def state_text(value: Any) -> str | None:
    """A ``kind`` / ``member`` / ``producer`` value as text, or ``None``.

    Integral numbers lose their ``.0`` (a realization read back from a
    float column is ``"0"``, not ``"0.0"``).
    """
    import numbers

    if _missing(value):
        return None
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, numbers.Real) and float(value).is_integer():
        return str(int(value))
    return str(value)


def state_parents(value: Any) -> list[str] | None:
    """A ``parents`` value as a list of URIs: one string becomes a list of one."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [state_text(value) or ""]
    if _missing(value):
        return None
    return [str(v) for v in value]


def coerce_state_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Give a frame's `STATE_COLUMNS` their v1 types; add the missing ones as nulls.

    A frame with none of the five names is returned unchanged: it is not a
    state catalog. One with any of them (a v0 catalog whose extras were
    filled by hand, or rows read back from Parquet or DuckDB) gets all
    five, typed: ``run_time`` as naive-UTC ``datetime64[us]``, ``member``
    as text (``0`` becomes ``"0"``), ``parents`` as lists of strings.

    Args:
        gdf: A catalog frame.

    Returns:
        A new frame (or ``gdf`` itself when it carries no state column).
    """
    import pandas as pd

    from geocatalog._src.utils.timeutil import naive_utc_datetimes

    if not any(name in gdf.columns for name in STATE_COLUMNS):
        return gdf
    gdf = gdf.copy()
    n = len(gdf)
    for name in ("kind", "member", "producer"):
        values = gdf[name] if name in gdf.columns else [None] * n
        gdf[name] = pd.Series(
            [state_text(v) for v in values], index=gdf.index, dtype="str"
        )
    if "run_time" in gdf.columns:
        times = naive_utc_datetimes(
            pd.Series(pd.to_datetime(gdf["run_time"], utc=False), index=gdf.index)
        )
    else:
        times = pd.Series(pd.NaT, index=gdf.index)
    gdf["run_time"] = times.astype("datetime64[us]")
    values = gdf["parents"] if "parents" in gdf.columns else [None] * n
    gdf["parents"] = pd.Series(
        [state_parents(v) for v in values], index=gdf.index, dtype=object
    )
    return gdf


__all__ = [
    "BACKEND_TAGS",
    "INTERNAL_COLUMNS",
    "LEGACY_UNVERSIONED",
    "MIGRATION_TOUCHES",
    "RESERVED_COLUMNS",
    "SCHEMA_VERSION_CURRENT",
    "STATE_COLUMNS",
    "STATE_KINDS",
    "CatalogKind",
    "StateKind",
    "StorageEngine",
    "check_kind",
    "check_schema_versions",
    "coerce_state_columns",
    "crs_config_string",
    "empty_frame",
    "migration_is_noop",
    "state_arrow_types",
    "state_parents",
    "state_text",
]
