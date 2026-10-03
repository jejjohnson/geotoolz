"""STAC interoperability helpers for catalog builders."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime as _PyDatetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely.geometry
import shapely.ops
from loguru import logger as log

from geocatalog._src._stac_item import (
    UndecodableItemError,
    asset_crs,
    asset_href,
    is_signed_href,
    item_geometry,
    item_interval,
    reproject_geometry,
)
from geocatalog._src._timeutil import to_naive_utc, to_rfc3339, to_utc_ts
from geocatalog._src.base import GeoCatalog
from geocatalog._src.memory import InMemoryGeoCatalog
from geocatalog._src.parquet import to_geoparquet


if TYPE_CHECKING:
    import pystac


_BACKEND_T = Literal["memory", "duckdb"]
_STAC_CRS = pyproj.CRS.from_epsg(4326)


def from_stac_items(
    items: Iterable[pystac.Item],
    *,
    asset_key: str | Literal["*"] = "data",
    backend: _BACKEND_T = "memory",
    target_crs: Any | None = None,
    out_path: Path | None = None,
    extra_properties: Sequence[str] = (),
) -> GeoCatalog:
    """Build a raster catalog from STAC items.

    Args:
        items: STAC items to index.
        asset_key: Asset key to index. Pass ``"*"`` to emit one row for
            every asset on each item.
        backend: ``"memory"`` or ``"duckdb"``.
        target_crs: Optional CRS for catalog footprints. Item geometries
            are lon/lat (EPSG:4326) and are reprojected, with edge
            densification, when supplied.
        out_path: GeoParquet destination required by ``backend="duckdb"``.
        extra_properties: STAC property keys to preserve as catalog columns.
            Must not name one of the columns the builder writes itself.

    Returns:
        A raster-backend `GeoCatalog` with one row per selected STAC asset.
        Footprints come from ``item.geometry`` (``item.bbox`` only when
        the geometry is missing), split at the antimeridian; ``crs`` is
        the asset's native CRS from the projection extension
        (``proj:code`` / ``proj:epsg`` / ``proj:wkt2`` /
        ``proj:projjson``); ``filepath`` is the absolute asset href and
        ``href_signed`` flags hrefs carrying an expiring SAS signature
        (re-sign them, e.g. with ``planetary_computer.sign``, before
        reading once the token has lapsed). Times are naive UTC.

        An item with neither a geometry nor a bbox, or with no time, is
        skipped with a warning.

    Raises:
        ValueError: ``extra_properties`` names a builder-owned column.
    """
    if backend not in ("memory", "duckdb"):
        raise ValueError(
            f"from_stac_items: backend must be 'memory' or 'duckdb'; got {backend!r}"
        )
    _require_pystac()
    clashes = sorted(set(extra_properties) & _STAC_ROW_COLUMNS)
    if clashes:
        raise ValueError(
            f"from_stac_items: extra_properties {clashes} would overwrite "
            "columns the builder writes; drop them from extra_properties."
        )

    catalog_crs = (
        pyproj.CRS.from_user_input(target_crs) if target_crs is not None else _STAC_CRS
    )
    rows: list[dict[str, Any]] = []
    for item in items:
        try:
            item_rows = _item_to_rows(
                item,
                asset_key=asset_key,
                catalog_crs=catalog_crs,
                extra_properties=extra_properties,
            )
        except UndecodableItemError as exc:
            # No footprint or no time: skip with a warning, the policy
            # every Source adapter follows too.
            log.warning("from_stac_items: skipping item {!r}: {}", item.id, exc)
            continue
        rows.extend(item_rows)
    # Empty STAC searches are common (overly tight bbox, future date
    # window, collection mismatch); return a typed empty catalog with
    # the same columns rather than raising so callers can branch on `len`.
    columns = {name: [r.get(name) for r in rows] for name in _STAC_ROW_ORDER}
    gdf = gpd.GeoDataFrame(
        {k: v for k, v in columns.items() if k not in _TIME_COLUMNS},
        geometry="geometry",
        crs=catalog_crs,
    )
    # An empty list infers float64; keep the flag boolean either way.
    gdf["href_signed"] = gdf["href_signed"].astype(bool)
    gdf.index = pd.IntervalIndex.from_arrays(
        pd.to_datetime(columns["start_time"]),
        pd.to_datetime(columns["end_time"]),
        closed="both",
        name="datetime",
    )
    for prop_key in extra_properties:
        gdf[prop_key] = [r.get(prop_key) for r in rows]
    catalog = InMemoryGeoCatalog(gdf, backend="raster")
    if backend == "memory":
        return catalog
    if out_path is None:
        raise ValueError("from_stac_items(backend='duckdb') requires out_path")
    to_geoparquet(catalog, out_path)
    from geocatalog._src.duckdb_backend import DuckDBGeoCatalog

    return DuckDBGeoCatalog.open(out_path, backend="raster", crs=catalog_crs)


def from_stac_search(
    client: Any | str,
    *,
    collections: Sequence[str],
    bbox: tuple[float, float, float, float] | None = None,
    datetime: str | None = None,
    asset_key: str | Literal["*"] = "data",
    backend: _BACKEND_T = "memory",
    max_items: int | None = None,
    target_crs: Any | None = None,
    out_path: Path | None = None,
    extra_properties: Sequence[str] = (),
) -> GeoCatalog:
    """Run a STAC API search and build a raster catalog from its items.

    Args:
        client: Open `pystac_client.Client` or STAC API URL.
        collections: Collection IDs to search.
        bbox: Optional lon/lat search bbox.
        datetime: Optional STAC datetime interval string.
        asset_key: Asset key to index. Pass ``"*"`` to emit one row for
            every asset on each item.
        backend: ``"memory"`` or ``"duckdb"``.
        max_items: Optional maximum number of returned search items to index.
        target_crs: Optional CRS for catalog footprints.
        out_path: GeoParquet destination required by ``backend="duckdb"``.
        extra_properties: STAC property keys to preserve as catalog columns.

    Returns:
        A raster-backend `GeoCatalog` over the matching STAC assets.
    """
    pystac_client = _require_pystac_client()
    if isinstance(client, str):
        client = pystac_client.Client.open(client)

    search = client.search(
        collections=collections,
        bbox=bbox,
        datetime=datetime,
        max_items=max_items,
    )
    item_iter = search.items()
    if max_items is not None:
        item_iter = itertools.islice(item_iter, max_items)
    return from_stac_items(
        item_iter,
        asset_key=asset_key,
        backend=backend,
        target_crs=target_crs,
        out_path=out_path,
        extra_properties=extra_properties,
    )


def to_stac_collection(
    catalog: GeoCatalog,
    *,
    collection_id: str,
    description: str = "",
    asset_key: str = "data",
) -> pystac.Collection:
    """Convert a catalog into a STAC collection.

    Rows are grouped into one item per scene: rows sharing a source
    ``stac_collection`` and a ``stac_item_id`` (or, failing that, an
    ``id`` — the `CatalogBundle` primary key) become one item whose
    assets are keyed by each row's ``asset_key``. An id that occurs in
    more than one source collection is exported as
    ``"<source collection>:<id>"`` so the output ids stay unique. A row
    with neither id gets its own item, ``"<collection_id>-<row>"``. The
    item's geometry, time and properties come from its first row.

    Properties are made JSON-safe: numpy scalars become Python numbers,
    timestamps RFC 3339 strings, and missing values (``None``, ``NaN``,
    ``NaT``, non-finite floats) are omitted, so
    ``json.dumps(collection.to_dict())`` always works.

    Args:
        catalog: Catalog to export.
        collection_id: The collection id; also stamped on every item.
        description: Collection description.
        asset_key: Asset key for rows without an ``asset_key`` column.

    Returns:
        A `pystac.Collection` holding the items.

    Raises:
        ValueError: Two rows of the same item carry the same asset key.
    """
    pystac = _require_pystac()
    extent = _collection_extent(catalog, pystac)
    collection = pystac.Collection(
        id=collection_id,
        description=description,
        extent=extent,
    )

    groups = _group_scenes(catalog, collection_id)
    for item_id, rows in groups.items():
        first = rows[0]
        geom = _geometry_to_stac_crs(first.geometry, first.crs)
        props = {
            key: value for key, value in first.extras.items() if key not in _EXPORT_SKIP
        }
        # An item-level CRS is inherited by every asset without its own,
        # so it is only written when all the item's assets share it.
        # Without a resolved `crs` column the carried `proj:*` properties
        # are the only CRS record, so they are kept as they are.
        has_crs_column = any("crs" in row.extras for row in rows)
        crs_values = [row.extras.get("crs") for row in rows]
        shared_crs = _same_crs(crs_values)
        if has_crs_column and not shared_crs:
            props.pop("crs", None)
            for field in _PROJ_CRS_FIELDS:
                props.pop(field, None)
        _normalize_crs_property(props)
        props = _json_safe_mapping(props)
        start = _datetime_or_none(first.interval.left)
        end = _datetime_or_none(first.interval.right)
        # STAC / RFC3339 require tz-aware ISO 8601 with `Z` (or offset)
        # for `start_datetime` / `end_datetime` and the item-level
        # `datetime`. Naive timestamps from the catalog time-axis are
        # treated as UTC.
        item_datetime = _to_utc_datetime(start) if _same_instant(start, end) else None
        if item_datetime is None:
            props["start_datetime"] = _to_rfc3339(start)
            props["end_datetime"] = _to_rfc3339(end)

        item = pystac.Item(
            id=item_id,
            geometry=shapely.geometry.mapping(geom),
            bbox=tuple(geom.bounds),
            datetime=item_datetime,
            properties=props,
            collection=collection_id,
        )
        for row in rows:
            key = row.extras.get("asset_key")
            key = str(key) if _present(key) else asset_key
            if key in item.assets:
                raise ValueError(
                    f"to_stac_collection: item {item_id!r} has more than one "
                    f"row for asset {key!r}"
                )
            fields: dict[str, Any] = {}
            crs = row.extras.get("crs")
            if _present(crs) and not shared_crs:
                # Per the projection extension, an asset-level CRS
                # overrides the item's.
                fields = {"crs": crs}
                _normalize_crs_property(fields)
            item.add_asset(key, pystac.Asset(href=row.filepath, extra_fields=fields))
        collection.add_item(item)
    return collection


def _same_crs(values: list[Any]) -> bool:
    """True when every value is a CRS and all denote the same one.

    Compared as parsed `pyproj.CRS` objects: a CRS may be stored in any
    form `pyproj.CRS.from_user_input` accepts, including unhashable
    PROJJSON mappings.
    """
    if not values or not all(_present(v) for v in values):
        return False
    first = pyproj.CRS.from_user_input(values[0])
    return all(pyproj.CRS.from_user_input(v).equals(first) for v in values[1:])


def _group_scenes(catalog: GeoCatalog, collection_id: str) -> dict[str, list[Any]]:
    """Rows grouped into scenes, keyed by a unique output item id.

    A scene is the rows sharing an upstream id *within its scope* —
    ``stac_collection``, else the bundle's ``(source, collection)`` —
    since upstream ids are only unique within a collection. Rows without
    an id are each their own scene. Output ids: the upstream id; one that
    occurs in several scopes becomes ``"<scope>:<id>"``; an id-less row
    gets ``"<collection_id>-<row>"``; any remaining clash (with an
    explicit id, say) gets a ``~<n>`` suffix. Explicit ids are allocated
    first, so they keep their spelling.
    """
    scenes: dict[tuple[Any, ...], list[Any]] = {}
    for idx, row in enumerate(catalog.iter_rows()):
        item_id = next(
            (
                str(row.extras[key])
                for key in ("stac_item_id", "id")
                if _present(row.extras.get(key))
            ),
            None,
        )
        if item_id is None:
            scenes[("row", idx)] = [row]
            continue
        # Tagged tuples keep the scope's components apart: flattening
        # ("a", "b/c") and ("a/b", "c") to one string would merge scenes.
        stac_collection = row.extras.get("stac_collection")
        if _present(stac_collection):
            scope: tuple[Any, ...] | None = ("stac", str(stac_collection))
        else:
            # Positional: a missing source or collection stays a `None`
            # slot, so ("a", None) and (None, "a") remain distinct.
            parts = tuple(
                str(row.extras[k]) if _present(row.extras.get(k)) else None
                for k in ("source", "collection")
            )
            scope = ("bundle", *parts) if any(parts) else None
        scenes.setdefault(("id", scope, item_id), []).append(row)

    scopes_per_id: dict[str, set[Any]] = {}
    for key in scenes:
        if key[0] == "id":
            scopes_per_id.setdefault(key[2], set()).add(key[1])

    def candidate(key: tuple[Any, ...]) -> str:
        if key[0] == "row":
            return f"{collection_id}-{key[1]}"
        _, scope, item_id = key
        if len(scopes_per_id[item_id]) > 1 and scope is not None:
            # Two scopes may spell the same; the `~<n>` pass separates them.
            return f"{'/'.join(p for p in scope[1:] if p is not None)}:{item_id}"
        return item_id

    used: set[str] = set()
    names: dict[tuple[Any, ...], str] = {}
    # Explicit ids first, then id-less rows, each made unique on clash.
    for key in sorted(scenes, key=lambda k: k[0] != "id"):
        name = base = candidate(key)
        n = 1
        while name in used:
            n += 1
            name = f"{base}~{n}"
        used.add(name)
        names[key] = name
    return {names[key]: rows for key, rows in scenes.items()}


# Catalog columns that describe the item / asset rather than being item
# properties.
_EXPORT_SKIP = frozenset(
    {"asset_key", "stac_item_id", "stac_collection", "href_signed", "id"}
)


def _present(value: Any) -> bool:
    """False for ``None`` and pandas / numpy missing scalars."""
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):  # list-likes: present
        return True


def _json_safe(value: Any) -> tuple[bool, Any]:
    """``(keep, value)`` with ``value`` made JSON-serialisable.

    Missing values (``None``, ``NaN``, ``NaT``, ``pd.NA``, ±inf) are
    dropped (``keep=False``); numpy scalars become Python scalars;
    timestamps become RFC 3339 strings; containers are converted
    recursively; anything else unknown becomes its ``str``.
    """
    # Before the generic numpy branch: `.item()` on a nanosecond
    # datetime64 returns a raw int, not a datetime.
    if isinstance(value, np.datetime64):
        ts = pd.Timestamp(value)
        return (False, None) if ts is pd.NaT else (True, to_rfc3339(ts))
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _json_safe(value[()])
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NaT or value is pd.NA:
        return False, None
    if isinstance(value, bool | int | str):
        return True, value
    if isinstance(value, float):
        return math.isfinite(value), value
    if isinstance(value, pd.Timestamp | _PyDatetime):
        ts = pd.Timestamp(value)
        return (False, None) if ts is pd.NaT else (True, to_rfc3339(ts))
    if isinstance(value, Mapping):
        return True, _json_safe_mapping(value)
    if isinstance(value, list | tuple | np.ndarray):
        items = [_json_safe(v) for v in value]
        return True, [v if keep else None for keep, v in items]
    return True, str(value)


def _json_safe_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        keep, safe = _json_safe(value)
        if keep:
            out[str(key)] = safe
    return out


# Columns `_item_to_rows` writes, in order; `start_time` / `end_time`
# become the catalog's IntervalIndex.
_STAC_ROW_ORDER = (
    "filepath",
    "href_signed",
    "geometry",
    "start_time",
    "end_time",
    "crs",
    "asset_key",
    "stac_item_id",
    "stac_collection",
)
_STAC_ROW_COLUMNS = frozenset(_STAC_ROW_ORDER)
_TIME_COLUMNS = frozenset({"start_time", "end_time"})


def _item_to_rows(
    item: pystac.Item,
    *,
    asset_key: str | Literal["*"],
    catalog_crs: pyproj.CRS,
    extra_properties: Sequence[str],
) -> list[dict[str, Any]]:
    props = item.properties
    interval = item_interval(item)
    geometry = reproject_geometry(item_geometry(item), _STAC_CRS, catalog_crs)
    keys = item.assets.keys() if asset_key == "*" else (asset_key,)
    rows: list[dict[str, Any]] = []
    for key in keys:
        try:
            asset = item.assets[key]
        except KeyError as exc:
            raise KeyError(f"STAC item {item.id!r} has no asset {key!r}") from exc
        href = asset_href(asset)
        row = {
            "filepath": href,
            "href_signed": is_signed_href(href),
            "geometry": geometry,
            "start_time": to_naive_utc(interval.left),
            "end_time": to_naive_utc(interval.right),
            # Per the STAC projection extension an asset's `proj:*`
            # overrides the item-level value.
            "crs": asset_crs(props, getattr(asset, "extra_fields", None) or {}),
            "asset_key": key,
            "stac_item_id": item.id,
            "stac_collection": item.collection_id,
        }
        for prop_key in extra_properties:
            if prop_key in props:
                row[prop_key] = props[prop_key]
        rows.append(row)
    return rows


def _collection_extent(catalog: GeoCatalog, pystac: Any) -> Any:
    bounds = catalog.total_bounds
    if any(pd.isna(value) for value in bounds):
        spatial = pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]])
    else:
        catalog_crs = (
            catalog.gdf.crs
            if isinstance(catalog.gdf.crs, pyproj.CRS)
            else pyproj.CRS.from_user_input(catalog.gdf.crs)
        )
        if catalog_crs != _STAC_CRS:
            transformer = pyproj.Transformer.from_crs(
                catalog_crs, _STAC_CRS, always_xy=True
            )
            bounds = transformer.transform_bounds(*bounds)
        spatial = pystac.SpatialExtent([list(bounds)])

    interval = catalog.temporal_extent
    start = None if interval is None else _datetime_or_none(interval.left)
    end = None if interval is None else _datetime_or_none(interval.right)
    temporal = pystac.TemporalExtent([[start, end]])
    return pystac.Extent(spatial, temporal)


def _geometry_to_stac_crs(
    geometry: shapely.geometry.base.BaseGeometry,
    crs: Any,
) -> shapely.geometry.base.BaseGeometry:
    return reproject_geometry(geometry, crs, _STAC_CRS)


def _datetime_or_none(value: Any) -> Any | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).to_pydatetime()


def _to_utc_datetime(value: Any | None) -> Any | None:
    """Coerce a Python datetime to tz-aware UTC; naive inputs assumed UTC.

    None-passing wrapper over `geocatalog._src._timeutil.to_utc_ts`.
    pystac requires tz-aware datetimes for RFC3339-canonical output,
    hence the ``.to_pydatetime()`` conversion.
    """
    if value is None:
        return None
    return to_utc_ts(value).to_pydatetime()


def _to_rfc3339(value: Any | None) -> str | None:
    """Serialize a datetime as STAC-canonical RFC3339 (UTC, ``Z`` suffix).

    None-passing wrapper over `geocatalog._src._timeutil.to_rfc3339`.
    """
    if value is None:
        return None
    return to_rfc3339(value)


def _same_instant(left: Any | None, right: Any | None) -> bool:
    if left is None or right is None:
        return False
    return pd.Timestamp(left) == pd.Timestamp(right)


_PROJ_CRS_FIELDS = ("proj:code", "proj:epsg", "proj:wkt2", "proj:projjson")


def _normalize_crs_property(props: dict[str, Any]) -> None:
    """Replace a row's ``crs`` with the projection-extension field for it.

    The row's ``crs`` is the selected asset's resolved CRS, so any
    projection field carried along as an item property (via
    ``extra_properties``) is dropped first: keeping it could contradict
    the asset CRS.
    """
    if "crs" not in props:
        return  # no resolved CRS column: leave carried properties alone
    crs = props.pop("crs")
    for field in _PROJ_CRS_FIELDS:
        props.pop(field, None)
    if crs is None or (isinstance(crs, float) and pd.isna(crs)):
        return  # explicitly unlocated asset: export no projection
    parsed = pyproj.CRS.from_user_input(crs)
    epsg = parsed.to_epsg()
    if epsg is not None:
        props["proj:epsg"] = epsg
    else:
        props["proj:wkt2"] = parsed.to_wkt()


def _require_pystac() -> Any:
    try:
        import pystac
    except ImportError as exc:
        raise ImportError(
            "`geocatalog` STAC helpers require the [stac] extra; install via "
            "`pip install 'geocatalog[stac]'`."
        ) from exc
    return pystac


def _require_pystac_client() -> Any:
    try:
        import pystac_client
    except ImportError as exc:
        raise ImportError(
            "`from_stac_search` requires the [stac] extra; install via "
            "`pip install 'geocatalog[stac]'`."
        ) from exc
    return pystac_client
