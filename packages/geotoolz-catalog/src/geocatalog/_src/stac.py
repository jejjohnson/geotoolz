"""STAC interoperability helpers for catalog builders."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import pandas as pd
import pyproj
import shapely.geometry
import shapely.ops

from geocatalog._src._stac_item import (
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
        rows.extend(
            _item_to_rows(
                item,
                asset_key=asset_key,
                catalog_crs=catalog_crs,
                extra_properties=extra_properties,
            )
        )
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
    """Convert a catalog into a STAC collection with one item per row."""
    pystac = _require_pystac()
    extent = _collection_extent(catalog, pystac)
    collection = pystac.Collection(
        id=collection_id,
        description=description,
        extent=extent,
    )

    for idx, row in enumerate(catalog.iter_rows()):
        geom = _geometry_to_stac_crs(row.geometry, row.crs)
        props = {
            key: value
            for key, value in row.extras.items()
            if key
            not in {"asset_key", "stac_item_id", "stac_collection", "href_signed"}
        }
        _normalize_crs_property(props)
        start = _datetime_or_none(row.interval.left)
        end = _datetime_or_none(row.interval.right)
        # STAC / RFC3339 require tz-aware ISO 8601 with `Z` (or offset)
        # for `start_datetime` / `end_datetime` and the item-level
        # `datetime`. Naive timestamps from the catalog time-axis are
        # treated as UTC.
        item_datetime = _to_utc_datetime(start) if _same_instant(start, end) else None
        if item_datetime is None:
            props["start_datetime"] = _to_rfc3339(start)
            props["end_datetime"] = _to_rfc3339(end)

        item = pystac.Item(
            id=str(row.extras.get("stac_item_id", f"{collection_id}-{idx}")),
            geometry=shapely.geometry.mapping(geom),
            bbox=tuple(geom.bounds),
            datetime=item_datetime,
            properties=props,
            collection=collection_id,
        )
        item.add_asset(
            str(row.extras.get("asset_key", asset_key)),
            pystac.Asset(href=row.filepath),
        )
        collection.add_item(item)
    return collection


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
