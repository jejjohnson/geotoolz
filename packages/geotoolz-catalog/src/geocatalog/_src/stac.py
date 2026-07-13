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

from geocatalog._src._timeutil import to_rfc3339, to_utc_ts
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
        target_crs: Optional CRS for catalog footprints. STAC item bboxes
            are interpreted as EPSG:4326 and reprojected when supplied.
        out_path: GeoParquet destination required by ``backend="duckdb"``.
        extra_properties: STAC property keys to preserve as catalog columns.

    Returns:
        A raster-backend `GeoCatalog` with one row per selected STAC asset.
    """
    if backend not in ("memory", "duckdb"):
        raise ValueError(
            f"from_stac_items: backend must be 'memory' or 'duckdb'; got {backend!r}"
        )
    _require_pystac()

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
    if not rows:
        # Empty STAC searches are common (overly tight bbox, future
        # date window, collection mismatch); return a typed empty
        # catalog rather than raising so callers can branch on `len`.
        empty_gdf = gpd.GeoDataFrame(
            {
                "filepath": [],
                "geometry": [],
                "start_time": pd.Series([], dtype="datetime64[ns]"),
                "end_time": pd.Series([], dtype="datetime64[ns]"),
                "crs": [],
                "asset_key": [],
                "stac_item_id": [],
            },
            geometry="geometry",
            crs=catalog_crs,
        )
        catalog = InMemoryGeoCatalog(empty_gdf, backend="raster")
    else:
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=catalog_crs)
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
            if key not in {"asset_key", "stac_item_id", "stac_collection"}
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


def _item_to_rows(
    item: pystac.Item,
    *,
    asset_key: str | Literal["*"],
    catalog_crs: pyproj.CRS,
    extra_properties: Sequence[str],
) -> list[dict[str, Any]]:
    props = item.properties
    start, end = _item_interval(item)
    geometry = _geometry_from_item(item)
    if catalog_crs != _STAC_CRS:
        transformer = pyproj.Transformer.from_crs(
            _STAC_CRS, catalog_crs, always_xy=True
        )
        geometry = shapely.ops.transform(transformer.transform, geometry)
    keys = item.assets.keys() if asset_key == "*" else (asset_key,)
    rows: list[dict[str, Any]] = []
    for key in keys:
        try:
            asset = item.assets[key]
        except KeyError as exc:
            raise KeyError(f"STAC item {item.id!r} has no asset {key!r}") from exc
        # Per the STAC projection extension, an asset's `proj:epsg` /
        # `proj:wkt2` overrides the item-level value when present.
        asset_extras = getattr(asset, "extra_fields", None) or {}
        asset_props = {**props, **asset_extras}
        row = {
            "filepath": asset.href,
            "geometry": geometry,
            "start_time": start,
            "end_time": end,
            "crs": _asset_crs(asset_props),
            "asset_key": key,
            "stac_item_id": item.id,
        }
        if item.collection_id is not None:
            row["stac_collection"] = item.collection_id
        for prop_key in extra_properties:
            if prop_key in props:
                row[prop_key] = props[prop_key]
        rows.append(row)
    return rows


def _geometry_from_item(item: pystac.Item) -> shapely.geometry.base.BaseGeometry:
    bbox = item.bbox
    if bbox is not None:
        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
        elif len(bbox) == 6:
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[3], bbox[4]
        else:
            raise ValueError(
                f"STAC item {item.id!r} bbox must have 4 or 6 values; got {bbox!r}"
            )
        return shapely.geometry.box(xmin, ymin, xmax, ymax)
    if item.geometry is None:
        raise ValueError(f"STAC item {item.id!r} has neither bbox nor geometry")
    return shapely.geometry.shape(item.geometry)


def _item_interval(item: pystac.Item) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Per STAC 1.0, when an item carries both a nominal `datetime` and a
    # `start_datetime`/`end_datetime` range, the range describes the
    # actual acquisition window and `datetime` is a representative
    # instant inside it. Preferring the range avoids collapsing real
    # time-extent metadata into a zero-width interval.
    props = item.properties
    start = props.get("start_datetime")
    end = props.get("end_datetime")
    if start is not None and end is not None:
        return _timestamp(start), _timestamp(end)
    if item.datetime is not None:
        timestamp = _timestamp(item.datetime)
        return timestamp, timestamp
    raise ValueError(
        f"STAC item {item.id!r} needs datetime or start_datetime/end_datetime"
    )


def _timestamp(value: Any) -> pd.Timestamp:
    """Normalize STAC timestamps to UTC-naive pandas timestamps."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _asset_crs(props: dict[str, Any]) -> str:
    epsg = props.get("proj:epsg")
    if epsg is not None:
        return f"EPSG:{epsg}"
    wkt2 = props.get("proj:wkt2")
    if wkt2 is not None:
        return str(wkt2)
    return "EPSG:4326"


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
    start = _datetime_or_none(interval.left)
    end = _datetime_or_none(interval.right)
    temporal = pystac.TemporalExtent([[start, end]])
    return pystac.Extent(spatial, temporal)


def _geometry_to_stac_crs(
    geometry: shapely.geometry.base.BaseGeometry,
    crs: Any,
) -> shapely.geometry.base.BaseGeometry:
    src = pyproj.CRS.from_user_input(crs)
    if src == _STAC_CRS:
        return geometry
    transformer = pyproj.Transformer.from_crs(src, _STAC_CRS, always_xy=True)
    return shapely.ops.transform(transformer.transform, geometry)


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


def _normalize_crs_property(props: dict[str, Any]) -> None:
    crs = props.pop("crs", None)
    if crs is None or "proj:epsg" in props or "proj:wkt2" in props:
        return
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
