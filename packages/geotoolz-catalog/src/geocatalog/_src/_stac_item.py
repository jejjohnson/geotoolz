"""One decoder for STAC items, shared by every STAC ingest path (#238).

`from_stac_items` / `from_stac_search` (catalog builders) and
`STACSource` (the `Source` adapter feeding `CatalogBundle`) used to read
the same item differently — one indexed ``item.bbox``, the other
``item.geometry``; one stripped time zones, the other kept them; neither
honoured ``proj:code``. Both now go through these helpers. pystac is
only touched through the duck-typed item / asset objects passed in, so
importing this module needs no optional extra.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, urlparse

import pandas as pd
import pyproj
import shapely
import shapely.geometry
import shapely.geometry.base
import shapely.ops

from geocatalog._src._antimeridian import (
    crosses_antimeridian,
    lonlat_box,
    lonlat_line,
    lonlat_polygon,
)
from geocatalog._src._timeutil import to_utc_ts


LONLAT = pyproj.CRS.from_epsg(4326)

#: `SourceRow.properties` key under which `STACSource` records each
#: asset's native CRS (``{asset_key: crs}``), so the bundle row can
#: carry the promoted asset's CRS like `from_stac_items` does.
ASSET_CRS_PROPERTY = "geocatalog:asset_crs"


def item_geometry(item: Any) -> shapely.geometry.base.BaseGeometry:
    """The item's footprint in lon/lat.

    Uses ``item.geometry`` (the real footprint — a notched swath stays
    notched) and falls back to ``item.bbox`` only when the geometry is
    missing. Rings that jump across the antimeridian and bboxes with
    ``west > east`` are split at ±180° rather than read as their
    complement.

    Raises:
        ValueError: The item has neither a geometry nor a usable bbox.
    """
    if item.geometry is not None:
        return _fix_antimeridian(shapely.geometry.shape(item.geometry))
    bbox = item.bbox
    if bbox is not None:
        if len(bbox) == 4:
            west, south, east, north = bbox
        elif len(bbox) == 6:
            west, south, east, north = bbox[0], bbox[1], bbox[3], bbox[4]
        else:
            raise ValueError(
                f"STAC item {item.id!r} bbox must have 4 or 6 values; got {bbox!r}"
            )
        return lonlat_box(west, south, east, north)
    raise ValueError(f"STAC item {item.id!r} has neither geometry nor bbox")


def _fix_antimeridian(
    geom: shapely.geometry.base.BaseGeometry,
) -> shapely.geometry.base.BaseGeometry:
    """Split every polygon / line part that jumps across ±180°.

    Footprints are 2D: a Z coordinate is dropped first (the
    antimeridian helpers work on lon/lat pairs).
    """
    geom = shapely.force_2d(geom)
    fixed, changed = _fix_parts(geom)
    if not changed:
        return geom
    if all(isinstance(g, shapely.Polygon) for g in fixed):
        return shapely.MultiPolygon(fixed) if len(fixed) > 1 else fixed[0]
    if all(isinstance(g, shapely.LineString) for g in fixed):
        return shapely.MultiLineString(fixed) if len(fixed) > 1 else fixed[0]
    return shapely.GeometryCollection(fixed)


def _fix_parts(
    geom: shapely.geometry.base.BaseGeometry,
) -> tuple[list[shapely.geometry.base.BaseGeometry], bool]:
    """Flattened simple parts of ``geom``, split at ±180°, and whether any was.

    Recurses into multi-part geometries and collections at any depth.
    """
    if hasattr(geom, "geoms"):
        fixed: list[shapely.geometry.base.BaseGeometry] = []
        changed = False
        for part in geom.geoms:
            sub, sub_changed = _fix_parts(part)
            fixed.extend(sub)
            changed = changed or sub_changed
        return fixed, changed
    out = geom
    if isinstance(geom, shapely.Polygon) and crosses_antimeridian(
        list(geom.exterior.coords)
    ):
        out = lonlat_polygon(
            list(geom.exterior.coords)[:-1],
            [list(ring.coords)[:-1] for ring in geom.interiors],
        )
    elif isinstance(geom, shapely.LineString) and crosses_antimeridian(
        list(geom.coords)
    ):
        out = lonlat_line(list(geom.coords))
    else:
        return [geom], False
    return list(getattr(out, "geoms", [out])), True


def item_interval(item: Any) -> pd.Interval:
    """The acquisition window as a closed UTC-aware interval.

    Per STAC 1.0 a ``start_datetime`` / ``end_datetime`` range describes
    the acquisition and ``datetime`` is a representative instant inside
    it, so the range wins when present; otherwise the interval is the
    zero-width ``datetime``. Naive values are taken as UTC.

    Raises:
        ValueError: The item has neither.
    """
    props = item.properties or {}
    start, end = props.get("start_datetime"), props.get("end_datetime")
    if start is not None and end is not None:
        return pd.Interval(to_utc_ts(start), to_utc_ts(end), closed="both")
    if item.datetime is not None:
        ts = to_utc_ts(item.datetime)
        return pd.Interval(ts, ts, closed="both")
    raise ValueError(
        f"STAC item {item.id!r} needs datetime or start_datetime/end_datetime"
    )


def asset_crs(
    item_properties: Mapping[str, Any], asset_fields: Mapping[str, Any]
) -> str | None:
    """The asset's native CRS from the projection extension.

    Any projection field on the asset overrides every one on the item —
    so an asset's v1 ``proj:epsg`` beats an item-level v2 ``proj:code``.
    Within one level ``proj:code`` (v2, e.g. ``"EPSG:32633"``) wins, then
    the v1 ``proj:epsg`` / ``proj:wkt2`` / ``proj:projjson``. Defaults
    to ``"EPSG:4326"`` when neither level has one.

    A projection field present but ``null`` is the extension's way of
    saying the asset (a thumbnail, say) has no CRS; it returns ``None``
    rather than falling back to the item's CRS.
    """
    for fields in (asset_fields, item_properties):
        crs = _proj_crs(fields)
        if crs is _UNLOCATED:
            return None
        if crs is not None:
            return crs
    return "EPSG:4326"


# `_proj_crs` result for fields that set a projection key to null.
_UNLOCATED: Any = object()


def _proj_crs(fields: Mapping[str, Any]) -> Any:
    """The CRS the fields name, `_UNLOCATED` for an explicit null, else None."""
    keys = ("proj:code", "proj:epsg", "proj:wkt2", "proj:projjson")
    if any(k in fields for k in keys) and all(fields.get(k) is None for k in keys):
        return _UNLOCATED
    code = fields.get("proj:code")
    if code:
        return str(code)
    epsg = fields.get("proj:epsg")
    if epsg is not None:
        return f"EPSG:{int(epsg)}"
    wkt2 = fields.get("proj:wkt2")
    if wkt2:
        return str(wkt2)
    projjson = fields.get("proj:projjson")
    if projjson:
        crs = pyproj.CRS.from_json_dict(projjson)
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg is not None else crs.to_wkt()
    return None


def asset_href(asset: Any) -> str:
    """The asset's href, made absolute against the item's self link.

    A relative href (``"./B04.tif"``) is only meaningful next to the
    item JSON; resolve it so the catalog row can be opened on its own.
    Falls back to the href as written when there is nothing to resolve
    it against.
    """
    absolute = (
        asset.get_absolute_href() if hasattr(asset, "get_absolute_href") else None
    )
    return str(absolute or asset.href)


# Query parameters that make up an Azure SAS token (Planetary Computer
# signs blob URLs this way; the signature expires after ~1 hour).
_SAS_KEYS = frozenset({"sig", "se"})


def is_signed_href(href: str) -> bool:
    """True when ``href`` carries an expiring Azure SAS signature."""
    query = urlparse(href).query
    return bool(query) and {k for k, _ in parse_qsl(query)} >= _SAS_KEYS


def reproject_geometry(
    geometry: shapely.geometry.base.BaseGeometry,
    src_crs: Any,
    dst_crs: Any,
    *,
    densify_pts: int = 64,
) -> shapely.geometry.base.BaseGeometry:
    """Reproject with edge densification.

    Straight edges in one CRS are curves in another; transforming only
    the vertices of a large footprint (a 15° x 10° tile into
    EPSG:3035) misplaces its edges by several percent of its area.
    Each edge is split into segments no longer than 1/``densify_pts``
    of its part's larger side before transforming.
    """
    src = pyproj.CRS.from_user_input(src_crs)
    dst = pyproj.CRS.from_user_input(dst_crs)
    if src.equals(dst):
        return geometry
    if densify_pts > 0:
        geometry = _densify_parts(geometry, densify_pts)
    transformer = pyproj.Transformer.from_crs(src, dst, always_xy=True)
    return shapely.ops.transform(transformer.transform, geometry)


def _densify_parts(
    geometry: shapely.geometry.base.BaseGeometry, densify_pts: int
) -> shapely.geometry.base.BaseGeometry:
    """Densify each simple part against its own span, recursing into nesting.

    Per part: a footprint split at ±180° spans ~360° as a whole, which
    would leave each narrow part barely densified — and a
    `GeometryCollection` may nest multi-part members.
    """
    parts = getattr(geometry, "geoms", None)
    if parts is None:
        return _densify(geometry, densify_pts)
    dense = [_densify_parts(p, densify_pts) for p in parts]
    if geometry.geom_type == "MultiPolygon":
        return shapely.MultiPolygon(dense)
    if geometry.geom_type == "MultiLineString":
        return shapely.MultiLineString(dense)
    if geometry.geom_type == "MultiPoint":
        return shapely.MultiPoint(dense)
    return shapely.GeometryCollection(dense)


def _densify(
    geometry: shapely.geometry.base.BaseGeometry, densify_pts: int
) -> shapely.geometry.base.BaseGeometry:
    xmin, ymin, xmax, ymax = geometry.bounds
    span = max(xmax - xmin, ymax - ymin)
    return shapely.segmentize(geometry, span / densify_pts) if span > 0 else geometry
