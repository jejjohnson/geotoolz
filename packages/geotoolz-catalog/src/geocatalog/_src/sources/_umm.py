"""Shared UMM-JSON parsing layer for the CMR and earthaccess adapters.

Both NASA-facing adapters (`geocatalog._src.sources.cmr` and
`geocatalog._src.sources.earthaccess`) consume the same CMR UMM
(Unified Metadata Model) granule schema — regardless of which client
fetched it. The pure helpers that decode a UMM granule dict into the
`SourceRow` building blocks (footprint geometry, observation interval,
asset keys, cloud cover, bounded metadata subset) live here so the two
adapters share one decoder.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import shapely
import shapely.geometry

from geocatalog._src._antimeridian import lonlat_box, lonlat_line, lonlat_polygon
from geocatalog._src._timeutil import (
    TIME_INVARIANT_END,
    TIME_INVARIANT_START,
    to_utc_ts,
)


def _lonlat_points(points: Any) -> list[tuple[float, float]]:
    """``[{"Longitude": x, "Latitude": y}, ...]`` → ``[(x, y), ...]``.

    Entries that are not mappings or miss a coordinate are skipped.
    """
    out: list[tuple[float, float]] = []
    for pt in points or ():
        if not isinstance(pt, Mapping):
            continue
        lon, lat = pt.get("Longitude"), pt.get("Latitude")
        if lon is None or lat is None:
            continue
        out.append((float(lon), float(lat)))
    return out


def _flatten(
    parts: list[shapely.geometry.base.BaseGeometry],
) -> shapely.geometry.base.BaseGeometry | None:
    """Combine footprint parts into one (multi-)geometry, or ``None``."""
    flat = [g for part in parts for g in getattr(part, "geoms", [part])]
    flat = [g for g in flat if not g.is_empty]
    if not flat:
        return None
    if len(flat) == 1:
        return flat[0]
    if all(isinstance(g, shapely.geometry.Polygon) for g in flat):
        return shapely.geometry.MultiPolygon(flat)
    if all(isinstance(g, shapely.geometry.LineString) for g in flat):
        return shapely.geometry.MultiLineString(flat)
    if all(isinstance(g, shapely.geometry.Point) for g in flat):
        return shapely.geometry.MultiPoint(flat)
    return shapely.geometry.GeometryCollection(flat)


def _planar_polygon(
    shell: list[tuple[float, float]], holes: list[list[tuple[float, float]]]
) -> shapely.geometry.base.BaseGeometry:
    poly = shapely.geometry.Polygon(shell, holes)
    return poly if poly.is_valid else shapely.make_valid(poly)


def granule_geometry(
    umm: Mapping[str, Any],
) -> shapely.geometry.base.BaseGeometry | None:
    """Extract a shapely footprint from the UMM SpatialExtent.

    UMM's spatial schema is a discriminated union of GPolygons,
    BoundingRectangles, Lines and Points. We accept the first one we
    recognise — most granules carry only one type.

    Footprints that cross the antimeridian — a rectangle with
    ``West > East`` or a polygon ring that jumps by more than 180° —
    are split at ±180° into a `MultiPolygon` rather than read as the
    complement band (#236) — unless the granule declares a CARTESIAN
    coordinate system, whose edges are straight in the lon/lat plane.
    A GPolygon's ``ExclusiveZone`` boundaries become holes.

    Args:
        umm: A UMM granule dict (the ``umm`` entry of a CMR item).

    Returns:
        A shapely geometry in lon/lat, or ``None`` when no usable
        footprint is present.
    """
    spatial = umm.get("SpatialExtent")
    if not isinstance(spatial, Mapping):
        return None
    h = spatial.get("HorizontalSpatialDomain")
    if not isinstance(h, Mapping):
        return None
    geom = h.get("Geometry")
    if not isinstance(geom, Mapping):
        return None
    # A CARTESIAN granule's edges are straight lines in the lon/lat
    # plane and never cross the antimeridian (CMR polygon support
    # notes), so a >180° edge is a real wide edge, not a crossing.
    cartesian = any(
        str(level.get(key) or "").upper() == "CARTESIAN"
        for level, key in (
            (geom, "CoordinateSystem"),
            (h, "CoordinateSystem"),
            (spatial, "GranuleSpatialRepresentation"),
        )
    )

    # GPolygons: list of polygons, each a Boundary with Points and an
    # optional ExclusiveZone of hole boundaries.
    polys: list[shapely.geometry.base.BaseGeometry] = []
    for poly in geom.get("GPolygons") or ():
        if not isinstance(poly, Mapping):
            continue
        boundary = poly.get("Boundary")
        shell = _lonlat_points(
            boundary.get("Points") if isinstance(boundary, Mapping) else None
        )
        if len(shell) < 3:
            continue
        zone = poly.get("ExclusiveZone")
        holes = [
            ring
            for b in (zone.get("Boundaries") or () if isinstance(zone, Mapping) else ())
            if isinstance(b, Mapping)
            and len(ring := _lonlat_points(b.get("Points"))) >= 3
        ]
        polys.append(
            _planar_polygon(shell, holes) if cartesian else lonlat_polygon(shell, holes)
        )
    if (found := _flatten(polys)) is not None:
        return found

    # BoundingRectangles: WGS84 W/S/E/N quads; West > East crosses 180°.
    boxes: list[shapely.geometry.base.BaseGeometry] = []
    for r in geom.get("BoundingRectangles") or ():
        if not isinstance(r, Mapping):
            continue
        try:
            boxes.append(
                lonlat_box(
                    float(r["WestBoundingCoordinate"]),
                    float(r["SouthBoundingCoordinate"]),
                    float(r["EastBoundingCoordinate"]),
                    float(r["NorthBoundingCoordinate"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if (found := _flatten(boxes)) is not None:
        return found

    # Lines (along-track granules, e.g. altimetry ground tracks).
    lines: list[shapely.geometry.base.BaseGeometry] = []
    for line in geom.get("Lines") or ():
        if not isinstance(line, Mapping):
            continue
        coords = _lonlat_points(line.get("Points"))
        if len(coords) >= 2:
            lines.append(
                shapely.geometry.LineString(coords)
                if cartesian
                else lonlat_line(coords)
            )
    if (found := _flatten(lines)) is not None:
        return found

    # Points (rare for raster granules but possible for vector
    # collections — e.g. AERONET station data).
    points = [shapely.geometry.Point(c) for c in _lonlat_points(geom.get("Points"))]
    return _flatten(list(points))


def granule_interval(umm: Mapping[str, Any]) -> pd.Interval | None:
    """Build a `pd.Interval` from the UMM TemporalExtent.

    UMM supports ``RangeDateTime`` (start + end) and ``SingleDateTime``
    (instantaneous). Both normalise to a UTC closed-both interval. An
    open-ended range — ``EndingDateTime`` missing (ongoing) or
    ``BeginningDateTime`` missing — is closed with the catalog's
    time-invariant sentinel (`TIME_INVARIANT_END` / `_START`) rather
    than dropping the granule or producing a ``NaT`` endpoint; an
    endpoint beyond the sentinel keeps the range pointing forward. A
    *malformed* endpoint (present but unparseable) is not read as open:
    the range is ignored and ``SingleDateTime`` is used if present.

    Args:
        umm: A UMM granule dict.

    Returns:
        A ``closed="both"`` interval of UTC-aware timestamps, or
        ``None`` when the granule carries no usable temporal extent.
    """
    temporal = umm.get("TemporalExtent")
    if not isinstance(temporal, Mapping):
        return None
    rng = temporal.get("RangeDateTime")
    if isinstance(rng, Mapping):
        start = _parse_time(rng.get("BeginningDateTime"))
        end = _parse_time(rng.get("EndingDateTime"))
        malformed = start is _MALFORMED or end is _MALFORMED
        if not malformed and (start is not None or end is not None):
            if start is None:
                start = min(_OPEN_START, end)
            elif end is None:
                end = max(_OPEN_END, start)
            elif end < start:
                start, end = end, start
            return pd.Interval(start, end, closed="both")
    single = _parse_time(temporal.get("SingleDateTime"))
    if isinstance(single, pd.Timestamp):
        return pd.Interval(single, single, closed="both")
    return None


_OPEN_START = TIME_INVARIANT_START.tz_localize("UTC")
_OPEN_END = TIME_INVARIANT_END.tz_localize("UTC")


# Returned by `_parse_time` for a value that is present but unparseable,
# so a corrupt endpoint is never mistaken for an open (missing) one.
_MALFORMED: Any = object()


def _parse_time(value: Any) -> Any:
    """A UTC timestamp; ``None`` if missing / empty; `_MALFORMED` if unparseable."""
    if value is None or value == "":
        return None
    try:
        ts = to_utc(value)
    except (TypeError, ValueError):
        return _MALFORMED
    return _MALFORMED if ts is pd.NaT else ts


def to_utc(value: str | datetime) -> pd.Timestamp:
    """Coerce any datetime-like to a UTC-aware `pd.Timestamp`.

    Thin re-export of `geocatalog._src._timeutil.to_utc_ts`, kept so
    UMM decoding reads self-contained at the call sites.

    Args:
        value: An ISO string or ``datetime`` from a UMM record.

    Returns:
        A tz-aware ``pd.Timestamp`` in UTC.
    """
    return to_utc_ts(value)


def asset_key_from_url(url: str) -> str:
    """Pick a short, readable asset key from a download URL.

    Heuristic: the filename's stem (last path segment minus
    extension). Falls back to the extension or a hash-truncated
    last segment if the path is degenerate.

    Args:
        url: The granule download URL.

    Returns:
        A non-empty key suitable for a STAC-shaped asset map.
    """
    parsed = urlparse(url)
    leaf = (parsed.path.rstrip("/").rsplit("/", 1) or [""])[-1]
    if "." in leaf:
        stem, ext = leaf.rsplit(".", 1)
        # Prefer the extension for keys when the stem is just the
        # granule UR repeated (common on opendap/data URLs).
        if stem and len(stem) <= 64:
            return stem
        return ext or leaf
    return leaf or url[-32:]


def extract_cloud_cover(umm: Mapping[str, Any]) -> float | None:
    """Pull cloud cover percentage out of UMM, if present.

    Stored under ``AdditionalAttributes`` for some sensors and
    under ``CloudCover`` directly for others. We try both.

    Args:
        umm: A UMM granule dict.

    Returns:
        Cloud cover as a float, or ``None`` when absent/unparseable.
    """
    if "CloudCover" in umm:
        try:
            return float(umm["CloudCover"])
        except (TypeError, ValueError):
            pass
    extras = umm.get("AdditionalAttributes", [])
    if isinstance(extras, list):
        for attr in extras:
            # `Name` may be present but null in real UMM records.
            name = attr.get("Name") if isinstance(attr, Mapping) else None
            if isinstance(attr, Mapping) and str(name or "").lower().startswith(
                "cloud"
            ):
                values = attr.get("Values", [])
                if values:
                    try:
                        return float(values[0])
                    except (TypeError, ValueError):
                        pass
    return None


def umm_essentials(umm: Mapping[str, Any]) -> dict[str, Any]:
    """A bounded copy of UMM for `SourceRow.properties`.

    Full UMM dicts can be several KB per granule; we keep the
    fields most users actually inspect and drop the rest. Anyone
    who wants the full UMM can re-query the originating service.

    Args:
        umm: A UMM granule dict.

    Returns:
        A dict containing only the retained top-level UMM keys.
    """
    keep = (
        "GranuleUR",
        "CollectionReference",
        "DataGranule",
        "TemporalExtent",
        "ProviderDates",
        "Platforms",
        "AdditionalAttributes",
        "CloudCover",
        "MetadataSpecification",
    )
    return {k: umm[k] for k in keep if k in umm}
