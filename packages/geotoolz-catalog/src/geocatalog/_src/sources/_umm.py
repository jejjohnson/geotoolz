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
import shapely.geometry

from geocatalog._src._timeutil import to_utc_ts


def granule_geometry(
    umm: Mapping[str, Any],
) -> shapely.geometry.base.BaseGeometry | None:
    """Extract a shapely footprint from the UMM SpatialExtent.

    UMM's spatial schema is a discriminated union of GPolygons,
    BoundingRectangles, Points, and (rarely) Lines. We accept the
    first one we recognise — most granules carry only one type.

    Args:
        umm: A UMM granule dict (the ``umm`` entry of a CMR item).

    Returns:
        A shapely geometry, or ``None`` when no usable footprint is
        present.
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

    # GPolygons: list of polygons, each a Boundary with Points.
    gpolys = geom.get("GPolygons")
    if gpolys:
        polys = []
        for poly in gpolys:
            boundary = poly.get("Boundary", {}) if isinstance(poly, Mapping) else {}
            points = boundary.get("Points") if isinstance(boundary, Mapping) else None
            if not points:
                continue
            ring: list[tuple[float, float]] = []
            for pt in points:
                if not isinstance(pt, Mapping):
                    continue
                lon = pt.get("Longitude")
                lat = pt.get("Latitude")
                if lon is None or lat is None:
                    continue
                ring.append((lon, lat))
            if len(ring) >= 3:
                polys.append(shapely.geometry.Polygon(ring))
        if polys:
            return shapely.geometry.MultiPolygon(polys) if len(polys) > 1 else polys[0]

    # BoundingRectangles: list of WGS84 N/S/E/W coordinate quads.
    rects = geom.get("BoundingRectangles")
    if rects:
        boxes = []
        for r in rects:
            if not isinstance(r, Mapping):
                continue
            try:
                box = shapely.geometry.box(
                    r["WestBoundingCoordinate"],
                    r["SouthBoundingCoordinate"],
                    r["EastBoundingCoordinate"],
                    r["NorthBoundingCoordinate"],
                )
            except KeyError:
                continue
            boxes.append(box)
        if boxes:
            return shapely.geometry.MultiPolygon(boxes) if len(boxes) > 1 else boxes[0]

    # Points (rare for raster granules but possible for vector
    # collections — e.g. AERONET station data).
    points = geom.get("Points")
    if points:
        shapes = []
        for pt in points:
            if not isinstance(pt, Mapping):
                continue
            lon = pt.get("Longitude")
            lat = pt.get("Latitude")
            if lon is None or lat is None:
                continue
            shapes.append(shapely.geometry.Point(lon, lat))
        if shapes:
            return shapely.geometry.MultiPoint(shapes) if len(shapes) > 1 else shapes[0]
    return None


def granule_interval(umm: Mapping[str, Any]) -> pd.Interval | None:
    """Build a `pd.Interval` from the UMM TemporalExtent.

    UMM supports two forms: ``RangeDateTime`` (start + end) and
    ``SingleDateTime`` (instantaneous). We normalise both to a UTC
    closed-both interval.

    Args:
        umm: A UMM granule dict.

    Returns:
        A ``closed="both"`` interval of UTC-aware timestamps, or
        ``None`` when the granule carries no temporal extent.
    """
    temporal = umm.get("TemporalExtent")
    if not isinstance(temporal, Mapping):
        return None
    rng = temporal.get("RangeDateTime")
    if isinstance(rng, Mapping):
        start = rng.get("BeginningDateTime")
        end = rng.get("EndingDateTime")
        if start and end:
            return pd.Interval(to_utc(start), to_utc(end), closed="both")
    single = temporal.get("SingleDateTime")
    if single:
        ts = to_utc(single)
        return pd.Interval(ts, ts, closed="both")
    return None


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
            if isinstance(attr, Mapping) and attr.get("Name", "").lower().startswith(
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
