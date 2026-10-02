"""Lon/lat footprints that cross the antimeridian.

Upstream metadata (UMM-G bounding rectangles and polygons, STAC
geometries) describes a footprint that crosses 180° either as a
rectangle with ``west > east`` or as a ring whose consecutive vertices
jump by more than 180° of longitude. Read naively, both become the
*complement* of the footprint — a band spanning ~340° of longitude.
These helpers rebuild such footprints as a `MultiPolygon` split at
±180°, the form GeoJSON (RFC 7946 §3.1.9) prescribes.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import shapely
import shapely.affinity
import shapely.geometry
import shapely.geometry.base


def lonlat_box(
    west: float, south: float, east: float, north: float
) -> shapely.geometry.base.BaseGeometry:
    """A lon/lat rectangle; ``west > east`` means it crosses the antimeridian.

    Returns:
        A `Polygon`, or a two-part `MultiPolygon` split at ±180° when
        ``west > east``.
    """
    if west <= east:
        return shapely.box(west, south, east, north)
    # `west == 180` / `east == -180` leave a zero-width sliver on one
    # side of the seam; dropping it keeps the result a valid polygon.
    parts = [
        part
        for part in (
            shapely.box(west, south, 180.0, north),
            shapely.box(-180.0, south, east, north),
        )
        if part.area > 0
    ]
    if not parts:
        return shapely.Polygon()
    return shapely.MultiPolygon(parts) if len(parts) > 1 else parts[0]


def unwrap_longitudes(
    coords: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Shift longitudes by ±360° so no step between vertices exceeds 180°.

    The first vertex keeps its longitude; each later one is moved to
    whichever of ``lon``, ``lon ± 360`` is closest to its predecessor,
    so an antimeridian-crossing ring becomes a continuous ring that
    extends past ±180° instead of wrapping the long way round.
    """
    out: list[tuple[float, float]] = []
    for lon, lat in coords:
        if out:
            prev = out[-1][0]
            lon = lon + 360.0 * round((prev - lon) / 360.0)
        out.append((float(lon), float(lat)))
    return out


def crosses_antimeridian(coords: Sequence[tuple[float, float]]) -> bool:
    """True when consecutive vertices jump by more than 180° of longitude."""
    return any(abs(b[0] - a[0]) > 180.0 for a, b in itertools.pairwise(coords))


def wrap_to_lonlat(
    geom: shapely.geometry.base.BaseGeometry,
) -> shapely.geometry.base.BaseGeometry:
    """Fold a geometry that extends past ±180° back into [-180, 180].

    The parts east of 180° (or west of -180°) are cut off and shifted by
    ∓360°, so a continuous antimeridian-crossing footprint becomes a
    multi-part geometry that touches ±180° from both sides. A line that
    crosses several times in one direction can unwrap past ±540°, so the
    360°-wide windows are generated from the geometry's bounds. Pieces
    of lower dimension than the input (a polygon edge lying exactly on
    the seam) are dropped.
    """
    xmin, _, xmax, _ = geom.bounds
    if xmin >= -180.0 and xmax <= 180.0:
        return geom
    dim = int(shapely.get_dimensions(geom))
    flat = []
    for k in range(
        math.floor((xmin + 180.0) / 360.0), math.ceil((xmax - 180.0) / 360.0) + 1
    ):
        shift = 360.0 * k
        window = shapely.box(-180.0 + shift, -90.0, 180.0 + shift, 90.0)
        piece = geom.intersection(window)
        if piece.is_empty:
            continue
        piece = shapely.affinity.translate(piece, xoff=-shift)
        flat.extend(
            p
            for p in getattr(piece, "geoms", [piece])
            if not p.is_empty and int(shapely.get_dimensions(p)) == dim
        )
    polys = [p for p in flat if isinstance(p, shapely.Polygon)]
    if polys and len(polys) == len(flat):
        return shapely.MultiPolygon(polys) if len(polys) > 1 else polys[0]
    lines = [p for p in flat if isinstance(p, shapely.LineString)]
    if lines and len(lines) == len(flat):
        return shapely.MultiLineString(lines) if len(lines) > 1 else lines[0]
    return shapely.GeometryCollection(flat)


def lonlat_polygon(
    shell: Sequence[tuple[float, float]],
    holes: Sequence[Sequence[tuple[float, float]]] = (),
) -> shapely.geometry.base.BaseGeometry:
    """A lon/lat polygon whose rings may cross the antimeridian.

    Rings are unwrapped (`unwrap_longitudes`) before the polygon is
    built and the result folded back with `wrap_to_lonlat`, so the
    footprint is the small region the vertices enclose rather than its
    complement. Holes are unwrapped next to the shell (shifted by 360°
    when needed) so they stay inside it.
    """
    if not crosses_antimeridian([*shell, shell[0]]) and not any(
        crosses_antimeridian([*h, h[0]]) for h in holes
    ):
        poly = shapely.Polygon(shell, [list(h) for h in holes if len(h) >= 3])
        return poly if poly.is_valid else shapely.make_valid(poly)
    outer = unwrap_longitudes(shell)
    centre = (min(x for x, _ in outer) + max(x for x, _ in outer)) / 2.0
    inner = []
    for hole in holes:
        if len(hole) < 3:
            continue
        ring = unwrap_longitudes(hole)
        hx = sum(x for x, _ in ring) / len(ring)
        shift = 360.0 * round((centre - hx) / 360.0)
        inner.append([(x + shift, y) for x, y in ring])
    poly = shapely.Polygon(outer, inner)
    if not poly.is_valid:
        poly = shapely.make_valid(poly)
    return wrap_to_lonlat(poly)


def lonlat_line(
    coords: Sequence[tuple[float, float]],
) -> shapely.geometry.base.BaseGeometry:
    """A lon/lat line, split at ±180° when it crosses the antimeridian.

    A line lying entirely on the seam is returned on both edges
    (``+180`` and ``-180``), so it intersects queries spelled either way.
    """
    if all(abs(lon) == 180.0 for lon, _ in coords):
        east = shapely.LineString([(180.0, lat) for _, lat in coords])
        west = shapely.LineString([(-180.0, lat) for _, lat in coords])
        return shapely.MultiLineString([east, west])
    runs = _split_at_poles(coords)
    if len(runs) > 1:
        parts = [_lonlat_run(run) for run in runs if len(run) >= 2]
        flat = [g for part in parts for g in getattr(part, "geoms", [part])]
        return shapely.MultiLineString(flat)
    return _lonlat_run(coords)


def _lonlat_run(
    coords: Sequence[tuple[float, float]],
) -> shapely.geometry.base.BaseGeometry:
    if not crosses_antimeridian(coords):
        return shapely.LineString(coords)
    return wrap_to_lonlat(shapely.LineString(unwrap_longitudes(coords)))


def _split_at_poles(
    coords: Sequence[tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    """Split a vertex list where a segment passes over a pole.

    Two vertices exactly 180° of longitude apart are joined, on the
    sphere, by the meridian arc over the nearer pole — not by a parallel.
    Such a segment becomes two meridian legs, each ending at that pole.
    """
    runs: list[list[tuple[float, float]]] = [[coords[0]]]
    for a, b in itertools.pairwise(coords):
        dlon = abs(b[0] - a[0])
        if dlon == 180.0 and abs(a[1]) < 90.0 and abs(b[1]) < 90.0:
            pole = 90.0 if a[1] + b[1] >= 0 else -90.0
            runs[-1].append((a[0], pole))
            runs.append([(b[0], pole), b])
        else:
            runs[-1].append(b)
    return runs
