"""Methane plume-complex outlines (GeoJSON) as a ragged xarray Dataset.

Reads the plume-complex GeoJSON of the EMIT L2B CH4 plume product
(``EMIT_L2B_CH4PLMMETA_<VVV>_<YYYYMMDDTHHMMSS>_<IIIIII>.json``, one feature
per file) and the multi-plume lists JPL's plume portal and SPIIRE work
from (one outline per plume). Positions are GeoJSON ``(lon, lat)`` on
WGS 84. The outlines are stored as a CF contiguous ragged array (CF 1.11,
section 9.3.3): the exterior rings, concatenated in plume order on a
``vertex`` dimension, with each plume's vertex count on ``plume``. Only the
standard library's :mod:`json` is needed.

The properties read, and where each spelling comes from:

- ``"Plume ID"``: the identifier (required).
- ``"UTC Time Observed"``: ISO-8601 time of the DAAC product.
- ``"Scene FIDs"`` (DAAC product) or ``"fids"`` (JPL plume list): the
  scene identifiers, a list.
- ``"Psuedo-Origin"`` (sic, JPL plume list): a string holding a GeoJSON
  point, the plume's source estimate. The DAAC product files do not carry
  it, so ``origin_lon`` / ``origin_lat`` are ``NaN`` for them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from xrproducts._src.local.errors import MissingPropertyError


_ID = "Plume ID"
_TIME = "UTC Time Observed"
_SCENES = ("Scene FIDs", "fids")
_ORIGIN = "Psuedo-Origin"
_POLYGONAL = ("Polygon", "MultiPolygon")


def open_ch4_plume_complexes(path: str | Path) -> xr.Dataset:
    """The plume-complex GeoJSON as a Dataset on ``plume``, with ragged outlines.

    One entry per ``Polygon`` / ``MultiPolygon`` feature; ``Point``
    features (the plume portal pairs each outline with its
    maximum-enhancement point) are skipped. A polygon keeps its exterior
    ring only. A multi-polygon keeps the polygon with the largest exterior
    ring (by area in degrees²) and the number of polygons dropped across
    the file is recorded in ``attrs["dropped_polygons"]``.

    Layout::

        Dimensions:   plume: P, vertex: V
        Coordinates:  plume (plume) str          "Plume ID", the index
                      plume_id (plume) str       the same identifiers
        Data vars:    time (plume) datetime64[ns]  "UTC Time Observed"
                      origin_lon, origin_lat (plume) float64  pseudo-origin
                      scene_ids (plume) str      scene identifiers, comma-joined
                      vertex_count (plume) int64  vertices of each exterior ring
                      outline_lon, outline_lat (vertex) float64
        Attributes:   crs "EPSG:4326", geometry_type "polygon",
                      dropped_polygons, origin_axis_order

    The pseudo-origin string is a GeoJSON point whose axis order varies
    between product versions (SPIIRE reads it as ``(lat, lon)`` for EMIT
    and ``(lon, lat)`` for AVIRIS-3). Each origin is read in the order that
    puts it inside, or nearest to, its outline's bounding box (``(lon,
    lat)`` on a tie), and ``attrs["origin_axis_order"]`` records
    ``"lon_lat"``, ``"lat_lon"`` or ``"mixed"`` (absent when no feature
    has an origin). Missing time is ``NaT``, missing scenes ``""``.

    Args:
        path: GeoJSON file with one feature per plume.

    Returns:
        The Dataset laid out above.

    Raises:
        MissingPropertyError: If a feature lacks its identifier or its
            geometry.
        ValueError: If a feature's geometry is neither a point nor a
            (multi-)polygon, or the file holds no outline.

    Example:
        >>> import json, tempfile, pathlib
        >>> ring = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
        >>> feature = {
        ...     "type": "Feature",
        ...     "geometry": {"type": "Polygon", "coordinates": [ring]},
        ...     "properties": {"Plume ID": "CH4_PlumeComplex-1"},
        ... }
        >>> doc = {"type": "FeatureCollection", "features": [feature]}
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "plumes.json"
        >>> _ = path.write_text(json.dumps(doc))
        >>> plumes = open_ch4_plume_complexes(path)
        >>> plumes.sizes["plume"], plumes.sizes["vertex"]
        (1, 4)
    """
    doc = json.loads(Path(path).read_text())
    rows: list[dict[str, Any]] = []
    dropped = 0
    for index, feature in enumerate(doc.get("features", ())):
        geometry = feature.get("geometry")
        properties = feature.get("properties") or {}
        if not geometry:
            raise MissingPropertyError(f"{path!s}: feature {index} has no geometry")
        kind = geometry.get("type")
        if kind == "Point":
            continue
        if kind not in _POLYGONAL:
            raise ValueError(
                f"{path!s}: feature {index} has a {kind!r} geometry; expected one of "
                f"{('Point', *_POLYGONAL)!r}"
            )
        if _ID not in properties:
            raise MissingPropertyError(f"{path!s}: feature {index} has no {_ID!r}")
        ring, n_dropped = _exterior_ring(geometry)
        dropped += n_dropped
        rows.append({"properties": properties, "ring": ring})
    if not rows:
        raise ValueError(f"{path!s} holds no Polygon or MultiPolygon feature")
    return _to_dataset(rows, dropped)


def plume_outline(plumes: xr.Dataset, plume_id: str) -> np.ndarray:
    """Exterior ring of one plume as ``(lon, lat)`` rows, shape ``(V_p, 2)``.

    Rows ``start_p : start_p + V_p`` of the vertex arrays, with
    ``start_p = ∑_{p' < p} V_p'`` and ``V_p`` the plume's vertex count.

    Args:
        plumes: Dataset from :func:`open_ch4_plume_complexes`.
        plume_id: The plume's ``"Plume ID"``.

    Returns:
        The ring, closed as stored in the GeoJSON (first row equals last).

    Raises:
        KeyError: If ``plume_id`` is not in the Dataset.

    Example:
        >>> import numpy as np, xarray as xr
        >>> plumes = xr.Dataset(
        ...     {
        ...         "vertex_count": ("plume", np.array([2, 3])),
        ...         "outline_lon": ("vertex", np.arange(5.0)),
        ...         "outline_lat": ("vertex", -np.arange(5.0)),
        ...     },
        ...     coords={"plume": ["a", "b"]},
        ... )
        >>> plume_outline(plumes, "b").tolist()
        [[2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]
    """
    ids = [str(v) for v in plumes["plume"].values]
    if plume_id not in ids:
        raise KeyError(f"plume {plume_id!r} is not in the Dataset")
    position = ids.index(plume_id)
    counts = np.asarray(plumes["vertex_count"].values, dtype=np.int64)
    start = int(counts[:position].sum())
    stop = start + int(counts[position])
    lon = np.asarray(plumes["outline_lon"].values[start:stop], dtype=np.float64)
    lat = np.asarray(plumes["outline_lat"].values[start:stop], dtype=np.float64)
    return np.column_stack([lon, lat])


def _exterior_ring(geometry: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    """Exterior ring ``(V, 2)`` of a (multi-)polygon and the polygons dropped."""
    if geometry["type"] == "Polygon":
        return _ring(geometry["coordinates"][0]), 0
    rings = [_ring(polygon[0]) for polygon in geometry["coordinates"]]
    areas = [_ring_area(ring) for ring in rings]
    return rings[int(np.argmax(areas))], len(rings) - 1


def _ring(positions: Sequence[Sequence[float]]) -> np.ndarray:
    """GeoJSON positions as ``(V, 2)`` float64 ``(lon, lat)`` (altitude dropped)."""
    return np.asarray([p[:2] for p in positions], dtype=np.float64).reshape(-1, 2)


def _ring_area(ring: np.ndarray) -> float:
    """Unsigned shoelace area of a ring, in degrees²."""
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _to_dataset(rows: list[dict[str, Any]], dropped: int) -> xr.Dataset:
    """The ragged Dataset from parsed rows."""
    ids = [str(row["properties"][_ID]) for row in rows]
    times = [_time(row["properties"].get(_TIME)) for row in rows]
    scenes = [_scene_ids(row["properties"]) for row in rows]
    origins = [_origin(row["properties"].get(_ORIGIN), row["ring"]) for row in rows]
    rings = [row["ring"] for row in rows]
    vertices = np.concatenate(rings, axis=0)
    orders = {order for *_, order in origins if order is not None}
    attrs: dict[str, Any] = {
        "crs": "EPSG:4326",
        "geometry_type": "polygon",
        "dropped_polygons": dropped,
    }
    if orders:
        attrs["origin_axis_order"] = orders.pop() if len(orders) == 1 else "mixed"
    deg_east = {"units": "degrees_east", "standard_name": "longitude"}
    deg_north = {"units": "degrees_north", "standard_name": "latitude"}
    return xr.Dataset(
        {
            "time": ("plume", np.array(times, dtype="datetime64[ns]")),
            "origin_lon": ("plume", np.array([o[0] for o in origins]), deg_east),
            "origin_lat": ("plume", np.array([o[1] for o in origins]), deg_north),
            "scene_ids": ("plume", np.array(scenes, dtype=str)),
            "vertex_count": (
                "plume",
                np.array([len(r) for r in rings], dtype=np.int64),
                {"sample_dimension": "vertex"},
            ),
            "outline_lon": ("vertex", vertices[:, 0], deg_east),
            "outline_lat": ("vertex", vertices[:, 1], deg_north),
        },
        coords={
            "plume": ("plume", np.array(ids, dtype=str), {"long_name": _ID}),
            "plume_id": ("plume", np.array(ids, dtype=str), {"long_name": _ID}),
        },
        attrs=attrs,
    )


def _time(value: object) -> np.datetime64:
    """A UTC time string as naive ``datetime64[ns]``; ``NaT`` when absent."""
    if value is None:
        return np.datetime64("NaT", "ns")
    ts = pd.Timestamp(str(value))
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_datetime64().astype("datetime64[ns]")


def _scene_ids(properties: Mapping[str, Any]) -> str:
    """Scene identifiers comma-joined; ``""`` when the feature has none."""
    value = next((properties[k] for k in _SCENES if k in properties), None)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return ",".join(str(v) for v in value)


def _origin(value: object, ring: np.ndarray) -> tuple[float, float, str | None]:
    """``(lon, lat, order)`` of a pseudo-origin; ``NaN``s and ``None`` if absent."""
    if value is None:
        return np.nan, np.nan, None
    point: Any = json.loads(value) if isinstance(value, str) else value
    first, second = (float(c) for c in point["coordinates"][:2])
    lon_min, lat_min = ring.min(axis=0)
    lon_max, lat_max = ring.max(axis=0)

    def distance(lon: float, lat: float) -> float:
        dx = max(lon_min - lon, 0.0, lon - lon_max)
        dy = max(lat_min - lat, 0.0, lat - lat_max)
        return float(np.hypot(dx, dy))

    if distance(second, first) < distance(first, second):
        return second, first, "lat_lon"
    return first, second, "lon_lat"
