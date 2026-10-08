"""`geopatcher.spatial.geometry` — The patch's spatial extent and boundary policy.

`Rectangular` (an ``(h, w)`` pixel box), `SphericalCap` (a radius on the
sphere), the point-cloud neighbourhoods `KNNGraph` / `RadiusGraph`, and
`PolygonIntersection`; `Geometry` is the base for custom shapes.
"""

from __future__ import annotations

from geopatcher._src.spatial.geometry import (
    Geometry,
    KNNGraph,
    PolygonIntersection,
    RadiusGraph,
    Rectangular,
    SphericalCap,
)


__all__ = [
    "Geometry",
    "KNNGraph",
    "PolygonIntersection",
    "RadiusGraph",
    "Rectangular",
    "SphericalCap",
]
