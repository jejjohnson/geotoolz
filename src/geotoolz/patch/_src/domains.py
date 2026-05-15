"""Concrete `Domain` types — one per data geometry.

Each `Domain` carries just enough metadata for `Sampler.anchors` and
`PatchGeometry.neighborhood` to do their work without touching the
underlying data. `RasterDomain` is reused from `georeader.GeoDataBase`
(every `RasterioReader` / `GeoTensor` already satisfies it); the other
three are introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# RasterDomain is the existing `GeoDataBase` Protocol from georeader.
# Re-exported here under the geopatcher name so user code can spell it
# uniformly with the new Grid/Vector/Point domains.
from georeader.abstract_reader import GeoDataBase as RasterDomain  # noqa: F401


@dataclass(eq=False)
class GridDomain:
    """A dense, labeled N-D grid — the `xarray.DataArray` shape.

    Args:
        coords: ``{dim_name: 1-D coordinate array}``. The order of keys
            defines the canonical dim order used by `Rectangular`
            samplers on this domain.
        crs: CRS of the spatial dims (lat/lon, x/y). Optional —
            non-georeferenced cubes are fine.

    Attributes:
        shape: ``tuple(len(coords[d]) for d in coords)``.
        bounds: ``(min, max)`` per dim — derived lazily on access.
    """

    coords: dict[str, np.ndarray]
    crs: Any | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(self.coords[d]) for d in self.coords)

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return {d: (float(np.min(c)), float(np.max(c))) for d, c in self.coords.items()}


@dataclass(eq=False)
class VectorDomain:
    """Vector geometries — polygons / lines indexed by an STRtree.

    Args:
        geometry: A `geopandas.GeoSeries` of shapely geometries.
        sindex: The `geometry.sindex` object — kept on the domain so
            spatial-index queries don't trigger a re-build.
        crs: The series' CRS.
    """

    geometry: Any
    sindex: Any
    crs: Any

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        b = self.geometry.total_bounds
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])


@dataclass(eq=False)
class PointDomain:
    """Scattered points — coordinates + a `cKDTree` for nearest-neighbor lookups.

    Args:
        coords: ``(N, 2)`` array of point coordinates in the domain's CRS.
        kdtree: A `scipy.spatial.cKDTree` over ``coords``. Built once at
            domain-construction and reused by `KNNGraph` / `RadiusGraph`.
        crs: Coordinate reference system, or ``None`` for arbitrary
            in-plane coordinates.
    """

    coords: np.ndarray
    kdtree: Any
    crs: Any | None = None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        mins = self.coords.min(axis=0)
        maxs = self.coords.max(axis=0)
        return float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])
