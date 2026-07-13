"""Concrete `Domain` types — one per data geometry.

Each `Domain` carries just enough metadata for `Sampler.anchors` and
`PatchGeometry.neighborhood` to do their work without touching the
underlying data. `RasterDomain` is reused from `georeader.GeoDataBase`
(every `RasterioReader` / `GeoTensor` already satisfies it); the other
three are introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

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
        interp: Interpolation used by `sample` when extracting raster
            values at this domain's points — ``"nearest"`` (containing
            pixel, the historical behaviour) or ``"bilinear"``
            (interpolate between the four surrounding pixel centres).
    """

    coords: np.ndarray
    kdtree: Any
    crs: Any | None = None
    interp: str = "nearest"

    _VALID_INTERP: ClassVar[tuple[str, ...]] = ("nearest", "bilinear")

    def __post_init__(self) -> None:
        if self.interp not in self._VALID_INTERP:
            raise ValueError(
                f"invalid interp {self.interp!r}; expected one of {self._VALID_INTERP}"
            )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        mins = self.coords.min(axis=0)
        maxs = self.coords.max(axis=0)
        return float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])

    def sample(self, raster: Any) -> np.ndarray:
        """Extract raster values at this domain's points.

        Uses the domain's `interp` mode: ``"nearest"`` reads the pixel
        containing each point, ``"bilinear"`` interpolates between the
        four surrounding pixel centres (clamped at the raster edge).
        Points falling outside the raster extent return ``NaN``.

        Args:
            raster: A raster-shaped object — anything carrying an affine
                ``transform`` plus pixel values under ``.values`` (a
                `georeader.GeoTensor`) or as the object itself. The
                raster is assumed to share this domain's CRS.

        Returns:
            ``(N,)`` float array of sampled values for 2-D rasters, or
            ``(..., N)`` for rasters with leading band dims.
        """
        values = np.asarray(getattr(raster, "values", raster), dtype=float)
        h, w = values.shape[-2], values.shape[-1]
        inv = ~raster.transform
        xs = self.coords[:, 0].astype(float)
        ys = self.coords[:, 1].astype(float)
        # Fractional pixel coords, measured from the raster's UL corner.
        cols = inv.a * xs + inv.b * ys + inv.c
        rows = inv.d * xs + inv.e * ys + inv.f
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out_shape = (*values.shape[:-2], len(self.coords))
        out = np.full(out_shape, np.nan, dtype=float)
        if not inside.any():
            return out
        if self.interp == "nearest":
            r = np.floor(rows[inside]).astype(int)
            c = np.floor(cols[inside]).astype(int)
            out[..., inside] = values[..., r, c]
            return out
        # Bilinear between pixel centres (which sit at +0.5 in pixel space).
        u = cols[inside] - 0.5
        v = rows[inside] - 0.5
        c0 = np.floor(u).astype(int)
        r0 = np.floor(v).astype(int)
        wu = u - c0
        wv = v - r0
        c0c = np.clip(c0, 0, w - 1)
        c1c = np.clip(c0 + 1, 0, w - 1)
        r0c = np.clip(r0, 0, h - 1)
        r1c = np.clip(r0 + 1, 0, h - 1)
        out[..., inside] = (
            values[..., r0c, c0c] * (1 - wv) * (1 - wu)
            + values[..., r0c, c1c] * (1 - wv) * wu
            + values[..., r1c, c0c] * wv * (1 - wu)
            + values[..., r1c, c1c] * wv * wu
        )
        return out
