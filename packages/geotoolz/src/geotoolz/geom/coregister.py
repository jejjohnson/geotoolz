"""`geotoolz.geom.coregister` — cross-modality coregistration operators.

Public re-exports of the operators implemented in
``geotoolz.geom._src.coregister``. Lives at ``geom.coregister`` rather
than the top level because these are genuine geometric operations —
the same module that owns ``Reproject`` / ``Resample`` / ``Rasterize``.

See ``docs/design/query-matchup.md`` §5.
"""

from __future__ import annotations

from geotoolz.geom._src.coregister import (
    GridToSwath,
    PointCloudToRaster,
    PointsToRaster,
    RasterToPointCloud,
    RasterToPoints,
    RasterToRasterLike,
    SwathToGrid,
    VectorToRasterAgg,
)


__all__ = [
    "GridToSwath",
    "PointCloudToRaster",
    "PointsToRaster",
    "RasterToPointCloud",
    "RasterToPoints",
    "RasterToRasterLike",
    "SwathToGrid",
    "VectorToRasterAgg",
]
