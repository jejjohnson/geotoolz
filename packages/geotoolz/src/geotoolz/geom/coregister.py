"""`geotoolz.geom.coregister` — cross-modality coregistration operators.

Public re-exports of the operators implemented in
``geotoolz.geom._src.coregister``. The subnamespace lives under ``geom``
because these are genuine geometric operations — the same module that
owns ``Reproject`` / ``Resample`` / ``Rasterize``. Like every public
operator, each is also re-exported at the top level
(``gz.RasterToRasterLike``, ...).

See ``docs/design/query-matchup.md`` §5.
"""

from __future__ import annotations

from geotoolz.geom._src.coregister import (
    PointCloudToRaster,
    PointsToRaster,
    RasterToPointCloud,
    RasterToPoints,
    RasterToRasterLike,
    VectorToRasterAgg,
)


__all__ = [
    "PointCloudToRaster",
    "PointsToRaster",
    "RasterToPointCloud",
    "RasterToPoints",
    "RasterToRasterLike",
    "VectorToRasterAgg",
]
