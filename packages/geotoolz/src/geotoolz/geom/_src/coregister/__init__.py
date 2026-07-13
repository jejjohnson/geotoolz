"""Cross-modality coregistration primitives + operators.

Single-source geometric ops (reproject / resample / rasterize /
phase-align / optical-flow) already live in
``geotoolz.geom._src.operators``. The operators in this subpackage
are their *multi-input cousins* — each takes two or more inputs of
potentially different modalities (LEO raster, GEO grid, vector,
point cloud, vector-cube of stations) and emits one aligned tensor.

Used downstream by ``geopatcher.matched.MatchedField`` so a single
sampler can read co-located neighborhoods across heterogeneous
sources without geopatcher knowing how the alignment was done.

See ``docs/design/query-matchup.md`` §5 for the full design.
"""

from __future__ import annotations

from geotoolz.geom._src.coregister.operators import (
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
