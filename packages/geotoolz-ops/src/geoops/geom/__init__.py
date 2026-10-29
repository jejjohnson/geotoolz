"""Geometry, projection, tiling, mosaicking, rasterization, and vectorization.

Cross-modality coregistration operators (raster↔raster grid alignment,
swath↔grid, raster↔points, point-cloud↔raster, vector-with-aggregation)
live in the ``coregister`` subnamespace (and at the top level) — see
``docs/catalog/design/query-matchup.md`` §5 for the design.
"""

from __future__ import annotations

from geoops.geom import coregister  # noqa: F401 — re-export subnamespace
from geoops.geom._src.array import (
    adaptive_curl,
    adaptive_difference,
    adaptive_divergence,
    adaptive_gradient,
    adaptive_laplacian,
    curl,
    difference,
    divergence,
    gradient,
    laplacian,
)
from geoops.geom._src.operators import (
    AntimeridianSplit,
    BowtieCorrection,
    CropTo,
    CropToBounds,
    Curl,
    Divergence,
    Georeference,
    GeostationaryParallaxCorrect,
    Gradient,
    Laplacian,
    Mosaic,
    OpticalFlowILK,
    OpticalFlowTVL1,
    PadTo,
    PhaseAlign,
    Rasterize,
    RasterizeLike,
    Reproject,
    ReprojectLike,
    Resample,
    ResampleLike,
    Resize,
    SegmentStitch,
    SlidingWindow,
    Stitch,
    Tile,
    Vectorize,
)
from geoops.geom._src.stencils import difference_coeffs, difference_offsets


__all__ = [
    "AntimeridianSplit",
    "BowtieCorrection",
    "CropTo",
    "CropToBounds",
    "Curl",
    "Divergence",
    "Georeference",
    "GeostationaryParallaxCorrect",
    "Gradient",
    "Laplacian",
    "Mosaic",
    "OpticalFlowILK",
    "OpticalFlowTVL1",
    "PadTo",
    "PhaseAlign",
    "Rasterize",
    "RasterizeLike",
    "Reproject",
    "ReprojectLike",
    "Resample",
    "ResampleLike",
    "Resize",
    "SegmentStitch",
    "SlidingWindow",
    "Stitch",
    "Tile",
    "Vectorize",
    "adaptive_curl",
    "adaptive_difference",
    "adaptive_divergence",
    "adaptive_gradient",
    "adaptive_laplacian",
    "curl",
    "difference",
    "difference_coeffs",
    "difference_offsets",
    "divergence",
    "gradient",
    "laplacian",
]
