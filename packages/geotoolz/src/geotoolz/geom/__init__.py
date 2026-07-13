"""Geometry, projection, tiling, mosaicking, rasterization, and vectorization.

Cross-modality coregistration operators (raster↔raster grid alignment,
swath↔grid, raster↔points, point-cloud↔raster, vector-with-aggregation)
live in the ``coregister`` subnamespace — see
``docs/design/query-matchup.md`` §5 for the design.
"""

from __future__ import annotations

from geotoolz.geom import coregister  # noqa: F401 — re-export subnamespace
from geotoolz.geom._src.operators import (
    AntimeridianSplit,
    BowtieCorrection,
    CropTo,
    CropToBounds,
    Georeference,
    GeostationaryParallaxCorrect,
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


__all__ = [
    "AntimeridianSplit",
    "BowtieCorrection",
    "CropTo",
    "CropToBounds",
    "Georeference",
    "GeostationaryParallaxCorrect",
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
]
