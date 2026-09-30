"""Sparse feature, edge, texture, and Hough operators."""

from __future__ import annotations

from geotoolz.feature._src.array import (
    canny_edges,
    multiscale_features,
    structure_tensor_eigvals,
)
from geotoolz.feature._src.operators import (
    HOG,
    BlobDOG,
    BlobDoH,
    BlobLoG,
    Canny,
    CornerHarris,
    HoughCircles,
    HoughLines,
    MultiscaleBasicFeatures,
    PeakLocalMax,
    StructureTensor,
)


__all__ = [
    "HOG",
    "BlobDOG",
    "BlobDoH",
    "BlobLoG",
    "Canny",
    "CornerHarris",
    "HoughCircles",
    "HoughLines",
    "MultiscaleBasicFeatures",
    "PeakLocalMax",
    "StructureTensor",
    "canny_edges",
    "multiscale_features",
    "structure_tensor_eigvals",
]
