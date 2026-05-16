"""Band-space operators for spectral remote-sensing workflows."""

from __future__ import annotations

from geotoolz.spectral._src.array import (
    band_ratio,
    continuum_removal,
    evaluate_band_math,
    normalized_difference,
    reorder_bands,
    select_bands,
    spectral_binning,
    spectral_smoothing,
)
from geotoolz.spectral._src.operators import (
    ApplySRF,
    BandMath,
    BandRatio,
    ContinuumRemoval,
    GaussianSRF,
    NormalizedDifference,
    ReorderBands,
    SelectBands,
    SpectralBinning,
    SpectralSmoothing,
    SplitBands,
    StackBands,
)


__all__ = [
    "ApplySRF",
    "BandMath",
    "BandRatio",
    "ContinuumRemoval",
    "GaussianSRF",
    "NormalizedDifference",
    "ReorderBands",
    "SelectBands",
    "SpectralBinning",
    "SpectralSmoothing",
    "SplitBands",
    "StackBands",
    "band_ratio",
    "continuum_removal",
    "evaluate_band_math",
    "normalized_difference",
    "reorder_bands",
    "select_bands",
    "spectral_binning",
    "spectral_smoothing",
]
