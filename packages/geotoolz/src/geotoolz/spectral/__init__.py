"""Band-space operators for spectral remote-sensing workflows.

Band selection / reordering / stacking / splitting, band math and
ratios, spectral binning and smoothing, and continuum removal. The
normalized difference lives in :mod:`geotoolz.indices`
(:class:`~geotoolz.indices.NormalizedDifference`) and the Gaussian-SRF
convolution in :mod:`geotoolz.radiometry`
(:class:`~geotoolz.radiometry.ApplySRF`).
"""

from __future__ import annotations

from geotoolz.spectral._src.array import (
    band_ratio,
    continuum_removal,
    evaluate_band_math,
    reorder_bands,
    select_bands,
    spectral_binning,
    spectral_smoothing,
)
from geotoolz.spectral._src.operators import (
    BandMath,
    BandRatio,
    ContinuumRemoval,
    ReorderBands,
    SelectBands,
    SpectralBinning,
    SpectralSmoothing,
    SplitBands,
    StackBands,
)


__all__ = [
    "BandMath",
    "BandRatio",
    "ContinuumRemoval",
    "ReorderBands",
    "SelectBands",
    "SpectralBinning",
    "SpectralSmoothing",
    "SplitBands",
    "StackBands",
    "band_ratio",
    "continuum_removal",
    "evaluate_band_math",
    "reorder_bands",
    "select_bands",
    "spectral_binning",
    "spectral_smoothing",
]
