"""Spectral evaluation metrics — public re-export.

Layer-0 functions: :func:`psd_error`, :func:`psd_score`,
:func:`psd_score_spacetime`, :func:`along_track_psd_score`,
:func:`psd_score_by_region`, :func:`resolved_scale`,
:func:`resolved_scale_2d`, :func:`find_intercept_1D`,
:func:`evaluate_by_frequency_band`, :func:`band_limited_rmse`,
:func:`wavelet_psd_score`, :func:`wavelet_resolved_scale_map`.

Layer-1 operators: :class:`PSDScore`, :class:`FrequencyBandSkill`,
:class:`BandLimitedRMSE`, :class:`SegmentedPSDScore`.

Implementation lives in :mod:`xrtoolz.metrics._src.spectral` for the
core PSD primitives, with the gap-tolerant segmented along-track
helpers (:func:`along_track_psd_score`, :func:`psd_score_by_region`,
:class:`SegmentedPSDScore`) defined in
:mod:`xrtoolz.metrics._src.segmented_psd` and the 2-D space-time
helpers (:func:`psd_score_spacetime`) in
:mod:`xrtoolz.metrics._src.composite`.
"""

from xrtoolz.metrics._src.composite import psd_score_spacetime
from xrtoolz.metrics._src.segmented_psd import (
    SegmentedPSDScore,
    along_track_psd_score,
    psd_score_by_region,
)
from xrtoolz.metrics._src.spectral import (
    BandLimitedRMSE,
    FrequencyBandSkill,
    PSDScore,
    WaveletPSDScore,
    band_limited_rmse,
    evaluate_by_frequency_band,
    find_intercept_1D,
    psd_error,
    psd_score,
    resolved_scale,
    resolved_scale_2d,
    wavelet_psd_score,
    wavelet_resolved_scale_map,
)


__all__ = [
    "BandLimitedRMSE",
    "FrequencyBandSkill",
    "PSDScore",
    "SegmentedPSDScore",
    "WaveletPSDScore",
    "along_track_psd_score",
    "band_limited_rmse",
    "evaluate_by_frequency_band",
    "find_intercept_1D",
    "psd_error",
    "psd_score",
    "psd_score_by_region",
    "psd_score_spacetime",
    "resolved_scale",
    "resolved_scale_2d",
    "wavelet_psd_score",
    "wavelet_resolved_scale_map",
]
