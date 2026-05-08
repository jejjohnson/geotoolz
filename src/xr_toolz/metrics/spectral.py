"""Spectral evaluation metrics — public re-export.

Layer-0 functions: :func:`psd_error`, :func:`psd_score`,
:func:`along_track_psd_score`, :func:`psd_score_by_region`,
:func:`resolved_scale`, :func:`find_intercept_1D`,
:func:`evaluate_by_frequency_band`, :func:`band_limited_rmse`.

Layer-1 operators: :class:`PSDScore`, :class:`FrequencyBandSkill`,
:class:`BandLimitedRMSE`, :class:`SegmentedPSDScore`.

Implementation lives in :mod:`xr_toolz.metrics._src.spectral` for the
core PSD primitives, with the gap-tolerant segmented along-track
helpers (:func:`along_track_psd_score`, :func:`psd_score_by_region`,
:class:`SegmentedPSDScore`) defined in
:mod:`xr_toolz.metrics._src.segmented_psd`.
"""

from xr_toolz.metrics._src.segmented_psd import (
    SegmentedPSDScore,
    along_track_psd_score,
    psd_score_by_region,
)
from xr_toolz.metrics._src.spectral import (
    BandLimitedRMSE,
    FrequencyBandSkill,
    PSDScore,
    band_limited_rmse,
    evaluate_by_frequency_band,
    find_intercept_1D,
    psd_error,
    psd_score,
    resolved_scale,
)


__all__ = [
    "BandLimitedRMSE",
    "FrequencyBandSkill",
    "PSDScore",
    "SegmentedPSDScore",
    "along_track_psd_score",
    "band_limited_rmse",
    "evaluate_by_frequency_band",
    "find_intercept_1D",
    "psd_error",
    "psd_score",
    "psd_score_by_region",
    "resolved_scale",
]
