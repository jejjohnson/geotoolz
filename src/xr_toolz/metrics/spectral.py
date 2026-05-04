"""Spectral evaluation metrics — public re-export.

Layer-0 functions: :func:`psd_error`, :func:`psd_score`,
:func:`resolved_scale`, :func:`find_intercept_1D`,
:func:`evaluate_by_frequency_band`, :func:`band_limited_rmse`.

Layer-1 operators: :class:`PSDScore`, :class:`FrequencyBandSkill`,
:class:`BandLimitedRMSE`.

Implementation lives in :mod:`xr_toolz.metrics._src.spectral`.
"""

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
    "band_limited_rmse",
    "evaluate_by_frequency_band",
    "find_intercept_1D",
    "psd_error",
    "psd_score",
    "resolved_scale",
]
