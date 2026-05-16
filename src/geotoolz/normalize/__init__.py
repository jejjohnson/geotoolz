"""Normalization operators for remote-sensing GeoTensors."""

from __future__ import annotations

from geotoolz.normalize._src.array import (
    asinh_scale,
    histogram_match,
    log_scale,
    minmax_scale,
    per_band_stats,
    percentile_clip,
    power_scale,
    robust_scale,
    standard_scale,
)
from geotoolz.normalize._src.operators import (
    AsinhScale,
    HistogramMatch,
    HistogramStretch,
    LogScale,
    MinMaxScaler,
    Normalize,
    PerBandStats,
    PercentileClip,
    PowerScale,
    RobustScaler,
    StandardScaler,
    ZeroOne,
)


__all__ = [
    "AsinhScale",
    "HistogramMatch",
    "HistogramStretch",
    "LogScale",
    "MinMaxScaler",
    "Normalize",
    "PerBandStats",
    "PercentileClip",
    "PowerScale",
    "RobustScaler",
    "StandardScaler",
    "ZeroOne",
    "asinh_scale",
    "histogram_match",
    "log_scale",
    "minmax_scale",
    "per_band_stats",
    "percentile_clip",
    "power_scale",
    "robust_scale",
    "standard_scale",
]
