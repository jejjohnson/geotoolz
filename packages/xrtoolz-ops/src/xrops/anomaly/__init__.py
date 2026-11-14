"""Anomalies — climatologies, mean removal, and anomaly calculation.

Layer-1 ``Operator`` wrappers live in :mod:`xrops.anomaly.operators`.
"""

from __future__ import annotations

from xrops.anomaly._src.detrend import (
    add_climatology,
    calculate_anomaly,
    calculate_anomaly_smoothed,
    calculate_climatology,
    calculate_climatology_season,
    calculate_climatology_smoothed,
    remove_climatology,
    remove_mean,
)
from xrops.anomaly.operators import (
    AddClimatology,
    CalculateClimatology,
    CalculateClimatologySmoothed,
    RemoveClimatology,
    RemoveMean,
)


__all__ = [
    "AddClimatology",
    "CalculateClimatology",
    "CalculateClimatologySmoothed",
    "RemoveClimatology",
    "RemoveMean",
    "add_climatology",
    "calculate_anomaly",
    "calculate_anomaly_smoothed",
    "calculate_climatology",
    "calculate_climatology_season",
    "calculate_climatology_smoothed",
    "remove_climatology",
    "remove_mean",
]
