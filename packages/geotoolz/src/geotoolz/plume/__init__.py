"""Trace-gas plume detection, footprints, and emission estimators."""

from __future__ import annotations

from geotoolz.plume._src.array import (
    convert_column_units,
    plume_mask,
    wind_advection_cone,
)
from geotoolz.plume._src.operators import (
    SBMP,
    ColumnToMass,
    CrossSectionalFlux,
    IMEEstimate,
    PlumeColumnStats,
    PlumeContours,
    PlumeFootprint,
    PlumeMask,
    PlumeQNDFeatures,
    PlumeShapeFilter,
    WindAdvectionCone,
)


__all__ = [
    "SBMP",
    "ColumnToMass",
    "CrossSectionalFlux",
    "IMEEstimate",
    "PlumeColumnStats",
    "PlumeContours",
    "PlumeFootprint",
    "PlumeMask",
    "PlumeQNDFeatures",
    "PlumeShapeFilter",
    "WindAdvectionCone",
    "convert_column_units",
    "plume_mask",
    "wind_advection_cone",
]
