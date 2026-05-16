"""Trace-gas plume detection, footprints, and emission estimators."""

from __future__ import annotations

from geotoolz.plume._src.array import (
    convert_column_units,
    label_components,
    otsu_threshold,
    plume_mask,
    resolve_threshold,
    wind_advection_cone,
)
from geotoolz.plume._src.operators import (
    SBMP,
    ColumnToMass,
    CrossSectionalFlux,
    IMEEstimate,
    PlumeContours,
    PlumeFootprint,
    PlumeMask,
    WindAdvectionCone,
)


__all__ = [
    "SBMP",
    "ColumnToMass",
    "CrossSectionalFlux",
    "IMEEstimate",
    "PlumeContours",
    "PlumeFootprint",
    "PlumeMask",
    "WindAdvectionCone",
    "convert_column_units",
    "label_components",
    "otsu_threshold",
    "plume_mask",
    "resolve_threshold",
    "wind_advection_cone",
]
