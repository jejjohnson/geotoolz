"""Atmospheric physics operators.

Layer-0 primitives are implemented in :mod:`xrtoolz.atm._src` and
re-exported here.

Content:

- Wind diagnostics: ``wind_speed``, ``wind_direction`` (meteorological
  "from" bearing by default) and their inverse ``wind_components``.
- Vertical-column diagnostics: ``column_integral`` (trapezoid or
  cell-width sum), ``hypsometric_height`` (pressure levels → geopotential
  height) and ``pbl_height_bulk_richardson``.
- Layer-1 ``Operator`` wrappers (:mod:`xrtoolz.atm.operators`), all
  re-exported here.

Trace-gas physics lives under :mod:`xrtoolz.atm.gas`. Potential
temperature is still planned (see ``docs/design/``).
"""

from xrtoolz.atm._src.vertical import (
    column_integral,
    hypsometric_height,
    pbl_height_bulk_richardson,
)
from xrtoolz.atm._src.wind import wind_components, wind_direction, wind_speed
from xrtoolz.atm.operators import (
    ColumnIntegral,
    HypsometricHeight,
    PBLHeightBulkRichardson,
    WindComponents,
    WindDirection,
    WindSpeed,
)


__all__ = [
    "ColumnIntegral",
    "HypsometricHeight",
    "PBLHeightBulkRichardson",
    "WindComponents",
    "WindDirection",
    "WindSpeed",
    "column_integral",
    "hypsometric_height",
    "pbl_height_bulk_richardson",
    "wind_components",
    "wind_direction",
    "wind_speed",
]
