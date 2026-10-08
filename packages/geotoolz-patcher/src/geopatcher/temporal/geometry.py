"""`geopatcher.temporal.geometry` — The patch's extent along time.

`FixedLookback`, `LookbackHorizon` (past + forecast horizon),
`MultiScale`, `StencilGeometry` and `PhaseWindow`; `Geometry` is the base.
"""

from __future__ import annotations

from geopatcher._src.temporal.geometry import (
    FixedLookback,
    Geometry,
    LookbackHorizon,
    MultiScale,
    PhaseWindow,
    StencilGeometry,
)


__all__ = [
    "FixedLookback",
    "Geometry",
    "LookbackHorizon",
    "MultiScale",
    "PhaseWindow",
    "StencilGeometry",
]
