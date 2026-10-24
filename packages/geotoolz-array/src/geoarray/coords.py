"""`geoarray.coords` — the standard coordinates of a carrier's dims.

- 1-D axes, stored in a carrier's ``coords``: `SpectralAxis` (``band``),
  `VariableAxis` (``variable``), `VerticalAxis` (``level``, with a
  `Vertical`) and `TimeAxis` (``time``).
- Grids, describing the spatial dims: `PixelGrid` (``y x``), `SwathGrid`
  and `CurvilinearGrid` (``row col``), `PointAxis` (``obs``).
- `Coordinate` — the protocol the 1-D axes satisfy.

Example:
    >>> from geoarray.coords import SpectralAxis
    >>> SpectralAxis(names=("B11", "B12"), wavelength_nm=(1610.0, 2190.0)).size
    2
"""

from __future__ import annotations

from geoarray._src.coords import (
    Coordinate,
    CurvilinearGrid,
    PixelGrid,
    PointAxis,
    SpectralAxis,
    SwathGrid,
    TimeAxis,
    VariableAxis,
    Vertical,
    VerticalAxis,
)


__all__ = [
    "Coordinate",
    "CurvilinearGrid",
    "PixelGrid",
    "PointAxis",
    "SpectralAxis",
    "SwathGrid",
    "TimeAxis",
    "VariableAxis",
    "Vertical",
    "VerticalAxis",
]
