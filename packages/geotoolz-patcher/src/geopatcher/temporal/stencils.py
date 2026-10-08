"""`geopatcher.temporal.stencils` — time stencils and their slicing helpers.

A `Stencil` describes a patch in time as offsets around an origin
(``Stencil(start, stop, step)`` in index units; `TimeStencil` in time
units such as ``"-9h"``). `temporal.geometry.StencilGeometry` and
`temporal.sampler.StencilSampler` use them; the helpers turn a stencil and a
time coordinate into index slices.
"""

from __future__ import annotations

from geopatcher._src.temporal.stencils import (
    Closed,
    Stencil,
    TimeStencil,
    build_sampling_slices,
    coord_step,
    exact_quotient,
    stencil_offsets,
    valid_origin_points,
)


__all__ = [
    "Closed",
    "Stencil",
    "TimeStencil",
    "build_sampling_slices",
    "coord_step",
    "exact_quotient",
    "stencil_offsets",
    "valid_origin_points",
]
