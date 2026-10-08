"""`geopatcher.temporal.sampler` — Where time anchors go.

`RegularStride` (a rolling walk forward), `Random`, `Explicit`
(event-aligned indices) and `StencilSampler` (every origin whose
`stencils.Stencil` fits); `Sampler` is the base.
"""

from __future__ import annotations

from geopatcher._src.temporal.sampler import (
    Explicit,
    Random,
    RegularStride,
    Sampler,
    StencilSampler,
)


__all__ = [
    "Explicit",
    "Random",
    "RegularStride",
    "Sampler",
    "StencilSampler",
]
