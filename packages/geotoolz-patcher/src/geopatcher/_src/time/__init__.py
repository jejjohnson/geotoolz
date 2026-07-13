"""Temporal counterparts of the spatial four-axis Patcher.

The shape of the API mirrors `geopatcher.spatial` exactly —
`TemporalGeometry`, `TemporalSampler`, `TemporalWindow`,
`TemporalAggregation` bases with concrete axes that drop the
``Temporal`` prefix (the submodule path provides the context). See
``docs/patching.md`` §"The four temporal axes" for the time-axis
framing.
"""

from __future__ import annotations

from geopatcher._src.time.aggregation import (
    TemporalAggregation,
    TemporalFold,
    TemporalForecast,
    TemporalHierarchicalCombine,
    TemporalMean,
)
from geopatcher._src.time.geometry import (
    TemporalFixedLookback,
    TemporalGeometry,
    TemporalLookbackHorizon,
    TemporalMultiScale,
    TemporalPhaseWindow,
    TemporalStencilGeometry,
)
from geopatcher._src.time.patcher import TemporalPatcher
from geopatcher._src.time.sampler import (
    TemporalCausalRolling,
    TemporalEventTriggered,
    TemporalExplicit,
    TemporalRandom,
    TemporalRegularStride,
    TemporalSampler,
    TemporalStencilSampler,
)
from geopatcher._src.time.stencils import (
    Closed,
    Stencil,
    TimeStencil,
    build_sampling_slices,
    divide_evenly,
    valid_origin_points,
)
from geopatcher._src.time.window import (
    TemporalCausalBoxcar,
    TemporalExponentialDecay,
    TemporalPeriodic,
    TemporalTaperedTukey,
    TemporalWindow,
)


__all__ = [
    "Closed",
    "Stencil",
    "TemporalAggregation",
    "TemporalCausalBoxcar",
    "TemporalCausalRolling",
    "TemporalEventTriggered",
    "TemporalExplicit",
    "TemporalExponentialDecay",
    "TemporalFixedLookback",
    "TemporalFold",
    "TemporalForecast",
    "TemporalGeometry",
    "TemporalHierarchicalCombine",
    "TemporalLookbackHorizon",
    "TemporalMean",
    "TemporalMultiScale",
    "TemporalPatcher",
    "TemporalPeriodic",
    "TemporalPhaseWindow",
    "TemporalRandom",
    "TemporalRegularStride",
    "TemporalSampler",
    "TemporalStencilGeometry",
    "TemporalStencilSampler",
    "TemporalTaperedTukey",
    "TemporalWindow",
    "TimeStencil",
    "build_sampling_slices",
    "divide_evenly",
    "valid_origin_points",
]
