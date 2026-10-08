"""`geopatcher.temporal` — the four axes of a `TemporalPatcher`, plus stencils.

The time-axis mirror of `geopatcher.spatial`, one module per axis:

- `geopatcher.temporal.geometry` — the extent along time (`FixedLookback`, …).
- `geopatcher.temporal.sampler` — where time anchors go (`RegularStride`, …).
- `geopatcher.temporal.window` — per-step weights (`CausalBoxcar`, …).
- `geopatcher.temporal.aggregation` — how outputs combine (`Mean`, …).
- `geopatcher.temporal.stencils` — `Stencil` / `TimeStencil` and the
  slicing helpers for irregular, label-based time axes.

```python
from geopatcher import TemporalPatcher, temporal

patcher = TemporalPatcher(
    geometry=temporal.geometry.FixedLookback(length=7),
    sampler=temporal.sampler.RegularStride(step=1),
    window=temporal.window.CausalBoxcar(),
    aggregation=temporal.aggregation.Mean(),
)
```
"""

from __future__ import annotations

from geopatcher.temporal import aggregation, geometry, sampler, stencils, window


__all__ = ["aggregation", "geometry", "sampler", "stencils", "window"]
