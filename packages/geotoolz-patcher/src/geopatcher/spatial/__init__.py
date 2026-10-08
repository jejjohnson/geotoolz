"""`geopatcher.spatial` — the four axes of a `SpatialPatcher`.

A spatial patcher is four independent choices, one module each:

- `geopatcher.spatial.geometry` — the patch's extent (`Rectangular`, …).
- `geopatcher.spatial.sampler` — where anchors go (`RegularStride`, …).
- `geopatcher.spatial.window` — per-pixel weights (`Hann`, `Boxcar`, …).
- `geopatcher.spatial.aggregation` — how outputs merge (`OverlapAdd`, …).

```python
from geopatcher import SpatialPatcher, spatial

patcher = SpatialPatcher(
    geometry=spatial.geometry.Rectangular(size=(256, 256)),
    sampler=spatial.sampler.RegularStride(step=192),
    window=spatial.window.Hann(),
    aggregation=spatial.aggregation.OverlapAdd(),
)
```
"""

from __future__ import annotations

from geopatcher.spatial import aggregation, geometry, sampler, window


__all__ = ["aggregation", "geometry", "sampler", "window"]
