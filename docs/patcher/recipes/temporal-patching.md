# Temporal patching

Cut a time axis into windows — rolling lookbacks, forecast samples, or
space × time cubes — with `geopatcher.TemporalPatcher` and
`geopatcher.SpatioTemporalPatcher`. The temporal axes mirror the spatial
ones; edge behaviour is in the
[Temporal boundary policy](../patching.md#temporal-boundary-policy).

| Axis | `temporal.` options |
|---|---|
| Geometry | `geometry.FixedLookback`, `LookbackHorizon`, `MultiScale`, `PhaseWindow`, `StencilGeometry` |
| Sampler | `sampler.RegularStride`, `Random`, `Explicit`, `StencilSampler` |
| Window | `window.CausalBoxcar`, `ExponentialDecay`, `TaperedTukey`, `Periodic` |
| Aggregation | `aggregation.Mean`, `Forecast`, `Fold` (RNN-style state passing), `HierarchicalCombine` |

These work in array steps. To say "9 hours of context" whatever the
cadence, use [Temporal stencils](temporal-stencils.md).

## Rolling windows over a series

`FixedLookback(length=12)` at anchor `t` covers the 12 steps ending at
`t`; `temporal.aggregation.Mean` averages overlapping windows per step.

```python
import numpy as np

import geopatcher as gp
from geopatcher import temporal

series: np.ndarray = np.arange(48, dtype=np.float32)   # (48,) float32

patcher: gp.TemporalPatcher = gp.TemporalPatcher(
    geometry=temporal.geometry.FixedLookback(length=12),
    sampler=temporal.sampler.RegularStride(step=6),
    window=temporal.window.CausalBoxcar(),
    aggregation=temporal.aggregation.Mean(time_len=48),
)

windows: list[gp.TemporalPatch] = list(patcher.split(series))  # 6 × (12,) float32
print(windows[0].anchor, windows[0].indices)          # 12 slice(1, 13, None)
smoothed: np.ndarray = patcher.merge(
    [w.with_data(np.asarray(w.data) * 2.0) for w in windows]   # (12,) float32 → (12,) float32
)                                                     # (48,) float64 · NaN = uncovered steps 0, 43–47
```

With the default `boundary="drop"`, anchors 0 and 6 would overflow the
start of the series, so the first window is anchored at 12. Pass a
multi-dimensional array and `time_axis=` to window one axis of it.

## Forecast samples

`LookbackHorizon(lookback, horizon)` returns the context and the target
in one window; `temporal.aggregation.Forecast` collects the horizon part
of each window by anchor.

```python
import numpy as np

import geopatcher as gp
from geopatcher import temporal

series: np.ndarray = np.arange(48, dtype=np.float32)   # (48,) float32

patcher: gp.TemporalPatcher = gp.TemporalPatcher(
    geometry=temporal.geometry.LookbackHorizon(lookback=12, horizon=3),
    sampler=temporal.sampler.RegularStride(step=1),
    window=temporal.window.CausalBoxcar(),
    aggregation=temporal.aggregation.Forecast(horizon=3),
)

samples: list[gp.TemporalPatch] = list(patcher.split(series))  # 34 × (15,) float32
context: np.ndarray = np.asarray(samples[0].data)[:12]          # (12,) float32 — steps 0..11
target: np.ndarray = np.asarray(samples[0].data)[12:]           # (3,) float32 — steps 12..14
horizons: dict[int, np.ndarray] = patcher.merge(samples)        # {anchor: (3,) float32}
```

## Space × time cubes

`SpatioTemporalPatcher` reads each spatial chip once, then slices it into
temporal windows. With `coupling="product"` every spatial anchor meets
every time anchor.

```python
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp
from geopatcher import temporal

# Shapes: T = 24 days, C = 1 band, H × W = 64 × 64 pixels.
field: gp.RasterField = gp.RasterField(
    GeoTensor(
        np.random.default_rng(0).random((24, 1, 64, 64), dtype=np.float32),  # (T, C, H, W) float32
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
cubes: gp.SpatioTemporalPatcher = gp.SpatioTemporalPatcher(
    spatial=gp.SpatialPatcher(
        geometry=gp.spatial.geometry.Rectangular(size=(32, 32)),
        sampler=gp.spatial.sampler.RegularStride(step=(32, 32)),
        window=gp.spatial.window.Boxcar(),
        aggregation=gp.spatial.aggregation.OverlapAdd(),
    ),
    temporal=gp.TemporalPatcher(
        geometry=temporal.geometry.FixedLookback(length=6),
        sampler=temporal.sampler.RegularStride(step=6),
        window=temporal.window.CausalBoxcar(),
        aggregation=temporal.aggregation.Mean(),
    ),
    coupling="product",
    time_axis=0,
)

patches: list[gp.SpatioTemporalPatch] = list(cubes.split(field))  # 4 chips × 3 windows
first: gp.SpatioTemporalPatch = patches[0]
print(first.space, first.time)                        # (0, 0) 6
chip: np.ndarray = np.asarray(first.data)             # (6, 1, 32, 32) float32
```

For event-triggered cubes (a plume detection, a storm track), use
`coupling="coupled"` with `spatial.sampler.Explicit(anchors_=[((row, col), t), …])`:
each `(space, time)` pair yields one patch.

## Pitfalls

- **The default boundary drops short windows.** Pass `boundary="shrink"`
  to keep shorter edge windows instead.
- **Keys.** Journal, cache and `errors` key a temporal patch by its
  anchor, or `(anchor, k)` for a multi-window geometry — see
  [Temporal and spatiotemporal keys](../patching.md#temporal-and-spatiotemporal-keys).
- `temporal.sampler.RegularStride(check_full_scan=True)` raises
  `IncompleteScanConfiguration` when windows leave steps uncovered.
