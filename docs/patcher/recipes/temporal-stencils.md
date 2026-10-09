# Temporal stencils

Ask for "9 hours of context and 3 hours of horizon" instead of a number
of array steps. A `geopatcher.temporal.stencils.TimeStencil` states the
window in physical units and checks that it tiles the source's time grid
exactly. A cadence change then either works unchanged or raises — it
never silently truncates.

Integer windows such as `temporal.geometry.LookbackHorizon(lookback=3)`
mean 9 hours on a 3-hourly store and 3 hours on an hourly one. Use a
stencil when the cadence belongs to the store, as with ARCO-ERA5.
Integer windows are in [Temporal patching](temporal-patching.md).

## Slice a dataset directly

The stencil functions work without a patcher: they return validated
`slice`s for `xarray.DataArray.isel`.

```python
import numpy as np
import xarray as xr

from geopatcher.temporal.stencils import TimeStencil, build_sampling_slices, valid_origin_points

# Shapes: T = 32 three-hourly steps, 8 × 8 grid.
time: np.ndarray = np.arange("2024-01-01T00", "2024-01-05T00", 3, dtype="datetime64[h]")  # (32,) datetime64[h]
t2m: xr.DataArray = xr.DataArray(
    np.random.default_rng(0).random((32, 8, 8), dtype=np.float32),  # (T, 8, 8) float32
    dims=("time", "lat", "lon"),
    coords={"time": time, "lat": np.arange(8.0), "lon": np.arange(8.0)},
)

stencil: TimeStencil = TimeStencil(start="-9h", stop="3h", step="3h", closed="both")  # offsets −9 h … +3 h
origins: np.ndarray = valid_origin_points(t2m["time"].values, stencil)[::2]  # (14,) datetime64, every 6 h
slices: list[slice] = build_sampling_slices(t2m["time"].values, origins, stencil)  # 14 slices
sample: xr.DataArray = t2m.isel(time=slices[0])        # (5, 8, 8) float32
```

- `valid_origin_points` returns only origins whose whole window fits — no
  half windows at either edge.
- `build_sampling_slices` raises a labelled `ValueError` when the stencil
  step does not divide the source step.

## Use a stencil in a TemporalPatcher

`temporal.geometry.StencilGeometry` and `temporal.sampler.StencilSampler`
put the stencil on the geometry and sampler axes. Pass the time
coordinate with `coord=`.

```python
import numpy as np
import xarray as xr

import geopatcher as gp
from geopatcher import temporal
from geopatcher.fields import XarrayField
from geopatcher.temporal.stencils import TimeStencil

time: np.ndarray = np.arange("2024-01-01T00", "2024-01-05T00", 3, dtype="datetime64[h]")  # (32,) datetime64[h]
field: XarrayField = XarrayField(
    xr.DataArray(
        np.random.default_rng(0).random((32, 8, 8), dtype=np.float32),  # (32, 8, 8) float32
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": np.arange(8.0), "lon": np.arange(8.0)},
    )
)
coord: np.ndarray = field.time_coord()                # (32,) datetime64[s]

stencil: TimeStencil = TimeStencil(start="-9h", stop="3h", step="3h", closed="both")
patcher: gp.TemporalPatcher = gp.TemporalPatcher(
    geometry=temporal.geometry.StencilGeometry(stencil, source_step=np.timedelta64(3, "h")),
    sampler=temporal.sampler.StencilSampler(stencil, every=2, shuffle=True, seed=0),
    window=temporal.window.CausalBoxcar(),
    aggregation=temporal.aggregation.Forecast(horizon=1),
)

windows: list[gp.TemporalPatch] = list(patcher.split(field, coord=coord))  # 14 × (5, 8, 8) float32
print(windows[0].anchor, windows[0].indices)          # 9 slice(6, 11, None)
```

- `coord=` is required when the geometry or sampler is coordinate-aware
  (`needs_coord = True`) and ignored otherwise. A `GridDomain` field's
  time coordinate is the default.
- The sampler still yields integer indices into `coord`.
- `get_config()` stores the stencil losslessly, so
  `geopatcher.config.from_config(geopatcher.config.axis_envelope(patcher))`
  rebuilds the same patcher.

## Timestamps in hooks

A hook whose `on_patch_start` takes a second argument receives the
anchor's coordinate value. Single-argument hooks keep working.

```python
import numpy as np

import geopatcher as gp
from geopatcher import temporal
from geopatcher.temporal.stencils import TimeStencil


class TimestampedProgress:
    """Print each window's anchor time."""

    def on_patch_start(self, anchor: int, coord_value: np.datetime64 | None = None) -> None:
        print(f"anchor={anchor} at {coord_value}")


time: np.ndarray = np.arange("2024-01-01T00", "2024-01-02T00", 3, dtype="datetime64[h]")  # (8,) datetime64[h]
series: np.ndarray = np.arange(8, dtype=np.float32)    # (8,) float32
stencil: TimeStencil = TimeStencil(start="-6h", stop="0h", step="3h", closed="both")
patcher: gp.TemporalPatcher = gp.TemporalPatcher(
    geometry=temporal.geometry.StencilGeometry(stencil),
    sampler=temporal.sampler.StencilSampler(stencil),
    window=temporal.window.CausalBoxcar(),
    aggregation=temporal.aggregation.Mean(),
)
windows: list[gp.TemporalPatch] = list(
    patcher.split(series, coord=time, hooks=[TimestampedProgress()])  # anchor=2 at 2024-01-01T06 …
)                                                     # 6 × (3,) float32
```

## Limits

- **Stride 1 only.** A stencil step larger than the source cadence (2 h on
  an hourly store) would need strided windows, which windows and
  aggregations do not support yet. `StencilGeometry` raises at
  construction when `source_step` is given, else when it resolves.
- **No `cftime` coordinates.** `XarrayField.time_coord` raises
  `TypeError`; convert with `DataArray.indexes["time"].to_datetimeindex()`.
- **Time only.** There is no spatial stencil geometry.
- ADR-004 in the [design decisions](../decisions.md) records the design.
