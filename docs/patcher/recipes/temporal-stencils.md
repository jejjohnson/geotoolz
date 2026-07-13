# Temporal stencils — coordinate-aware time windows

The integer-index temporal samplers (`TemporalRegularStride`,
`TemporalLookbackHorizon`, …) work in *array steps*. That is fine when
you know the source cadence up front, but it couples your notebook to a
specific store. Re-point a `lookback=3` window from a 3-hourly ARCO-ERA5
store to a 1-hourly one and you silently get a 3-hour window instead of
a 9-hour one.

`TimeStencil` lets you say what you mean — "9 hours of context, 3 hours
of horizon, sampled at the source cadence" — and validates that the
request exactly tiles the source grid. Switching cadence either works
unchanged or raises a clear error; it never silently truncates.

This recipe shows three layers:

1. The pure-function path — `Stencil` + `xarray.isel`.
2. The four-axis path — `TemporalStencilGeometry` +
   `TemporalStencilSampler` inside a `TemporalPatcher`.
3. What the v0.1 constraints are and what they catch.

See **ADR-004** in `docs/decisions.md` for the design rationale and the
backwards-compatibility story.

## Prerequisites

```bash
pip install 'geopatcher[grid]'   # xarray + numpy already in core
```

A 1-D coordinate array along the time axis. For an `xarray.DataArray`,
the easiest path is `XarrayField(da).time_coord()` — it returns
`da["time"].values` for `datetime64`-typed coords and raises a typed
error on `cftime`-typed ones.

## 1. Pure-function path

The stencil math is independent of the patcher. If you already have an
`xarray` Dataset and just want validated, no-truncate slices, use the
primitives directly:

```python
import xarray as xr
import numpy as np
from geopatcher.time import (
    TimeStencil, valid_origin_points, build_sampling_slices,
)

ds = xr.open_zarr("gs://.../era5.zarr", chunks=None)
time = ds["time"].values                                # datetime64[ns]

stencil = TimeStencil(start="-9h", stop="3h", step="3h", closed="both")

origins = valid_origin_points(time, stencil)[::2]       # every 6h
slices  = build_sampling_slices(time, origins, stencil)

sample  = ds.isel(time=slices[0])                       # lazy xarray.Dataset
```

`valid_origin_points` returns only the origins for which the full window
fits — no half-windows at either edge. `build_sampling_slices` raises a
labelled `ValueError` if `stencil.step` doesn't evenly divide the source
step, so cadence mismatches surface at the slicing call, not in a
malformed batch downstream.

## 2. Four-axis path

If you're already living inside `TemporalPatcher`, the stencil drops in
as a Geometry + Sampler pair:

```python
from geopatcher.fields import XarrayField
from geopatcher.time import (
    TimeStencil, TemporalPatcher,
    TemporalStencilGeometry, TemporalStencilSampler,
    TemporalCausalBoxcar, TemporalForecast,
)

field = XarrayField(ds["t2m"])
coord = field.time_coord()                              # 1-D datetime64

stencil = TimeStencil(start="-9h", stop="3h", step="3h", closed="both")

tp = TemporalPatcher(
    geometry    = TemporalStencilGeometry(stencil, source_step=np.timedelta64(3, "h")),
    sampler     = TemporalStencilSampler(stencil, every=2, shuffle=True, seed=0),
    window      = TemporalCausalBoxcar(),
    aggregation = TemporalForecast(horizon=1),
)

for patch in tp.split(field.da.values, coord=coord, prefetch=4):
    yhat = model(patch.data)                            # patch.data is one window
```

Three things to notice:

- `tp.split` (and every other `TemporalPatcher` method that takes a
  series) gains an optional `coord=`. It is **required** when the
  geometry or sampler is coordinate-aware; otherwise it is ignored.
- The sampler still yields integer indices into `coord`. The patcher
  resolves each anchor to a coordinate value internally for dispatch and
  for the hook payload.
- `get_config()` round-trips the stencil as YAML-friendly strings, so
  the same notebook can be replayed by another consumer without
  re-stating the cadence.

Hook authors can opt into the new payload by accepting a trailing
`coord_value` arg:

```python
class TimestampedProgress:
    def on_patch_start(self, anchor, coord_value=None):
        print(f"anchor={anchor} at {coord_value}")
```

Existing single-arg hooks keep working — the patcher dispatches with the
exact number of positionals the callback declares.

## 3. v0.1 constraints

- **Stride-1 only.** A stencil whose `step` is greater than the source
  cadence (`step=2h` against a 1-hourly source) would yield strided
  slices, which `TemporalWindow.weights` and
  `TemporalAggregation.merge` don't yet support. The geometry raises at
  construction when `source_step` is supplied, and re-checks at resolve
  time when it isn't.
- **`cftime` coords.** Not yet supported. `XarrayField.time_coord`
  raises a typed `TypeError` with a pointer at
  `DataArray.indexes['time'].to_datetimeindex()`.
- **Spatial stencils.** This work covers the time axis only; the
  symmetric `SpatialStencilGeometry` is tracked as future work in the
  same issue.
- **`amerge` for stencil-driven async aggregations.** Out of scope for
  v0.1; tracked alongside the async `asplit`/`amerge` thread.

## See also

- `TimeStencil` / `Stencil` / `divide_evenly` /
  `build_sampling_slices` / `valid_origin_points` — primitives.
- `TemporalStencilGeometry` / `TemporalStencilSampler` — four-axis
  integration.
- `XarrayField.time_coord` — helper to extract the coordinate vector.
- ADR-004 — design rationale.
