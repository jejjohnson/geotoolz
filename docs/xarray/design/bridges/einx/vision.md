---
status: draft
version: 0.1.0
---

# xrtoolz.einx — Vision

**Named-tensor algebra (einsum, rearrange, reduce, repeat, …) where the
named axes *are* the DataArray dims.**

[einx](https://github.com/fferflo/einx) gives the most expressive
named-axis pattern language available in the Python array ecosystem —
a single notation for `einsum`, `rearrange` (a la einops), reductions,
elementwise broadcasts, and scatter/gather, parameterized by axis
*names* rather than positions. xarray DataArrays already carry axis
names. The bridge between them is mechanical and small, and it
removes a class of bugs that show up when a pipeline has to flatten a
labeled array, do the math, and reshape it back.

## Why a bridge instead of using einx directly

einx already accepts arbitrary array backends, including numpy / JAX /
PyTorch. So *technically* the user can write:

```python
import einx
arr = da.values                                    # numpy / dask / jax
out = einx.einsum("t lat lon, lat lon -> t", arr, w_values)
result = xr.DataArray(out, dims=("time",), coords={"time": da.time})
```

The pain is the bottom two lines. The user has to:

1. Drop into raw values, losing dim names.
2. Type the einx pattern *positionally* (the pattern's `t lat lon`
   slots must match the DataArray's actual dim *order*).
3. Reconstruct a DataArray, manually picking which coords survive.

The bridge collapses all three:

```python
import xrtoolz.einx as xnx
out = xnx.einsum("time lat lon, lat lon -> time", da, w)
# out is a DataArray with dim 'time' and the right coords.
```

- The pattern is written in terms of *dim names*, never positions.
- xrtoolz introspects the input dims and transposes to match the
  pattern before dispatching to einx.
- Coords on surviving dims are forwarded automatically; coords on
  reduced/dropped dims are dropped explicitly with no rename ambiguity.

## User stories

### S1 — Field-times-field reductions

> "Multiply two fields and sum over space, keeping the time dim."

```python
# Energy at each time step.
energy = xnx.einsum(
    "time lat lon, time lat lon -> time",
    u, v,
)
```

Today this is `(u * v).sum(dim=("lat", "lon"))` — works, but breaks
the moment one of the operations is non-trivial (e.g. an outer product
across a new dim).

### S2 — Block reshape / patchify

> "Split the lat/lon grid into 4×4 patches, keeping time."

```python
patches = xnx.rearrange(
    da,
    "time (lat_blk lat_in) (lon_blk lon_in) -> time (lat_blk lon_blk) lat_in lon_in",
    lat_in=4, lon_in=4,
)
```

einops does this, but the input/output dim names disappear in the
process. The bridge keeps `time` named, names the new patch dim, and
keeps the spatial sub-dims labeled (`lat_in`, `lon_in`).

### S3 — Batched matmul over a "feature" dim

> "Project a vector field through a learned weight matrix, batched
> over time × lat × lon."

```python
y = xnx.einsum(
    "time lat lon channel_in, channel_in channel_out -> time lat lon channel_out",
    field, weights,
)
```

The natural shape for ML features in earth-science cubes. Today this
is `xr.dot(field, weights, dims="channel_in")` if you happened to name
both arrays' dim `channel_in` correctly. The einx surface generalises
to arbitrary fan-in / fan-out without the user having to remember
which dim names align.

### S4 — Reduction with custom op

> "Take the median over time at each (lat, lon)."

```python
m = xnx.reduce(
    da,
    "time lat lon -> lat lon",
    op="median",
)
```

xarray ships `mean` / `sum` / `std` / `min` / `max` on `.reduce(...)`.
einx exposes a richer op space via `np.add`/`np.maximum`/`jnp.median`
that flows through automatically.

### S5 — Inside an Operator pipeline

> "Use rearrange as one step of a Sequential pipeline."

```python
from xrtoolz import Sequential
from xrtoolz.einx import Rearrange, Einsum

pipeline = Sequential([
    Rearrange("time lat lon -> (lat lon) time"),
    # ... downstream linalg op on the (space, time) matrix
])
```

Layer-1 wrappers (`Rearrange`, `Einsum`, `Reduce`, `Repeat`) inherit
`xrtoolz.Operator`, so they get `Sequential`, `Graph`, `Augment`, and
`DataTree` dispatch for free.

## Design principles

1. **Patterns reference dim names.** `time lat lon` in the pattern
   refers to the input DataArray's dims, in any order. xrtoolz
   transposes input arrays to match the pattern before dispatching.
2. **No silent dim invention.** If the pattern mentions a dim that
   isn't on any input *and* isn't bound by a kwarg, raise — never
   pull a size from elsewhere.
3. **Coord policy is part of the signature.** Surviving dims keep
   their coords; dropped dims lose theirs; new dims have no coords
   unless the user provides `coords=` explicitly.
4. **Backend-neutral.** Whatever array library the DataArray wraps,
   einx + that backend handles. The bridge doesn't force a single
   backend.
5. **Thin.** No new pattern language, no new dispatch rules. einx
   does the work; xrtoolz wires the labels.

## Anti-goals

- **No general einsum / einops replacement.** Users who want raw
  einops continue to use it.
- **No CRS / physical units.** This module is pure named-tensor.
- **No coord arithmetic / interpolation.** That's `xrtoolz.interpolate`.
- **No backend abstraction layer.** We follow einx's choices.

## What success looks like

A user reading a notebook that mixes `xrtoolz.geo` regridding,
`xrtoolz.einx` rearranges, and `xrtoolz.linalg.solve` reads
end-to-end without `da.values` ever appearing. Patterns are typed
once, the labels survive the whole pipeline, and `Sequential.summary()`
can render the dim shapes through every step using `Signature`.
