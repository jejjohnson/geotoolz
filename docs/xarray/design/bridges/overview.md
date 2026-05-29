---
status: draft
version: 0.1.0
---

# Bridge modules — shared overview

The three new submodules (`einx`, `linalg`, `prob`) are independent
bridges but share the same shape, the same dim-bookkeeping convention,
and the same place in the xrtoolz layer stack. This doc fixes the
shared model so the per-module docs only need to say what's different.

## Why these three

xrtoolz today is strong on Earth-system **diagnostics** (operators that
read a Dataset and return a Dataset of derived quantities). The
three things diagnostic pipelines reach for that don't fit that mold —
and don't have an xarray-native home — are:

1. **Named-tensor algebra.** Multi-axis einsums, rearranges, reductions
   over the dims a DataArray already carries. Today this means
   `da.values` → numpy → `einops` / `np.einsum` → wrap back. The
   round-trip drops coords and dim names that the user already wrote
   down.
2. **Structured linear algebra.** Covariance operators, Kalman steps,
   GP solves, low-rank updates over spatiotemporal grids. gaussx
   handles the math elegantly, but the user manually flattens
   `(time, lat, lon) → (N,)`, calls `gaussx.solve(K, y_flat)`, and
   reshapes the result. The labeled metadata is lost in the middle.
3. **Probabilistic modeling.** Loc/scale parameters whose values are
   *DataArrays* (e.g. a `Normal(loc=climatology, scale=stddev)` where
   both are `(month, lat, lon)`-shaped). numpyro plates implement the
   right semantics but require careful manual indexing.

Each module adds the missing label-preserving layer between xarray and
its respective backend. None of them reimplement the backend.

## Place in the stack

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 2 — Graph                                                  │
│  pipekit.Graph DAGs, ModelOp inference wrappers                   │
├──────────────────────────────────────────────────────────────────┤
│  Layer 1 — Operators                                              │
│  xrtoolz.Operator subclasses; Sequential chains                  │
│  ┌─ existing ──────────────┐  ┌─ new ────────────────────────┐   │
│  │ geo, ocn, atm, rs, ice  │  │ einx, linalg, prob ops       │   │
│  └─────────────────────────┘  └──────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│  Layer 0 — Primitives                                             │
│  Pure functions over DataArray (and Dataset adapters)            │
│  ┌─ existing ──────────────┐  ┌─ new ────────────────────────┐   │
│  │ regrid, detrend, masks  │  │ einsum, solve, sample, …     │   │
│  └─────────────────────────┘  └──────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                       │
       ┌───────────────┼─────────────────────────┐
       ▼               ▼                         ▼
   ┌─────────┐    ┌─────────┐               ┌─────────┐
   │  einx   │    │ gaussx  │ ── lineax ──▶ │ numpyro │
   └─────────┘    └─────────┘    matfree    └─────────┘
                                            └─ pyrox ─
```

The new modules sit beside the existing domain submodules in Layer 0/1
and inherit DataTree dispatch, `Sequential`, `Graph`, and `Augment`
behaviour from the existing core unchanged.

## The contract

Every Layer-0 function in the three bridges obeys:

```python
def f(*arrays: DataArray, **named_args) -> DataArray:
    ...
```

- **Inputs.** One or more `xarray.DataArray`s plus keyword arguments.
  `Dataset` is supported via `xrtoolz.linalg.pack` /
  `xrtoolz.einx.pack_dataset` style helpers — explicit, never
  automatic.
- **Output.** A single `DataArray`. Multi-output ops return a `Dataset`
  *only* when the output is genuinely a collection of named arrays
  (e.g. `cholesky_factors` returning `L, log_diag`); otherwise prefer
  a tuple of DataArrays.
- **Dims as type.** Operations are specified by dim *name*, not
  position. `einsum("... lat lon, ... lat lon -> ... lat lon", a, b)`
  is the rule, not `a * b` with implicit axis ordering.
- **Coord policy.** Each function's docstring states, in this exact
  order:
  - which input coords are forwarded onto the output (by name),
  - which dims are dropped (e.g. reduced over),
  - what happens to misaligned coords on shared dims (default: raise,
    matching xarray's strict alignment).

## Layer-1 wrappers

For every Layer-0 function `xrtoolz.<module>.foo(da, ...)` there is an
`Operator` subclass `xrtoolz.<module>.Foo` whose `_apply(da)` calls the
function. Operators are how the bridges enter `Sequential` / `Graph` /
`Augment`. The convention follows the existing `xrtoolz.geo` pattern:

```python
from xrtoolz import Sequential, Augment
from xrtoolz.einx import Rearrange
from xrtoolz.linalg import Solve
from xrtoolz.prob import Sample

pipeline = Sequential([
    Rearrange("time lat lon -> (lat lon) time"),
    Solve(operator=K),                    # gaussx structured op
    Sample(prior, key=key, sample_shape=()),
])
```

`Augment` / `ApplyToEach` from `xrtoolz.combinators` continue to work,
since these operators inherit `xrtoolz.Operator`.

## Dependencies and install

| Bridge   | New runtime deps                  | Extra               |
| -------- | --------------------------------- | ------------------- |
| `einx`   | `einx` (and its `numpy`/`jax`/`torch` backend choice) | **core dep** (no extra; see note) |
| `linalg` | `gaussx`, `lineax`, `jax`         | `pip install xrtoolz[linalg]` |
| `prob`   | `numpyro`, `pyrox`, `jax`         | `pip install xrtoolz[prob]`   |

> **einx is a core dependency (v0.2.0).** Unlike `linalg` / `prob`,
> einx is installed with base `xrtoolz` rather than behind an extra,
> because core kernels across the package adopt it. This reverses the
> einx decision D9. `import xrtoolz` still stays light — the
> `xrtoolz.einx` submodule is not imported at package init and imports
> einx lazily inside its functions.

All three sit behind lazy imports (xrtoolz D4 pattern: see
`src/xrtoolz/inference/__init__.py`). `import xrtoolz` never imports
JAX, einx, numpyro, or anything heavy. Importing `xrtoolz.linalg.solve`
imports gaussx the first time the function is called, not at module
load.

## Backend-array policy

xarray is array-library-agnostic — a `DataArray` can wrap numpy, dask,
jax, cupy, or any duck array. The bridges keep that property:

- **`einx`** delegates to einx, which dispatches on the underlying
  array library. xrtoolz only re-labels the result.
- **`linalg`** requires the underlying array to be a JAX array
  (gaussx / lineax / matfree are JAX-only). The bridge raises
  `TypeError` with an actionable message if the input wraps numpy.
  No silent conversion.
- **`prob`** requires JAX for the same reason. Output samples are JAX
  arrays inside DataArrays; users opt in to `.compute()` /
  `np.asarray` if they want numpy.

## Non-goals (shared)

- **No vendored copy of einx / gaussx / numpyro semantics.** If
  upstream changes, the bridge follows. The bridges don't try to
  stabilise APIs across upstream versions; we pin the lower bound
  instead.
- **No DataFrame interop.** Pandas integration is a separate question
  and lives elsewhere (`pyrox.api`, etc.).
- **No new composition primitives.** `Sequential` / `Graph` / `Augment`
  cover everything we need; the bridges add operators, not
  combinators.
- **No CRS or geophysical knowledge in these modules.** That stays in
  `xrtoolz.geo` / `xrtoolz.ocn` / `xrtoolz.atm` / `xrtoolz.rs`. The
  bridge modules are domain-agnostic.

## Open shared questions

1. **Naming.** `linalg` collides with `numpy.linalg` mental model;
   `prob` is short but generic. Alternatives considered:
   `xrtoolz.{gauss,ops,stats}`. Sticking with `{einx, linalg, prob}`
   for now — see per-module decisions docs.
2. **Coord realignment.** Should `einx.einsum(...)` reindex on shared
   dims with mismatched coords (a la xarray's default), or raise (a la
   numpy semantics)? Default raise; opt-in `align=True` flag.
3. **DataTree story.** Inheriting leaf-wise dispatch is free; what
   we don't have is *cross-leaf* ops (a Kronecker product whose
   factors live on different leaves). Out of scope for v1.
