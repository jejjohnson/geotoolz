---
status: draft
version: 0.1.0
---

# xrtoolz.linalg — Decisions

## Resolved

| #  | Question                                          | Resolution                                                                                                                                  |
| -- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | Module name                                       | `xrtoolz.linalg`. Functional description over backend name; future additions (e.g. `scipy`-backed dense ops) fit without renaming.           |
| 2  | Backend                                           | JAX. gaussx is JAX-only and we don't paper over that.                                                                                        |
| 3  | Carrier for vectors / fields                      | `DataArray`.                                                                                                                                |
| 4  | Carrier for operators                             | `NamedOperator` (new class) wrapping `lineax.AbstractLinearOperator` + a `dims` tuple.                                                       |
| 5  | Dim ordering                                      | `NamedOperator.dims` is authoritative; inputs are transposed to match.                                                                       |
| 6  | Solver-strategy selection                         | Pass-through via `strategy=` kwarg. We don't abstract over gaussx's strategy types.                                                          |
| 7  | Batch dims                                        | Inferred as `set(input.dims) - set(operator.dims)`; overridable via `batch_dims=`.                                                           |
| 8  | Distribution surface                              | `xrtoolz.linalg.MultivariateNormal` wraps `gaussx.MultivariateNormal`; takes DataArray loc, NamedOperator cov, returns DataArray samples.    |
| 9  | Install                                           | `pip install xrtoolz[linalg]` adds gaussx + lineax + matfree + jax. Lazy.                                                                    |

## Key design choices

### D1 — `NamedOperator` is a new type, not a subclass of `lineax.AbstractLinearOperator`

A subclass would inherit gaussx's full surface for free, but it would
also make the bridge type a leaf in lineax's structural-dispatch
hierarchy — meaning gaussx's `isinstance`-based dispatch would have
to learn about `NamedOperator`, which is a leak.

Composition is the right model. `NamedOperator.operator` is the lineax
op; `NamedOperator` itself only adds labels. Primitives unwrap the
operator before calling gaussx; the result is wrapped back. lineax
and gaussx don't know `NamedOperator` exists.

### D2 — Dim order is operator-defined, not call-site-defined

A `Kronecker(K_lat, K_lon)` operator labels its dims as `(lat, lon)`,
in that order. If the user passes a DataArray with dims `(lon, lat)`,
the bridge transposes `(lon, lat) → (lat, lon)` before dispatch.
Reversing dim order *changes the math* (Kronecker is non-commutative);
the operator wins, the input transposes.

The user can override by re-wrapping: `NamedOperator(op, dims=("lon",
"lat"))`. The bridge does not auto-detect "you probably meant the
other order."

### D3 — JAX-only, with explicit error for numpy inputs

Silent conversion to JAX would hide compute-graph implications
(losing dask laziness, losing gradient tracking when called inside a
`grad`). We raise. The error message points at `jnp.asarray(da)` and
documents the trade-off.

### D4 — `solve` returns a DataArray, `logdet` returns a JAX scalar

`solve`'s result is naturally labeled (same dims as `y`). `logdet`'s
result is a scalar; wrapping it in a 0-d DataArray adds noise for the
common case. Layer-1 `Logdet` *does* return a 0-d DataArray for
pipeline composition; the function-level API stays scalar.

### D5 — Operator algebra forwards through `NamedOperator`

`K1 + K2` returns a new `NamedOperator` whose `.operator` is
`K1.operator + K2.operator` (gaussx handles the addition) and whose
`.dims` is `K1.dims` after asserting `K1.dims == K2.dims`. Same for
`@`, scalar `*`, and `.T`.

This means a user can build composite operators inline:

```python
import xrtoolz.linalg as xla

K = (K_prior + K_obs).materialize()           # if needed for debug
posterior = xla.solve(K_prior + K_obs, y)
```

### D6 — Distributions split between `linalg` and `prob`

`MultivariateNormal` lives in *both* `xrtoolz.linalg` and
`xrtoolz.prob`, with different contracts:

- `xrtoolz.linalg.MultivariateNormal` — operator-parameterized,
  optimized for structured covariances, JAX-native, returns
  DataArray samples. Use when the covariance has gaussx structure.
- `xrtoolz.prob.MultivariateNormal` — numpyro-backed, sits inside
  numpyro models with site registration. Use inside an inference
  loop.

They convert into one another via constructors (`from_numpyro`,
`to_numpyro_dist`). See `prob/decisions.md` D4 ("Two MVN classes
coexist") for the symmetric explanation. Two surfaces is a deliberate
cost.

### D7 — `sample_shape` accepts named tokens

`sample_shape=("draw=100",)` rather than `sample_shape=(100,)` so the
output dims are properly named. The integer form is accepted with a
one-time warning for numpyro-style call-sites.

### D8 — `NamedOperator` is frozen + PyTree-friendly

`@dataclass(frozen=True)` + custom `tree_flatten`/`tree_unflatten` so
JAX transformations work cleanly. `dims` and `coords` flow as static
PyTree metadata; the wrapped `lineax` operator is the leaf.

### D9 — Coord checks are size-strict, value-soft by default

If `NamedOperator.coords` is supplied, the bridge verifies that input
DataArray coords on shared dims *match* (raise on mismatch). If
`coords` is `None`, only the *sizes* are checked. This matches
numpy-style positional broadcasting unless the user opts in to
xarray-style label alignment.

### D10 — Recipes deferred

`gaussx.kalman_filter`, `gaussx.matheron_update`, etc. are
domain-spanning workflows. Wrapping them in DataArray-IO is
worthwhile but architecturally separate from primitives. Defer until
v1 ships and we know which recipes actually get called.

## Open

| #  | Question                                                                                          | Status                                                                                                                     |
| -- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| O1 | Should `NamedOperator.coords` be required when an operator is built from a kernel function?       | Probably yes — the kernel needs to know the coords. Encode by accepting `coords` in `NamedOperator.kronecker(...)` etc.    |
| O2 | Should `solve` accept a `Dataset` of multiple RHS and solve each?                                 | Defer; this is `Augment(Solve(...))` territory if the user really wants it.                                                |
| O3 | Where do "block" dims live for Block diag operators? `dim="ensemble"` style or implicit?           | Leaning explicit `dim=` — matches gaussx kwargs.                                                                            |
| O4 | Eager Cholesky vs lazy Cholesky operator (gaussx returns the latter)?                              | Eager via `cholesky(K).materialize()`; lazy by default (returns a `NamedOperator` whose `.operator` is a gaussx Cholesky). |
| O5 | Should `pack`/`unpack` survive on the public API or stay internal?                                  | Public — users will need it for one-off escapes to gaussx.                                                                  |

## Rejected

| #  | Option                                                                            | Why rejected                                                                                                                                  |
| -- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| R1 | `NamedOperator` subclasses `lineax.AbstractLinearOperator`                         | Bleeds labels into lineax's dispatch system; couples bridge to lineax's PyTree assumptions. Composition wins.                                  |
| R2 | DataArray-typed *covariance* (not via NamedOperator)                                | DataArrays don't model structured operators (Kronecker, low-rank). Forcing a DataArray representation either materialises everything (defeats gaussx) or invents a parallel structure registry. |
| R3 | Auto-vmap detection                                                                | Already implicit via JAX. Explicit `batch_dims=` lets users override; auto-detect is one more thing to debug when wrong.                       |
| R4 | Wrapping `lineax` solvers directly without going through gaussx                    | gaussx is the structural-dispatch layer the user already knows. Going around it splits the namespace and breaks struct-aware solves.            |
| R5 | A separate `xrtoolz.gaussx` namespace (literal upstream name)                       | Bridge is *not* gaussx — it's a labeled-data interface. Functional name reads cleaner for users.                                                |
