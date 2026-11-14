---
status: draft
version: 0.1.0
---

# xrtoolz.linalg — Vision

**Structured linear algebra on labeled arrays — solve, logdet,
cholesky, sample, and all the gaussx primitives — where rows and
columns are tagged by DataArray dim names.**

[gaussx](https://github.com/jejjohnson/gaussx) provides
structure-exploiting linear-algebra primitives for JAX: `solve`,
`logdet`, `cholesky`, `diag`, `trace`, `sqrt`, `inv`, plus operator
classes (`Kronecker`, `BlockDiag`, `LowRankUpdate`, ...) on top of
`lineax`. Earth-system covariance operators have natural structure —
a spatial covariance is `K_lat ⊗ K_lon`, a spatiotemporal one is
`K_time ⊗ K_space`, an obs-space covariance has block structure by
sensor — but plugging that structure into a pipeline today means
manually flattening DataArrays into JAX vectors, calling `gaussx.solve`,
and reshaping the result. The bridge keeps the dim labels attached
the whole way through.

## Why a bridge

Concrete pain points the bridge removes.

### Pain 1 — Manual flatten + reshape

Today, applying a Kronecker covariance to a DataArray means:

```python
y = field.transpose("time", "lat", "lon").values     # (T, H, W) → numpy/JAX
y_flat = y.reshape(-1)                                # (T*H*W,)

K = gaussx.Kronecker(K_time, K_lat, K_lon)
x_flat = gaussx.solve(K, y_flat)                      # (T*H*W,)

x = x_flat.reshape(T, H, W)
out = xr.DataArray(x, dims=("time", "lat", "lon"), coords=field.coords)
```

Four lines of bookkeeping for one solve. Every operator in the chain
has to remember the dim order. If the user pulls `lon` to the front
of `field` and forgets to update the Kronecker factor order, the
result is *silently wrong* — the math runs, the numbers don't mean
what they look like.

With the bridge:

```python
K = xrtoolz.linalg.NamedOperator(
    gaussx.Kronecker(K_time, K_lat, K_lon),
    dims=("time", "lat", "lon"),
)
x = xrtoolz.linalg.solve(K, field)         # field is a DataArray
# x is a DataArray with field's dims and coords.
```

Reordering `field`'s dims now changes nothing — `solve` reads
`field`'s dims, looks at `K.dims`, transposes to match, dispatches,
re-labels.

### Pain 2 — Multi-batched solves

A common Earth-system pattern is "solve `K x = y` for each member
of an ensemble" or "for each time step". gaussx handles the math via
`vmap`; the bookkeeping for *which axis is the batch* is on the user.

The bridge formalises this with a `batch_dims=` kwarg that's part of
the function signature:

```python
x = xrtoolz.linalg.solve(K, y, batch_dims=("ensemble",))
```

`K` operates on `(lat, lon)`; `y` carries `(ensemble, lat, lon)`;
the result carries `(ensemble, lat, lon)`. No explicit `vmap`, no
manual reshape.

### Pain 3 — Distribution / sampling

`gaussx.MultivariateNormal` consumes a covariance operator and a
flat mean vector. Samples are flat JAX arrays. A user with a 2D mean
field has to flatten the mean, get flat samples, reshape, and re-wrap.

The bridge exposes the same distribution with DataArray-typed `loc`
and DataArray-shaped samples:

```python
mvn = xrtoolz.linalg.MultivariateNormal(loc=mean_da, cov=K_named)
sample = mvn.sample(key, sample_shape=("draw=10",))   # 'draw' is a new dim
```

`sample` has dims `("draw", "time", "lat", "lon")` and the inherited
coords on the labeled dims.

### Pain 4 — Solver selection at pipeline boundaries

gaussx's solver strategies (`DenseSolver`, `CGSolver`, `BBMMSolver`,
`AutoSolver`, ...) are first-class objects. The bridge surfaces them
on the `solve` / `logdet` boundary so users pick a solver per-call
or attach one to a `Solve` operator:

```python
Solve(operator=K, strategy=gaussx.PreconditionedCGSolver(...))
```

The strategy threads through gaussx unchanged; the bridge does not
abstract over it.

## User stories

### S1 — Kronecker GP regression

```python
K_space = xrtoolz.linalg.NamedOperator(
    gaussx.Kronecker(K_lat, K_lon),
    dims=("lat", "lon"),
)
K_time = xrtoolz.linalg.NamedOperator(K_t, dims=("time",))

alpha = xrtoolz.linalg.solve(K_time, observations)
posterior_mean = xrtoolz.linalg.solve(K_space, alpha)
```

Solve at each factor, no flatten in sight.

### S2 — Ensemble Kalman update

```python
P = ensemble_covariance(ensemble_da, dim="member")       # NamedOperator
K_gain = xrtoolz.linalg.solve(R + P, observations)       # observation update
state = mean + K_gain @ innovation                       # @ via NamedOperator
```

R + P is composed via gaussx operator arithmetic; the bridge only
labels the inputs and outputs.

### S3 — Low-rank covariance sampling

```python
U = xnx.einsum("rank lat lon -> rank lat lon", basis)
K = xrtoolz.linalg.LowRankCovariance(U, dim="rank")
draws = xrtoolz.linalg.sample(K, key, sample_shape=("draw=20",))
```

Pipes naturally from `xrtoolz.einx`-shaped basis matrices into
gaussx's `LowRankUpdate`.

### S4 — Drop into a Sequential

```python
from xrtoolz import Sequential
from xrtoolz.linalg import Solve, Cholesky

pipeline = Sequential([
    # ...build K from data...
    Cholesky(),                                # K → L (lower factor)
    Solve(rhs=lambda ds: ds["y"]),             # forward solve
    # ...post-processing...
])
```

Layer-1 `Operator` subclasses make linalg fit the existing operator
contract.

## Design principles

1. **Operators carry dim labels.** `NamedOperator` wraps any
   `lineax.AbstractLinearOperator` (or gaussx subclass) plus a
   `dims=(...)` tuple that names what the rows/columns operate on.
2. **Functions accept DataArrays.** `solve(K, y)` — never a flat
   vector. The bridge handles the transpose/flatten/dispatch/reshape
   loop.
3. **JAX-only.** gaussx is JAX-only and we do not paper over that.
   Inputs must wrap JAX arrays. Numpy inputs raise with a pointer to
   `jnp.asarray(da)`.
4. **Strategies are pass-through.** Solver strategy / preconditioner
   choice flows verbatim from the call site to gaussx.
5. **Distributions are labeled.** `MultivariateNormal` etc. take
   DataArray `loc` and produce DataArray `sample`s.
6. **Layer-1 operators for every Layer-0 function.** `solve` →
   `Solve`, `logdet` → `Logdet`, etc.

## Anti-goals

- **No new operator algebra.** We do not add `Toeplitz`, `Circulant`,
  or anything gaussx doesn't already have. Bridge code lives next to
  the labeled layer; new structured operators are upstream contributions
  to gaussx.
- **No solver re-implementation.** Cholesky, CG, BBMM, SLQ all live
  in lineax / matfree / gaussx. The bridge does not reimplement them.
- **No multi-backend abstraction.** JAX only. NumPy users who want
  dense linalg can use `xarray`'s own `.dot` / `.linalg` or
  `scipy.linalg`.

## What success looks like

A user can write the linalg core of a Kalman filter, a Gaussian
process, or an ensemble update entirely in DataArray-space:
build operators by composing gaussx primitives + `NamedOperator`,
apply them with `solve` / `logdet` / `sample`, and never see a
`.values` call or a manual `reshape`.
