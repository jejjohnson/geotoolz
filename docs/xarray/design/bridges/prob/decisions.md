---
status: draft
version: 0.1.0
---

# xrtoolz.prob — Decisions

## Resolved

| #  | Question                                          | Resolution                                                                                                                                |
| -- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | Module name                                       | `xrtoolz.prob`. Generic, future-proof against adding non-numpyro backends (none planned, but the name doesn't preclude it).               |
| 2  | Carrier                                           | `DataArray` for parameters, samples, and obs.                                                                                              |
| 3  | Backend                                           | numpyro + pyrox.                                                                                                                          |
| 4  | Distributions surface                             | Subclasses of `DistributionBridge` with explicit constructor kwargs (no `**kwargs` magic).                                                |
| 5  | Plate semantics                                   | `plate(name=...)` keyed by dim name; bridge computes numpyro `dim=` from parameter dim positions.                                          |
| 6  | Sample shape                                      | `("draw=100",)` tokens (named, preferred); int tuples accepted with warning for numpyro compat.                                            |
| 7  | Inside-model vs outside-model use                 | `.to_numpyro()` for inside-model; `.sample()` / `.log_prob()` for outside (DataArray-typed).                                              |
| 8  | pyrox integration                                 | Re-exports of pyrox `_core` symbols. No wrapping.                                                                                          |
| 9  | Posterior IO                                      | `to_dataset()` — plain xarray Dataset, not arviz InferenceData.                                                                            |
| 10 | Install                                           | `pip install xrtoolz[prob]` adds numpyro + pyrox + jax. Lazy import.                                                                       |

## Key design choices

### D1 — Distribution constructors take DataArrays

Parameter-shaped kwargs (`loc`, `scale`, `concentration`, ...) accept
`xr.DataArray` directly. Scalar kwargs (`df`, `rate` when scalar)
accept floats. Mixing is fine: `Normal(loc=0.0, scale=stddev_da)`
broadcasts the scalar against the DataArray.

The contract: a DataArray param's dims become the distribution's
batch_dims (for univariate) or batch_dims + event_dims (for
multivariate; the user opts into event_dims explicitly).

### D2 — Dim names align across parameters

If two parameters share a dim name, the bridge enforces coord
equality on that dim. If a parameter has a dim no other parameter
has, the bridge broadcasts. Unlike numpy/JAX positional broadcasting,
which can silently swap axes when a refactor reorders dims, named
alignment is robust to dim reordering upstream.

### D3 — `MultivariateNormal` accepts NamedOperator covariance

`MultivariateNormal(loc=mu_da, covariance_matrix=K)` where `K` is a
`xrtoolz.linalg.NamedOperator`. The bridge:

1. Verifies `K.dims` equals `loc.dims[-len(K.dims):]`.
2. If the user wants the structured solve path (Kronecker etc.),
   `K`'s underlying gaussx operator is used for `log_prob` (via
   `gaussx.MultivariateNormal`).
3. If `K` is dense, falls through to numpyro's
   `MultivariateNormal(covariance_matrix=K.materialize())`.

So users get gaussx-flavoured structured MVN when they want it, and
plain numpyro semantics when they don't. The same class covers both.

### D4 — Two MVN classes coexist

`xrtoolz.linalg.MultivariateNormal` and `xrtoolz.prob.MultivariateNormal`
look similar but serve different roles:

| Class                                    | Use case                            | Backend           | Sample type   |
| ---------------------------------------- | ----------------------------------- | ----------------- | ------------- |
| `xrtoolz.linalg.MultivariateNormal`      | Standalone sampling / log-prob with structured cov; pipelines | gaussx            | `DataArray`   |
| `xrtoolz.prob.MultivariateNormal`        | Inside numpyro / pyrox models; supports `.to_numpyro()` for site registration | numpyro (+ gaussx for log_prob backend) | numpyro `Distribution` for inside-model; `DataArray` for outside |

They convert into each other via factory methods. Documented as
deliberate co-existence in both modules' decisions.

### D5 — pyrox is re-exported, not wrapped

`xrtoolz.prob.PyroxModule` is `pyrox._core.PyroxModule`. Same class,
not a subclass, not a re-init wrapper. Users who already use pyrox
import from `pyrox` directly; users who came through xrtoolz get the
convenience import from `xrtoolz.prob`. Two import paths, one class.

### D6 — `.sample()` outside a numpyro context returns a free draw

When called outside a numpyro / pyrox model body, `dist.sample(key,
sample_shape=...)` draws an i.i.d. sample using `dist.to_numpyro().sample(key, sample_shape)`
under the hood and rewraps as a DataArray. No site registration, no
trace effect.

Inside a numpyro model body (i.e., during `MCMC.run` / `SVI.run` /
`Predictive(...)` ), users call `xpr.sample(name, dist, ...)` —
mirroring `numpyro.sample`. The labeled version registers a site
and returns the sampled DataArray.

### D7 — `plate` size inference

If `plate("station")` is opened with no explicit size, the bridge
finds the first distribution inside the context that carries a
`"station"` dim and uses that dim's size. If none does, raise. If
multiple distributions carry the dim with conflicting sizes, raise.

### D8 — `to_dataset` requires explicit `dims_per_site`

numpyro samples are raw JAX arrays with shape `(chain?, draw,
*batch, *event)`. The bridge cannot infer dim names from the array
shape — that's the user's choice (e.g. a `(50, 73, 144)`-shaped
posterior could be `(draw, station, time)` or `(draw, lat, lon)`).
We require `dims_per_site={"alpha": ("station",), ...}` rather than
guess.

### D9 — Sample shapes are *named* by default

`sample_shape=("draw=100",)` is the canonical form. Integer-tuple
syntax (`sample_shape=(100,)`) works for numpyro compatibility but
emits a one-time warning per call site explaining the named form.
Reduces silent "what's `sample_0`?" mysteries.

### D10 — No `numpyro` re-exports

Unlike pyrox re-exports, we do *not* re-export `numpyro.handlers`,
`numpyro.distributions`, etc., from `xrtoolz.prob`. Reasons:

- numpyro is a large surface that evolves on its own schedule.
  Re-exporting freezes a snapshot.
- Users mixing labeled and raw numpyro will reach for both
  namespaces anyway; better one explicit import than half-mirrored
  conveniences.

## Open

| #  | Question                                                                                          | Status                                                                                                                                          |
| -- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| O1 | Should `to_dataset` auto-introspect dims when the user passes a `pyrox` model that ran the trace? | Maybe — pyrox can record site dim names if we add a hook. Defer until we have a couple of real-world models to test against.                    |
| O2 | First-class `arviz` `InferenceData` converter?                                                    | Defer until users ask. `to_dataset` produces a valid arviz input.                                                                                |
| O3 | Should `factor(name, log_factor)` register inside a numpyro trace automatically?                  | Yes — but only when called inside a trace. Outside, raise.                                                                                       |
| O4 | Distribution `expand` / `mask` / `to_event` analogs labeled by dim name?                          | Useful but expanded surface. Defer to v1.1.                                                                                                      |
| O5 | Should `Sample` operator accept a *callable* that builds the distribution from the threaded Dataset, like `xrtoolz.linalg.Solve`? | Yes — symmetric with `Solve(operator=Callable)`. Add in initial implementation.                                                                  |
| O6 | Coord forwarding when `plate(name, data=da)` subsamples? Posterior dim should carry `da.coords[name]` slice. | Defer; subsample is a numpyro feature we may not surface in v1.                                                                                  |

## Rejected

| #  | Option                                                                            | Why rejected                                                                                                                                |
| -- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| R1 | One unified `MultivariateNormal` class across linalg + prob                        | The contracts differ — linalg is operator-typed and JAX-typed; prob is numpyro-trace-aware. Forcing one class either bloats the API or loses semantics. |
| R2 | Subclass numpyro `Distribution` directly                                          | numpyro distributions are JAX-PyTrees with specific shape conventions; subclassing locks the bridge into numpyro's internal API. Composition is cleaner. |
| R3 | Implicit plate inference (no `plate(name)` context, dim-name based auto-plating)   | Too magical; broadcasts can be ambiguous. Explicit `plate(name)` is one extra line and keeps the model readable.                            |
| R4 | First-class arviz `InferenceData` in v1                                            | Arviz adds dependencies and opinions. Plain Datasets are already arviz-ingestible. Defer.                                                  |
| R5 | Re-export `numpyro.distributions` from `xrtoolz.prob`                              | Pinning a numpyro snapshot. Users import numpyro directly.                                                                                  |
| R6 | A `pyrox`-style `Parameterized` distribution mixin                                 | Adds a third axis of variation (param/prior/guide modes) onto a class that already has named dims and event/batch shapes. Out of scope.    |
