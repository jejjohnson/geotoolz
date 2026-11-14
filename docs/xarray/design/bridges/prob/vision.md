---
status: draft
version: 0.1.0
---

# xrtoolz.prob — Vision

**Probability and statistics on labeled arrays. Distributions whose
parameters are DataArrays; sampling and log-prob that round-trip
through xarray; numpyro / pyrox plate semantics tied to DataArray
dims.**

[numpyro](https://num.pyro.ai/) is the inference engine — MCMC, SVI,
predictive sampling, every handler under the sun. [pyrox](https://github.com/jejjohnson/pyrox)
glues numpyro to [Equinox](https://docs.kidger.site/equinox/) modules
so a single `__call__` can host params, sample sites, priors, and
guides. xrtoolz already speaks `Operator`-flavoured pipelines over
`DataArray`s. What's missing is the bridge that lets a user write:

```python
import xrtoolz.prob as xpr

prior = xpr.Normal(loc=climatology_da, scale=stddev_da)
sample = prior.sample(key, sample_shape=("ensemble=50",))
# sample.dims == ("ensemble", "time", "lat", "lon")
```

— a distribution parameterised by DataArrays, with samples that come
back as DataArrays carrying the right coords. And, the same
distribution drops into a numpyro / pyrox model without surgery.

## Why a bridge

### Pain 1 — Parameter shapes are silently positional

numpyro distributions broadcast by *position*. A `Normal(loc=loc,
scale=scale)` where `loc` is `(12, 73, 144)` and `scale` is `(73,
144)` works fine — numpyro broadcasts the last two axes. But if a
user has *DataArrays* with dims `(month, lat, lon)` and `(lat, lon)`
and someone reorders them upstream to `(lat, lon, month)` and `(lat,
lon)`, the broadcast still succeeds but now the distribution is
broadcasting `month` against `month`, not against the spatial grid.
The numbers are wrong; the code doesn't complain.

The bridge ties parameter axes to *dim names*. `loc.dims = ("month",
"lat", "lon")`, `scale.dims = ("lat", "lon")`, and the bridge knows
those align by name; reorders are safe.

### Pain 2 — Plate semantics by hand

numpyro's `plate` is the right primitive for conditionally-independent
draws. Used correctly, it adds the i.i.d. semantics that match a
DataArray's natural "this dim is the batch" interpretation. Used
incorrectly (wrong axis, missing `dim=` kwarg), it silently changes
which axis is shared and which is independent.

The bridge attaches plate semantics to dim names. ``plate`` is only
meaningful inside a numpyro model trace (``numpyro.plate`` scopes
``numpyro.sample`` sites, not free distribution ``.sample()`` calls),
so the model-facing form is:

```python
def model(mu_da, sigma_da):
    with xpr.plate("ensemble"):
        obs = xpr.sample("obs", xpr.Normal(loc=mu_da, scale=sigma_da))
```

and the bridge maps ``"ensemble"`` to numpyro's ``dim=`` argument by
finding the position of ``"ensemble"`` in the parameter DataArrays.
For free draws outside a trace, just call ``.sample(key, ...)``
directly without ``plate`` — the bridge labels the output axes from
the parameter dims either way.

### Pain 3 — Sample-shape labels

`numpyro` samples have shape `(*sample_shape, *batch_shape,
*event_shape)`. Translating this back to a DataArray requires the
user to know which axis is which and what to call it. The bridge
takes labeled `sample_shape=("draw=100",)` tokens and produces
DataArrays whose new sample dims are properly named.

### Pain 4 — Round-tripping through pyrox modules

A pyrox `PyroxModule` calls `self.pyrox_sample("w", dist)` which
registers a numpyro sample site. If `dist` is parameter-shaped by
DataArray, the user wants:

```python
class GPSpatialPrior(PyroxModule):
    @pyrox_method
    def __call__(self, climatology: xr.DataArray):
        return self.pyrox_sample(
            "field",
            xpr.Normal(loc=climatology, scale=self.scale_da).to_numpyro(),
        )
```

The bridge provides `.to_numpyro()` (returns a `numpyro.distributions.Distribution`
for use inside numpyro / pyrox models) and `.sample()` /
`.log_prob()` (DataArray-typed, for use *outside* models — e.g.
prior predictives in a notebook).

## User stories

### S1 — Climatology-anchored Normal

```python
sst_anomaly_prior = xpr.Normal(
    loc=climatology_sst,           # (month, lat, lon)
    scale=climatology_std,          # (month, lat, lon)
)
draws = sst_anomaly_prior.sample(key, sample_shape=("draw=200",))
# draws.dims == ("draw", "month", "lat", "lon")
```

### S2 — Hierarchical with a plate

```python
def model(data: xr.DataArray):
    # data.dims = ("station", "time")
    mu = xpr.sample("mu", xpr.Normal(0.0, 5.0))
    sigma = xpr.sample("sigma", xpr.HalfNormal(1.0))
    with xpr.plate("station", data):
        station_effect = xpr.sample(
            "alpha",
            xpr.Normal(mu, sigma),
        )
    xpr.sample(
        "obs",
        xpr.Normal(loc=station_effect, scale=0.1),
        obs=data,
    )
```

`plate` reads the size from `data`'s `"station"` dim; the bridge
hands numpyro the correct `dim=` argument under the hood.

### S3 — Posterior sample as DataArray

```python
mcmc = NUTS(model)
mcmc.run(key, data)
posterior = xpr.to_dataset(mcmc.get_samples(), dims_per_site={
    "alpha": ("station",),
    "mu": (),
    "sigma": (),
})
# posterior is an xarray.Dataset with the right dim names per site.
```

### S4 — Inside a pyrox module

```python
class GP_Prior(PyroxModule):
    @pyrox_method
    def __call__(self, X: xr.DataArray):
        K = build_kernel_matrix(X, self.lengthscale)        # NamedOperator
        return self.pyrox_sample(
            "f",
            xpr.MultivariateNormal(loc=0.0, cov=K).to_numpyro(),
        )
```

The MVN here is the labeled MVN from `xrtoolz.linalg`, exposed
through `xrtoolz.prob` for numpyro composition.

### S5 — Diagnostics inside a Sequential

```python
from xrtoolz import Sequential
from xrtoolz.prob import Sample, LogProb

prior_sweep = Sequential([
    Sample(prior, key=key, sample_shape=("draw=100",)),
    # ...downstream diagnostics on prior draws...
])
```

`Sample` / `LogProb` are Layer-1 `Operator`s, so they fit alongside
`xrtoolz.geo` operators in a pipeline.

## Design principles

1. **Distributions take DataArrays.** Parameters that have a natural
   shape are DataArrays with dim names; scalar parameters stay as
   floats or 0-d arrays.
2. **Dim names are plates.** Where numpyro needs `dim=`, the bridge
   computes it from dim names on the parameter arrays.
3. **`.sample()` and `.log_prob()` use DataArrays.** Outside-the-model
   use (priors, diagnostics, predictive checks) returns DataArrays.
4. **`.to_numpyro()` returns a plain numpyro Distribution.**
   Inside-the-model use (sample sites, log-probs computed by the
   inference engine) drops down to numpyro types unchanged.
5. **pyrox is a peer, not a wrapper.** `xpr.PyroxSample` and friends
   are thin re-exports so users importing from one namespace get the
   full surface, but the bridge does not vendor or shim pyrox.
6. **MCMC / SVI output → xarray Dataset.** `xpr.to_dataset` provides
   the canonical "raw numpyro samples → labeled posterior" conversion.

## Anti-goals

- **No new inference algorithms.** numpyro owns MCMC, SVI,
  Predictive.
- **No re-implementation of pyrox.** pyrox's PyroxModule /
  Parameterized live in pyrox.
- **No standalone Bayesian-modeling DSL.** Users write models with
  numpyro / pyrox; the bridge just makes the parameter and sample
  surface labeled.
- **No arviz integration in v1.** `to_dataset` produces a plain
  `xarray.Dataset`; arviz can ingest that. We don't ship
  arviz-flavoured InferenceData converters until usage demands them.

## What success looks like

A notebook can declare a Bayesian model whose every distribution is
parameterised by DataArrays, run NUTS or SVI, and get a posterior
Dataset whose coords match the model's inputs — with no manual
shape bookkeeping, no `da.values`, and no `np.swapaxes` calls.
