---
status: draft
version: 0.1.0
---

# xrtoolz.prob — Architecture

## Package layout

```
src/xrtoolz/prob/
├── __init__.py            # Public re-exports
├── _src/                  # Layer-0 pure functions
│   ├── distributions.py   # Normal, HalfNormal, LogNormal, Beta, Poisson, MultivariateNormal, ...
│   ├── _base.py           # DistributionBridge — common label-bookkeeping for all dists
│   ├── plate.py           # plate() context manager
│   ├── primitives.py      # sample(), log_prob(), prior_predictive(), posterior_predictive()
│   ├── io.py              # to_dataset() — convert raw numpyro samples to xr.Dataset
│   └── _shape.py          # Translate dim names <-> numpyro batch/event/sample axes
├── operators.py           # Layer-1: Sample, LogProb, …
└── pyrox.py               # Thin re-exports of PyroxModule, PyroxParam, PyroxSample, pyrox_method
```

## Layer stack

```
┌────────────────────────────────────────────────────────────────────┐
│  Layer 2 — Models                                                   │
│    User's numpyro / pyrox model functions and modules               │
│    Composed via numpyro handlers + pyrox PyroxModule                  │
├────────────────────────────────────────────────────────────────────┤
│  Layer 1 — Operators                                                │
│    Sample(dist, key=…, sample_shape=…) -> DataArray | Dataset       │
│    LogProb(dist, *, var=…) -> 0-d DataArray                         │
│    PriorPredictive(model, key=…) -> Dataset                          │
│    PosteriorPredictive(model, posterior_samples, key=…) -> Dataset   │
├────────────────────────────────────────────────────────────────────┤
│  Layer 0 — Distributions and primitives                             │
│    DistributionBridge subclasses: Normal, HalfNormal, MVN, …        │
│      .sample(key, sample_shape=…) -> DataArray                       │
│      .log_prob(x: DataArray) -> jax scalar                          │
│      .to_numpyro() -> numpyro.distributions.Distribution             │
│    Module-level: sample(), plate(), to_dataset()                     │
├────────────────────────────────────────────────────────────────────┤
│  Bridge core                                                        │
│    DataArray params -> numpyro args:                                 │
│      - introspect dims                                              │
│      - reconcile via dim-name alignment                              │
│      - convert to jax arrays (broadcast-compatible)                  │
│    Sample shape interpretation:                                     │
│      - ("draw=100",) -> prepend dim 'draw' with size 100             │
│    Output reshape:                                                  │
│      - assemble result dims as (*sample_shape, *param_dims)          │
│      - forward coords from inputs                                    │
├────────────────────────────────────────────────────────────────────┤
│  Backend                                                            │
│    numpyro, pyrox, jax                                              │
└────────────────────────────────────────────────────────────────────┘
```

## The `DistributionBridge` base

```python
class DistributionBridge:
    """Base class for labeled wrappers around numpyro distributions.

    Subclasses register the underlying numpyro distribution class and
    declare which constructor kwargs are "parameter-shaped" (can be
    DataArrays) versus "scalar-shaped" (always floats/0-d arrays).

    The bridge handles:
      - dim-name reconciliation across DataArray-typed parameters;
      - conversion to JAX arrays for numpyro;
      - parameter broadcasting along the union of declared dims;
      - sample-shape interpretation and output relabeling.

    Subclasses must not override sample / log_prob / to_numpyro —
    they should only declare the parameter set.
    """

    numpyro_cls: type[numpyro.distributions.Distribution]
    param_names: tuple[str, ...]
    event_param: str | None = None       # For multivariate (e.g. "cov")

    def __init__(self, **params: xr.DataArray | float | jax.Array) -> None: ...
    def to_numpyro(self) -> numpyro.distributions.Distribution: ...
    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[str, ...] | tuple[int, ...] = (),
    ) -> xr.DataArray: ...
    def log_prob(self, x: xr.DataArray) -> jax.Array: ...
    @property
    def batch_dims(self) -> tuple[str, ...]:
        """Union of parameter dim names, in canonical order."""
```

Subclassing is one-line per distribution:

```python
class Normal(DistributionBridge):
    numpyro_cls = numpyro.distributions.Normal
    param_names = ("loc", "scale")


class MultivariateNormal(DistributionBridge):
    numpyro_cls = numpyro.distributions.MultivariateNormal
    param_names = ("loc", "covariance_matrix")
    event_param = "covariance_matrix"
```

## Parameter reconciliation

When a `Normal(loc=loc_da, scale=scale_da)` is constructed, the
bridge runs:

```text
1. Collect parameter DataArrays; non-DataArray params become 0-d.
2. Compute the union of dim names, preserving first-appearance order:
     batch_dims = ("month", "lat", "lon")
3. For each DataArray param:
     - assert its dims are a subset of batch_dims;
     - check shared-dim coords match (or raise CoordMismatch);
     - transpose to the canonical order (with missing dims left out).
4. Convert each param to a jax array, then ``jnp.broadcast_to`` to
   the union shape.
5. Stash (param_jax_arrays, batch_dims, coords) on the bridge for
   downstream sample / log_prob / to_numpyro calls.
```

The reconciled batch shape is the bridge's source of truth for both
`.sample()` output dims and `.to_numpyro()` distribution batch_shape.

## Sample-shape interpretation

```python
def _parse_sample_shape(
    sample_shape: tuple[str, ...] | tuple[int, ...],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return (dim_names, sizes). Accepts:

    - tuple of "name=size" tokens -> named output dims
    - tuple of ints -> auto-named "sample_0", "sample_1", ... with warning
    - empty -> ((), ())
    """
```

The sampled DataArray's dims are `(*sample_dim_names, *batch_dims,
*event_dims)` — the leading sample axes carry user-named draws, then
the broadcast batch axes, then any event axes for multivariate
distributions (`MultivariateNormal` → ``("event",)``, ``Dirichlet`` →
``("category",)``, etc.). Univariate distributions have an empty
``event_dims`` so the tail collapses to `(*sample_dim_names,
*batch_dims)`. Coords are forwarded from the parameter DataArrays
(for batch dims) and from the multivariate parameter (for event
dims, when present); sample-dim coords come from whatever the caller
passed.

## `plate` semantics

```python
@contextmanager
def plate(
    name: str,
    size_or_array: int | xr.DataArray | None = None,
    *,
    subsample_size: int | None = None,
) -> Iterator[numpyro.plate]:
    """Open a numpyro plate keyed by a dim name.

    Args:
        name: dim name. Inside the context, distributions whose
            parameters carry this dim treat it as conditionally
            independent.
        size_or_array: either an int (passed to numpyro.plate as
            ``size=``), a DataArray (size taken from its ``name``
            dim), or None (size inferred from the first distribution
            that carries the dim).
        subsample_size: optional minibatch size (passed through to
            numpyro).

    Yields:
        The numpyro plate object, so the caller can nest it normally.
    """
```

The bridge resolves `dim=` numpyro plate kwargs from the dim's
position in the canonical batch_dims of distributions inside the
context.

## `to_dataset` — posterior IO

```python
def to_dataset(
    samples: Mapping[str, jax.Array],
    *,
    dims_per_site: Mapping[str, tuple[str, ...]],
    coords: Mapping[str, xr.DataArray] | None = None,
    chain_dim: str = "chain",
    draw_dim: str = "draw",
) -> xr.Dataset:
    """Convert raw numpyro MCMC / SVI samples to an xarray Dataset.

    Args:
        samples: ``mcmc.get_samples()`` or ``svi.get_samples()`` output.
        dims_per_site: dim names per sample site, in the order numpyro
            laid out batch axes. The bridge prepends (chain, draw) as
            needed based on the array shape.
        coords: optional coords to attach.
        chain_dim, draw_dim: names for the leading axes.

    Returns:
        Dataset with one variable per site, dims named per
        ``dims_per_site``.
    """
```

This is the canonical exit point from a numpyro inference run back to
the xarray world.

## Operator class skeleton

```python
class Sample(Operator):
    """Draw from a distribution inside a Sequential / Graph."""

    def __init__(
        self,
        distribution: DistributionBridge,
        *,
        key: jax.Array,
        sample_shape: tuple[str, ...] = (),
    ) -> None:
        self.distribution = distribution
        self.key = key
        self.sample_shape = sample_shape

    def _apply(self, _ds: xr.Dataset | None = None) -> xr.DataArray:
        return self.distribution.sample(self.key, sample_shape=self.sample_shape)
```

`Sample` is a generator operator (no input). `LogProb` is a
single-input operator that returns a 0-d DataArray.

## pyrox integration

`pyrox.py` re-exports the pyrox primitives:

```python
from pyrox._core import (
    PyroxModule,
    PyroxParam,
    PyroxSample,
    Parameterized,
    pyrox_method,
)
```

Users importing from `xrtoolz.prob` get the pyrox surface directly:

```python
from xrtoolz.prob import PyroxModule, pyrox_method, Normal

class MyPrior(PyroxModule):
    pyrox_name = "MyPrior"

    @pyrox_method
    def __call__(self, climatology):
        return self.pyrox_sample(
            "field",
            Normal(loc=climatology, scale=0.1).to_numpyro(),
        )
```

No wrapper, no shim — just convenience re-exports.

## Dependencies

| Dep         | Version | Purpose                                              |
| ----------- | ------- | ---------------------------------------------------- |
| `numpyro`   | >=0.14  | Distributions, plates, handlers, inference.          |
| `pyrox`     | latest  | PyroxModule, Parameterized, etc.                     |
| `gaussx`    | latest  | For `MultivariateNormal` cov; bridge to xrtoolz.linalg. |
| `jax`       | latest  | Array backend.                                       |

Lazy-imported per the xrtoolz D4 pattern. `import xrtoolz` does not
pull numpyro or pyrox.

## Error policy

| Condition                                                          | Behaviour                                                                                                                       |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Two DataArray params have unequal coords on a shared dim            | Raise `CoordMismatch` from `xrtoolz.prob.errors`.                                                                                |
| Parameter has a dim numpyro doesn't know what to do with            | Raise `ValueError` naming the offending dim.                                                                                     |
| `sample_shape` is an int tuple                                      | Auto-name dims and `warnings.warn` once per call site.                                                                           |
| `plate(name, ...)` opened with no inner distribution carrying `name`| Raise `ValueError` — the plate would be vacuous.                                                                                 |
| `log_prob(x)` with `x.dims` incompatible with distribution batch_dims | Raise `DimMismatch`.                                                                                                            |

## Integration with existing xrtoolz

- `xrtoolz.einx.rearrange` reshapes parameter arrays before they're
  fed into distributions.
- `xrtoolz.linalg.NamedOperator` is the covariance type for
  `xrtoolz.prob.MultivariateNormal`.
- `Sample` and `LogProb` compose into `Sequential` / `Graph` like
  any other operator.
- `xrtoolz.signature.Signature` extends naturally — distribution
  batch_dims contribute to output sigs.
