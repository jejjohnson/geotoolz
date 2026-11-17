---
status: draft
version: 0.1.0
---

# xrtoolz.prob — API

Proposed public surface. Names track numpyro where possible to keep
the user's mental model intact.

## Imports

```python
import xrtoolz.prob as xpr               # Function API + distributions

from xrtoolz.prob import (                # Distributions
    Normal, HalfNormal, LogNormal, StudentT, Cauchy,
    Beta, Gamma, Exponential, Uniform,
    Bernoulli, Categorical, Poisson, Binomial,
    MultivariateNormal, Dirichlet,
)
from xrtoolz.prob import (                # Sampling / log-prob primitives
    sample, log_prob, plate, deterministic, factor,
)
from xrtoolz.prob import (                # Pipeline operators
    Sample, LogProb, PriorPredictive, PosteriorPredictive,
)
from xrtoolz.prob import (                # I/O
    to_dataset, from_dataset,
)
from xrtoolz.prob import (                # pyrox re-exports
    PyroxModule, PyroxParam, PyroxSample, Parameterized, pyrox_method,
)
```

`import xrtoolz` does not pull numpyro or pyrox.

## Distributions

Each labeled distribution is a subclass of `DistributionBridge` and
exposes the same three methods plus distribution-specific
constructor kwargs.

### Univariate

```python
class Normal(DistributionBridge):
    """Normal distribution parameterized by loc and scale.

    Args:
        loc: mean. Scalar or DataArray.
        scale: stddev. Scalar or DataArray.

    Shapes:
        batch_dims = union of loc.dims and scale.dims.
        event_dims = ().
    """
    def __init__(self, loc, scale): ...


class HalfNormal(DistributionBridge):
    def __init__(self, scale): ...

class LogNormal(DistributionBridge):
    def __init__(self, loc, scale): ...

class StudentT(DistributionBridge):
    def __init__(self, df, loc=0.0, scale=1.0): ...

class Cauchy(DistributionBridge):
    def __init__(self, loc, scale): ...

class Beta(DistributionBridge):
    def __init__(self, concentration1, concentration0): ...

class Gamma(DistributionBridge):
    def __init__(self, concentration, rate): ...

class Exponential(DistributionBridge):
    def __init__(self, rate): ...

class Uniform(DistributionBridge):
    def __init__(self, low, high): ...
```

### Discrete

```python
class Bernoulli(DistributionBridge):
    def __init__(self, *, probs=None, logits=None): ...

class Categorical(DistributionBridge):
    def __init__(self, *, probs=None, logits=None, dim: str = "category"): ...
    # `dim` declares the name of the categorical event axis on probs/logits.

class Poisson(DistributionBridge):
    def __init__(self, rate): ...

class Binomial(DistributionBridge):
    def __init__(self, total_count, *, probs=None, logits=None): ...
```

### Multivariate

```python
class MultivariateNormal(DistributionBridge):
    """MVN parameterized by loc DataArray and covariance.

    Args:
        loc: mean DataArray. The trailing ``event_dims`` are the MVN
            event axes.
        covariance_matrix: NamedOperator (preferred) or DataArray
            with row/col dims for a dense (N, N) covariance.
        event_dims: tuple of dim names on ``loc`` that are the MVN
            event axes. Remaining dims are batch.
    """
    def __init__(
        self,
        loc: xr.DataArray,
        *,
        covariance_matrix: NamedOperator | xr.DataArray | None = None,
        scale_tril: NamedOperator | xr.DataArray | None = None,
        precision_matrix: NamedOperator | xr.DataArray | None = None,
        event_dims: tuple[str, ...] | None = None,
    ): ...


class Dirichlet(DistributionBridge):
    def __init__(self, concentration: xr.DataArray, *, dim: str = "category"): ...
```

### Common methods

```python
class DistributionBridge:
    def sample(
        self, key: jax.Array, *,
        sample_shape: tuple[str, ...] | tuple[int, ...] = (),
    ) -> xr.DataArray:
        """Draw a labeled sample.

        ``sample_shape`` accepts ``("draw=100",)``-style tokens
        (named, recommended) or ``(100,)``-style ints (numpyro
        convention; auto-names the dim and warns once).
        """

    def log_prob(self, x: xr.DataArray) -> jax.Array:
        """Element-wise log-prob; sums broadcast over event_dims.

        ``x.dims`` must include the distribution's batch_dims +
        event_dims; extra dims are treated as sample dims.
        """

    def to_numpyro(self) -> numpyro.distributions.Distribution:
        """Return the underlying numpyro distribution.

        Use inside ``numpyro.sample`` / ``pyrox_sample`` / model
        bodies where you want numpyro semantics.
        """

    @property
    def batch_dims(self) -> tuple[str, ...]: ...
    @property
    def event_dims(self) -> tuple[str, ...]: ...
    @property
    def batch_coords(self) -> dict[str, xr.DataArray]: ...
```

## Module-level primitives

```python
def sample(
    name: str,
    distribution: DistributionBridge,
    *,
    obs: xr.DataArray | None = None,
    sample_shape: tuple[str, ...] = (),
) -> xr.DataArray:
    """Register a sample site with numpyro (labeled version).

    Equivalent to ``numpyro.sample(name, distribution.to_numpyro(), obs=obs)``
    when called inside a numpyro / pyrox model context; ``sample_shape``
    is only honored outside a model context.

    Returns a DataArray with the distribution's batch_dims + event_dims
    + (optional) sample_dims; the return value is also registered in the
    numpyro trace if one is active.
    """


def deterministic(name: str, value: xr.DataArray) -> xr.DataArray: ...

def factor(name: str, log_factor: jax.Array) -> None: ...

@contextmanager
def plate(
    name: str,
    size_or_array: int | xr.DataArray | None = None,
    *,
    subsample_size: int | None = None,
) -> Iterator[numpyro.plate]: ...
```

## Predictive helpers

```python
def prior_predictive(
    model: Callable,
    key: jax.Array,
    *,
    num_samples: int = 1,
    sample_dim: str = "draw",
    dims_per_site: Mapping[str, tuple[str, ...]] | None = None,
    **model_kwargs: Any,
) -> xr.Dataset:
    """Draw from the prior predictive of a numpyro / pyrox model.

    Wraps ``numpyro.infer.Predictive``; converts the result to a
    Dataset via ``to_dataset``.
    """


def posterior_predictive(
    model: Callable,
    posterior_samples: Mapping[str, jax.Array] | xr.Dataset,
    key: jax.Array,
    *,
    sample_dim: str = "draw",
    dims_per_site: Mapping[str, tuple[str, ...]] | None = None,
    **model_kwargs: Any,
) -> xr.Dataset: ...
```

## I/O

```python
def to_dataset(
    samples: Mapping[str, jax.Array],
    *,
    dims_per_site: Mapping[str, tuple[str, ...]],
    coords: Mapping[str, xr.DataArray] | None = None,
    chain_dim: str = "chain",
    draw_dim: str = "draw",
) -> xr.Dataset:
    """Convert raw numpyro samples to an xarray Dataset."""


def from_dataset(
    ds: xr.Dataset,
    *,
    chain_dim: str = "chain",
    draw_dim: str = "draw",
) -> dict[str, jax.Array]:
    """Inverse: Dataset -> {site: jax_array}.

    Useful for feeding posterior samples back into
    ``posterior_predictive`` after manipulating them in xarray.
    """
```

## Layer-1 operators

| Function              | Operator              | Notes                                                              |
| --------------------- | --------------------- | ------------------------------------------------------------------ |
| `dist.sample`         | `Sample`              | Generator op; no input. Constructor takes dist + key + sample_shape. |
| `dist.log_prob`       | `LogProb`             | Single-input; returns 0-d DataArray.                                |
| `prior_predictive`    | `PriorPredictive`     | Generator op; runs the model under `Predictive`.                    |
| `posterior_predictive`| `PosteriorPredictive` | Generator op; takes a posterior Dataset as init kwarg.              |

## pyrox re-exports

The full pyrox `_core` API is re-exported from `xrtoolz.prob`:

```python
from xrtoolz.prob import (
    PyroxModule,
    Parameterized,
    PyroxParam,
    PyroxSample,
    pyrox_method,
)
```

These are *direct re-exports* — no wrapping. Users who already know
pyrox get the same surface; users who came in via xrtoolz get a
single namespace for the probabilistic stack.

## Exception hierarchy

```python
class ProbBridgeError(Exception):
    """Base for xrtoolz.prob errors."""


class CoordMismatch(ProbBridgeError):
    """Two parameter DataArrays disagree on coords for a shared dim."""


class DimMismatch(ProbBridgeError, ValueError):
    """log_prob input dims don't match distribution batch + event dims."""


class PlateMisuse(ProbBridgeError, ValueError):
    """Plate name doesn't match any inner distribution's dims."""
```

Upstream numpyro / pyrox / JAX exceptions propagate unchanged.

## What this API explicitly does not include

- **arviz `InferenceData` converters.** `to_dataset` produces a
  plain xarray Dataset; arviz can ingest it. Promote to first-class
  if users ask.
- **MCMC / SVI wrappers.** Users call `MCMC(NUTS(model)).run(...)`
  directly; the bridge converts the inputs (DataArrays) and outputs
  (raw samples).
- **A `Distribution` base class users subclass to add new
  distributions.** Users who need a custom distribution write a
  numpyro `Distribution` and wrap with `to_numpyro`-compatible
  glue; we don't ship a custom-distribution kit.
- **Custom guides.** numpyro AutoGuides + pyrox `Parameterized`
  cover the guide story. The bridge doesn't reimplement guide
  construction.
