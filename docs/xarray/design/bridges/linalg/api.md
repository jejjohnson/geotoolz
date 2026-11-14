---
status: draft
version: 0.1.0
---

# xrtoolz.linalg — API

Proposed public surface.

## Imports

```python
import xrtoolz.linalg as xla            # Function API + NamedOperator
from xrtoolz.linalg import (            # Operator API
    Solve, Logdet, Cholesky, Diag, Trace, Inv, Sqrt,
    Sample, LogProb,
    NamedOperator, MultivariateNormal,
)
```

Importing `xrtoolz.linalg` does not import JAX or gaussx; the first
call into a primitive triggers the lazy import.

## `NamedOperator`

```python
@dataclass(frozen=True)
class NamedOperator:
    operator: lineax.AbstractLinearOperator
    dims: tuple[str, ...]
    coords: Mapping[str, xr.DataArray] | None = None

    # operator algebra
    def __add__(self, other: NamedOperator) -> NamedOperator: ...
    def __sub__(self, other: NamedOperator) -> NamedOperator: ...
    def __mul__(self, scalar: float | jax.Array) -> NamedOperator: ...
    def __matmul__(
        self, other: NamedOperator | xr.DataArray,
    ) -> NamedOperator | xr.DataArray: ...
    @property
    def T(self) -> NamedOperator: ...

    # materialisation
    def materialize(self) -> jax.Array:
        """Return the dense (N, N) matrix. Loses structure."""
    def to_dataarray(
        self, *, row_suffix: str = "_row", col_suffix: str = "_col",
    ) -> xr.DataArray:
        """Return a 2D labeled view of the operator.

        Row and column dims are renamed by appending ``row_suffix``
        and ``col_suffix`` to the original ``dims`` so the rendered
        matrix has distinct row/column labels.
        """

    # constructors
    @classmethod
    def from_dense(cls, da: xr.DataArray, *, row_dim: str, col_dim: str) -> NamedOperator: ...
    @classmethod
    def diagonal(cls, da: xr.DataArray) -> NamedOperator: ...
    @classmethod
    def kronecker(cls, *factors: NamedOperator) -> NamedOperator: ...
    @classmethod
    def block_diag(cls, *blocks: NamedOperator, dim: str) -> NamedOperator: ...
    @classmethod
    def low_rank(
        cls,
        U: xr.DataArray, *,
        rank_dim: str,
        D: NamedOperator | None = None,
        V: xr.DataArray | None = None,
    ) -> NamedOperator: ...
```

The class methods are convenience builders that map onto gaussx
constructors (`gaussx.Kronecker`, `gaussx.BlockDiag`,
`gaussx.LowRankUpdate`, ...) and assemble the right `dims` tuple.

## Layer-0 functions

### Solving and factorising

```python
def solve(
    K: NamedOperator, y: xr.DataArray, *,
    strategy: gaussx.AbstractSolveStrategy | None = None,
    batch_dims: tuple[str, ...] | None = None,
) -> xr.DataArray: ...

def logdet(
    K: NamedOperator, *,
    strategy: gaussx.AbstractLogdetStrategy | None = None,
) -> jax.Array: ...

def cholesky(K: NamedOperator) -> NamedOperator:
    """Return the lower-triangular factor as a NamedOperator."""

def sqrt(K: NamedOperator) -> NamedOperator:
    """Symmetric square root (matrix square root)."""

def inv(K: NamedOperator) -> NamedOperator:
    """Inverse operator. Prefer ``solve`` over materialising inv when possible."""
```

### Scalar / diagonal extractions

```python
def diag(K: NamedOperator) -> xr.DataArray:
    """Diagonal of ``K`` as a DataArray with ``K.dims``."""

def trace(K: NamedOperator) -> jax.Array: ...

def quadratic_form(K: NamedOperator, x: xr.DataArray) -> jax.Array:
    """x.T @ K @ x — scalar even for batched x."""
```

### Matrix-vector and matrix-matrix products

```python
def matvec(K: NamedOperator, y: xr.DataArray) -> xr.DataArray:
    """Apply ``K`` to a labeled vector. Same as ``K @ y``."""

def matmul(K: NamedOperator, M: NamedOperator | xr.DataArray) -> ...
```

### Spectral

```python
def eig(K: NamedOperator) -> tuple[xr.DataArray, NamedOperator]:
    """Returns (eigenvalues, eigenvectors-as-NamedOperator)."""

def eigvals(K: NamedOperator) -> xr.DataArray: ...

def svd(K: NamedOperator) -> tuple[NamedOperator, xr.DataArray, NamedOperator]: ...
```

### Sampling and log-prob

```python
def sample(
    K: NamedOperator, key: jax.Array, *,
    sample_shape: tuple[str, ...] = (),
    mean: xr.DataArray | None = None,
) -> xr.DataArray:
    """Draw from N(mean, K). New dims listed in ``sample_shape``
    (each a ``"name=size"`` token) prepend the output.
    """

def log_prob(
    K: NamedOperator, x: xr.DataArray, *,
    mean: xr.DataArray | None = None,
) -> jax.Array: ...
```

## `MultivariateNormal`

```python
class MultivariateNormal:
    def __init__(
        self,
        loc: xr.DataArray,
        cov: NamedOperator,
        *,
        solver: gaussx.AbstractSolveStrategy | None = None,
    ) -> None: ...

    def sample(
        self, key: jax.Array, *,
        sample_shape: tuple[str, ...] = (),
    ) -> xr.DataArray: ...

    def log_prob(self, x: xr.DataArray) -> jax.Array: ...
    def entropy(self) -> jax.Array: ...
    def mean(self) -> xr.DataArray: ...
    @property
    def covariance(self) -> NamedOperator: ...
```

`sample_shape` follows the same `"name=size"` convention used by
`sample()` above. NumPyro callers can pass a plain int tuple
(`sample_shape=(3,)`) — the bridge auto-names the resulting dim
`"sample_0"`, `"sample_1"`, ... and warns once per call site.

## Layer-1 operators

| Function                | Operator        | Notes                                                                                  |
| ----------------------- | --------------- | -------------------------------------------------------------------------------------- |
| `solve`                 | `Solve`         | `Solve(operator, rhs=..., strategy=...)`. `rhs` may be a string (var name on input Dataset) or a callable. |
| `logdet`                | `Logdet`        | Returns a 0-d DataArray for pipeline composition.                                       |
| `cholesky`              | `Cholesky`      | Returns a NamedOperator wrapped in a single-variable Dataset for thread-through.        |
| `diag`                  | `Diag`          |                                                                                        |
| `trace`                 | `Trace`         | 0-d DataArray output.                                                                  |
| `sqrt` / `inv`          | `Sqrt`, `Inv`   |                                                                                        |
| `sample`                | `Sample`        | `Sample(operator=..., key=..., sample_shape=("draw=10",))`.                            |
| `log_prob`              | `LogProb`       |                                                                                        |
| MVN.sample              | `MVNSample`     | Convenience wrapper.                                                                   |

Operators all accept `operator` as either a `NamedOperator` instance
*or* a `Callable[[xr.Dataset], NamedOperator]` so the operator can be
data-dependent (e.g. a learned kernel re-evaluated on each call).

## Exception hierarchy

```python
class LinAlgBridgeError(Exception):
    """Base for xrtoolz.linalg errors."""


class CoordMismatch(LinAlgBridgeError):
    """Operator coords disagree with input DataArray coords."""


class DimMismatch(LinAlgBridgeError, ValueError):
    """Operator dim names or sizes don't match the input."""


class BackendError(LinAlgBridgeError, TypeError):
    """Input DataArray wraps a non-JAX array."""
```

Upstream `gaussx` / `lineax` exceptions propagate unchanged.

## Pack / unpack helpers (publicly exposed)

```python
def pack(da: xr.DataArray, dims: tuple[str, ...]) -> tuple[jax.Array, Callable]: ...
def unpack(flat: jax.Array, restore: Callable) -> xr.DataArray: ...
```

Exposed so users can drop to gaussx directly when they need
something not in the bridge surface, without re-implementing the
flatten/reshape loop.

## What this API explicitly does not include

- **Recipe wrappers** (Kalman, RTS, Matheron, etc.) — deferred until
  the primitive surface is stable.
- **A `Gaussian` distribution mixin** that flows through both
  `xrtoolz.linalg` *and* `xrtoolz.prob`. Two distribution surfaces
  is one too many; `xrtoolz.prob.Normal` (numpyro-backed) and
  `xrtoolz.linalg.MultivariateNormal` (gaussx-backed) coexist with
  documented boundaries (see `prob/decisions.md` D4 — "Two MVN classes
  coexist").
- **Automatic strategy selection**. gaussx has `AutoSolver`; users
  who want auto-dispatch pass it explicitly. The bridge does not
  guess.
- **Backwards-compat with non-JAX arrays.** The error message
  points users at `jnp.asarray`; no silent conversion.
