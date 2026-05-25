---
status: draft
version: 0.1.0
---

# xrtoolz.linalg — Architecture

## Package layout

```
src/xrtoolz/linalg/
├── __init__.py            # Public re-exports (functions + Operators + NamedOperator)
├── _src/                  # Layer-0 pure functions
│   ├── named_operator.py  # NamedOperator — lineax op + dim labels
│   ├── reshape.py         # DataArray <-> JAX vector loop, lazy + label-aware
│   ├── primitives.py      # solve, logdet, cholesky, diag, trace, sqrt, inv
│   ├── sample.py          # sample, log_prob — distribution-level entry points
│   ├── distributions.py   # MultivariateNormal labeled wrapper
│   └── recipes.py         # Optional: labeled wrappers for kalman_filter, etc.
└── operators.py           # Layer-1: Solve, Logdet, Cholesky, Sample, …
```

## Layer stack

```
┌────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Operators                                                │
│    Solve(operator | factory, *, strategy=…)                         │
│    Logdet(operator | factory, *, strategy=…)                        │
│    Cholesky(operator | factory)                                     │
│    Sample(operator | distribution, key, sample_shape=("draw=N",))   │
│    LogProb(distribution)                                            │
│    Each subclasses xrtoolz.Operator; _apply delegates to Layer 0.   │
├────────────────────────────────────────────────────────────────────┤
│  Layer 0 — Pure functions                                           │
│    solve(K: NamedOperator, y: DataArray, …) -> DataArray            │
│    logdet(K) -> jax scalar (or 0-d DataArray)                       │
│    cholesky(K) -> NamedOperator                                     │
│    sample(K | dist, key, sample_shape=…) -> DataArray                │
│    log_prob(dist, x: DataArray) -> jax scalar                       │
├────────────────────────────────────────────────────────────────────┤
│  Labeled-operator + reshape adapter                                 │
│    NamedOperator: lineax op + dims tuple                            │
│    pack(da, dims) -> (jnp_vec, restore_fn)                          │
│    unpack(jnp_vec, restore_fn) -> da                                 │
├────────────────────────────────────────────────────────────────────┤
│  Backend                                                            │
│    gaussx (primitives, operators, strategies, distributions)        │
│    lineax (AbstractLinearOperator, solvers)                          │
│    matfree (transitively via gaussx)                                 │
│    JAX                                                              │
└────────────────────────────────────────────────────────────────────┘
```

## The core abstraction — `NamedOperator`

```python
@dataclass(frozen=True)
class NamedOperator:
    """A lineax linear operator labeled with dim names.

    Attributes:
        operator: any ``lineax.AbstractLinearOperator`` — including
            every gaussx structured operator (Kronecker, BlockDiag,
            LowRankUpdate, ...).
        dims: tuple of dim names that the operator's rows and columns
            address. The flat operator dimension is the product of
            the input DataArray's sizes along these dims, in order.

    Args:
        operator: the lineax operator. xrtoolz never inspects its
            structure — it forwards to gaussx primitives, which
            dispatch on type.
        dims: dim names, in the order the operator's flat index
            iterates them. Required.
        coords: optional per-dim coords. If supplied, ``solve``
            checks that the input DataArray's coords on ``dims``
            match. If absent, no coord check is performed (size match
            is still enforced).

    Example:
        K = NamedOperator(
            gaussx.Kronecker(K_lat, K_lon),
            dims=("lat", "lon"),
        )
    """

    operator: lineax.AbstractLinearOperator
    dims: tuple[str, ...]
    coords: Mapping[str, xr.DataArray] | None = None

    # Operator algebra forwards to gaussx via operator arithmetic.
    def __add__(self, other: "NamedOperator") -> "NamedOperator": ...
    def __matmul__(self, other: "NamedOperator | xr.DataArray") -> ...
    @property
    def T(self) -> "NamedOperator": ...
    def materialize(self) -> jax.Array: ...
```

`NamedOperator` is the type that flows between operators in the
linalg pipeline. It's frozen, PyTree-friendly (since `lineax`
operators are PyTrees), and never mutated.

## Pack / unpack loop

```python
def pack(da: xr.DataArray, dims: tuple[str, ...]) -> tuple[jax.Array, RestoreFn]:
    """Transpose so listed dims trail, flatten them into one axis.

    Returns:
        jax_array: shape ``(*batch_shape, prod(sizes_along_dims))``.
        restore: function that takes a flat jax array of the same
            trailing-shape and returns a DataArray with the original
            dims, coords, and name.
    """


def unpack(flat: jax.Array, restore: RestoreFn) -> xr.DataArray: ...
```

Notes:

- The reshape is dim-name-driven; the trailing-axis ordering matches
  `NamedOperator.dims`.
- Batch dims (anything in `da.dims` not in `dims`) are preserved as
  leading axes. `gaussx.solve` is `vmap`-friendly, so the batched
  call is automatic.
- `restore` closes over the original dims order, sizes, and coords so
  the round-trip is lossless modulo the math.

## Solver-strategy pass-through

```python
def solve(
    K: NamedOperator,
    y: xr.DataArray,
    *,
    strategy: gaussx._strategies.AbstractSolveStrategy | None = None,
    batch_dims: tuple[str, ...] | None = None,
) -> xr.DataArray:
    """Solve K @ x = y with x, y labeled DataArrays.

    Args:
        K: the labeled operator.
        y: rhs DataArray. Must carry every dim in ``K.dims``;
            additional dims become batch axes.
        strategy: gaussx solver strategy. Forwarded as-is to
            ``gaussx.solve``. If omitted, gaussx picks its default.
        batch_dims: optional explicit list of batch dim names. If
            omitted, batch dims are inferred as ``[d for d in y.dims
            if d not in K.dims]`` — preserving the original axis order
            of ``y`` so reshape and coord restore stay deterministic
            when ``y`` carries several batch dims.

    Returns:
        DataArray with the same dims as ``y`` (post any
        transposition) and forwarded coords.
    """
```

The strategy argument is the bridge's only concession to gaussx's
solver landscape. Anything gaussx accepts here works; we never
intercept or transform the strategy.

## Distribution surface

```python
class MultivariateNormal:
    """Labeled MVN over a labeled covariance.

    Wraps ``gaussx._distributions.MultivariateNormal``. Loc is a
    DataArray; covariance is a NamedOperator; samples and log-probs
    accept / return DataArrays.

    Args:
        loc: mean DataArray. Must carry exactly the operator's dims
            (plus optional batch dims).
        cov: NamedOperator. Its ``dims`` must match the trailing dims
            of ``loc``.
        solver: optional gaussx solver strategy for log_prob.

    Methods:
        sample(key, sample_shape=("draw=N",)) -> DataArray
        log_prob(x: DataArray) -> jax scalar
        entropy() -> jax scalar
        mean() -> DataArray
    """
```

`sample_shape` accepts either an int tuple (numpyro convention) or a
sequence of `"<name>=<size>"` tokens that the bridge parses into
new named dims on the output.

## Recipe wrappers (optional, future)

gaussx has higher-level recipes — `kalman_filter`, `rts_smoother`,
`kronecker_mll`, `love_variance`, `matheron_update`. These could be
wrapped in `xrtoolz.linalg.recipes` with DataArray IO. Defer to a
follow-up; the v1 surface is primitives + distributions only.

## Operator class skeleton

```python
class Solve(Operator):
    """Solve a labeled linear system in a Sequential / Graph."""

    def __init__(
        self,
        operator: NamedOperator | Callable[[xr.Dataset], NamedOperator],
        *,
        rhs: str | Callable[[xr.Dataset], xr.DataArray] | None = None,
        strategy: gaussx._strategies.AbstractSolveStrategy | None = None,
    ) -> None:
        self.operator = operator
        self.rhs = rhs
        self.strategy = strategy

    def _apply(self, ds: xr.Dataset) -> xr.DataArray | xr.Dataset:
        K = self.operator(ds) if callable(self.operator) else self.operator
        y = self._resolve_rhs(ds)
        from xrtoolz.linalg._src.primitives import solve
        return solve(K, y, strategy=self.strategy)
```

The `operator | Callable[Dataset, NamedOperator]` polymorphism lets
the user either pre-build `K` once or compute it from the threaded
Dataset (e.g. a learned-length-scale kernel that re-evaluates per
call).

## Dependencies

| Dep             | Version | Purpose                                                     |
| --------------- | ------- | ----------------------------------------------------------- |
| `gaussx`        | latest  | Primitives, operators, strategies, distributions, recipes.  |
| `lineax`        | latest  | Base operator abstraction (transitive via gaussx).          |
| `matfree`       | latest  | Stochastic linalg (transitive via gaussx).                  |
| `jax`, `jaxlib` | latest  | Array backend.                                              |

All lazy-imported inside Layer-0 functions. `import xrtoolz` does not
import JAX or gaussx.

## Error policy

| Condition                                                     | Behaviour                                                                                                                |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Input DataArray wraps numpy (not JAX)                          | Raise `TypeError("xrtoolz.linalg requires JAX-backed DataArrays; rebuild the DataArray with JAX-backed data, e.g. `xr.DataArray(jnp.asarray(da.data), dims=da.dims, coords=da.coords)`")`. |
| Missing dim on input that `K.dims` references                  | Raise `KeyError` naming the missing dim.                                                                                  |
| Coord on shared dim differs from `NamedOperator.coords`        | Raise `CoordMismatch` (same exception type as `xrtoolz.einx`).                                                            |
| `K.dims` size disagrees with input dim size                    | Raise `ValueError` showing the mismatch.                                                                                  |
| gaussx solver fails (e.g. non-PSD operator into Cholesky)      | Propagate the gaussx exception unchanged.                                                                                 |

## Integration with existing xrtoolz

- `xrtoolz.einx` outputs feed naturally into `xrtoolz.linalg`:
  `Sequential([Rearrange(...), Cholesky(), Solve(rhs=...)])`.
- `xrtoolz.combinators.Augment` works since `Solve._apply` can
  return a `Dataset`-with-one-new-variable when wrapped accordingly.
- `xrtoolz.signature.Signature` propagates through linalg operators:
  `compute_output_signature` on `Solve(K)` reads `K.dims` and the
  input signature to predict the output dims.
