---
status: draft
version: 0.1.0
---

# xrtoolz.einx — Architecture

## Package layout

```
src/xrtoolz/einx/
├── __init__.py         # Re-exports the public surface (functions + Operators)
├── _src/               # Layer-0 pure functions
│   ├── core.py         # einsum, rearrange, reduce, repeat — DataArray in/out
│   ├── linalg.py       # matmul, outer, batch_matmul, kron — convenience over einsum
│   ├── coords.py       # Coord forwarding + dim-name reconciliation helpers
│   └── _pattern.py     # Parse einx patterns into our internal AST, validate dims
├── operators.py        # Layer-1 Operators: Einsum, Rearrange, Reduce, Repeat, …
└── dataset.py          # pack_dataset / unpack_dataset — multi-variable convenience
```

Mirrors the existing `xrtoolz.geo._src` / `xrtoolz.geo.operators.py`
split.

## Layer stack

```
┌────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Operators                                                │
│    Einsum(pattern, **kwargs)                                        │
│    Rearrange(pattern, **kwargs)                                     │
│    Reduce(pattern, op, **kwargs)                                    │
│    Repeat(pattern, **kwargs)                                        │
│    Each subclasses xrtoolz.Operator; _apply delegates to Layer 0.   │
├────────────────────────────────────────────────────────────────────┤
│  Layer 0 — Pure functions                                           │
│    einsum(pattern, *das, **kwargs) -> DataArray                     │
│    rearrange(pattern, da, **kwargs) -> DataArray                    │
│    reduce(pattern, da, *, op, **kwargs) -> DataArray                │
│    repeat(pattern, da, **kwargs) -> DataArray                       │
│    Plus linalg conveniences: matmul, outer, batch_matmul.           │
├────────────────────────────────────────────────────────────────────┤
│  Carrier adapter                                                    │
│    DataArray <-> backend array (numpy / jax / dask / torch)         │
│    Handled per-call: extract .values, run einx, wrap result.        │
├────────────────────────────────────────────────────────────────────┤
│  Backend                                                            │
│    einx + whatever array library the DataArray wraps                │
└────────────────────────────────────────────────────────────────────┘
```

## Call lifecycle

A representative call walks through these steps:

```text
xnx.einsum("time lat lon, lat lon -> time", da, w)

  1. Parse pattern -> inputs=[["time","lat","lon"], ["lat","lon"]],
                       output=["time"], reduced={"lat","lon"}.
  2. For each input DataArray:
       - assert that every dim in its slot is a real dim on the input
         (clear error otherwise);
       - transpose to match the pattern slot's order;
       - if shared dims have misaligned coords, raise (or align if the
         caller passed align=True).
  3. Stack the values arrays: arr_i = da_i.transpose(...).values.
  4. Call einx.einsum(pattern, *arr_i, **kwargs).
  5. Build output coords:
       - for each output dim, take coords from the first input that
         carries them;
       - drop reduced dims;
       - any pattern-only dims (e.g. expanded via kwargs) get no coords
         unless `coords={...}` was passed.
  6. Wrap result in DataArray(values, dims=output, coords=...).
```

The shape of this flow is identical for `rearrange` / `reduce` /
`repeat` — only the einx call differs.

## Coord-forwarding helper

`coords.py` exposes a single helper:

```python
def forward_coords(
    inputs: list[xr.DataArray],
    output_dims: tuple[str, ...],
    *,
    extra: Mapping[str, xr.DataArray] | None = None,
) -> dict[str, xr.DataArray]:
    """Build the coord dict for the output of a labeled einx call.

    Args:
        inputs: input DataArrays in pattern order.
        output_dims: dim names of the output, in result order.
        extra: explicit coord overrides supplied by the caller.

    Returns:
        Coord dict suitable for ``xr.DataArray(..., coords=coords)``.

    Rules:
        - For each output dim, the first input carrying it donates its
          coord. Multi-coord conflicts raise unless they're equal.
        - ``extra`` wins over inferred coords.
        - Output dims not present on any input get coords only if
          ``extra`` provides them; otherwise they remain unindexed.
    """
```

Every Layer-0 function uses this helper for output construction.

## Operator class skeleton

Every Layer-1 operator follows the same template:

```python
class Einsum(Operator):
    """Layer-1 wrapper around xrtoolz.einx.einsum."""

    def __init__(self, pattern: str, **kwargs: Any) -> None:
        self.pattern = pattern
        self.kwargs = dict(kwargs)

    def _apply(self, *das: xr.DataArray) -> xr.DataArray:
        from xrtoolz.einx._src.core import einsum

        return einsum(self.pattern, *das, **self.kwargs)

    def get_config(self) -> dict[str, Any]:
        return {"pattern": self.pattern, **self.kwargs}

    def __repr__(self) -> str:
        return f"Einsum({self.pattern!r})"

    def compute_output_signature(
        self, *sigs: Signature
    ) -> Signature:
        from xrtoolz.einx._src._pattern import infer_output_signature

        return infer_output_signature(self.pattern, sigs, self.kwargs)
```

Notable: `compute_output_signature` parses the pattern and threads dim
sizes from the input signatures, so `Sequential.summary()` /
`Graph.summary()` render shapes without executing data paths — the
same trick used by `xrtoolz.geo` operators.

## Dataset helpers

`dataset.py` adds two thin functions:

```python
def pack_dataset(
    ds: xr.Dataset,
    variables: Sequence[str] | None = None,
    *,
    new_dim: str = "variable",
) -> xr.DataArray:
    """Stack named variables of a Dataset along a new dim.

    The variables must share dims and coords; the helper assembles
    them into a single DataArray with one extra dim whose coord is
    the variable names. Used as input to einx operations that want a
    'channels' axis.
    """


def unpack_dataset(
    da: xr.DataArray,
    *,
    dim: str = "variable",
) -> xr.Dataset:
    """Inverse of pack_dataset.

    Takes a DataArray with a 'variable'-style dim whose coord is a
    sequence of strings and returns a Dataset with each slice as a
    named variable.
    """
```

Both are pure xarray; they don't touch einx. They live here because the
canonical einx use case (`channels` axis) is the canonical reason a
user would pack/unpack.

## Signature inference

`_pattern.py` exposes:

```python
def parse_pattern(pattern: str) -> EinxPattern: ...
def infer_output_signature(
    pattern: str,
    input_sigs: tuple[Signature, ...],
    kwargs: Mapping[str, Any],
) -> Signature: ...
```

The parser is intentionally light — we delegate the *semantics* of the
pattern to einx itself (calling its parser at apply time). Our parser
only needs to know which dim names appear on which side so it can:

- check inputs carry the right dims,
- determine output dim names + order,
- pull sizes from input signatures (when known) or kwargs.

## Dependencies

| Dep    | Version | Purpose                                                    |
| ------ | ------- | ---------------------------------------------------------- |
| `einx` | latest  | Named-axis algebra. Imported lazily inside Layer-0 funcs.   |
| `xarray`, `numpy` | inherited | Already xrtoolz core deps. |

No JAX requirement here — einx works on numpy / dask / jax / torch.
This is the lightest of the three bridges.

## Integration with existing xrtoolz

| Hook                           | Behaviour                                                                          |
| ------------------------------ | ---------------------------------------------------------------------------------- |
| `xrtoolz.Operator` dispatch    | Operators inherit DataTree leaf-wise dispatch unchanged.                            |
| `xrtoolz.Sequential` / `Graph` | Operators compose naturally; `summary()` shows dim shapes via `Signature`.          |
| `xrtoolz.combinators.Augment`  | `Augment(Einsum(...))` merges the einsum result back into a Dataset by name.        |
| `xrtoolz.signature.Signature`  | `compute_output_signature` on each operator threads dims through pipeline summary.  |

## Non-architectural notes

- The module sits next to existing top-level submodules
  (`xrtoolz.geo`, `xrtoolz.ocn`, ...). It is *not* nested under
  `xrtoolz.transforms` even though it does transform-like work —
  einx is a backend dependency, not an internal transform family.
- The single dependency makes this the natural "first" of the three
  bridges to land. Patterns settled here inform `linalg` and `prob`.
