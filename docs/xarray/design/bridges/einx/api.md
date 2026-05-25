---
status: draft
version: 0.1.0
---

# xrtoolz.einx — API

Proposed public surface. Signatures are draft; implementations have not
landed.

## Imports

```python
import xrtoolz.einx as xnx                # Function API
from xrtoolz.einx import (                # Operator API
    Einsum, Rearrange, Reduce, Repeat,
    Matmul, Outer, BatchMatmul,
    pack_dataset, unpack_dataset,
)
```

`import xrtoolz` does **not** pull in einx. The first call into any
`xrtoolz.einx` function triggers the lazy import.

## Layer-0 functions

### `einsum`

```python
def einsum(
    pattern: str,
    *arrays: xr.DataArray,
    coords: Mapping[str, xr.DataArray] | None = None,
    align: bool = False,
    **shape_kwargs: int,
) -> xr.DataArray:
    """Labeled einx einsum.

    Args:
        pattern: einx pattern. Each axis token is a DataArray dim
            name; parentheses, ellipsis, and bracket syntax follow
            einx's rules.
        *arrays: input DataArrays, one per pattern input. Each
            input's slot in the pattern must reference the array's
            existing dim names; xrtoolz transposes to match before
            dispatch.
        coords: optional explicit coords for output dims that aren't
            on any input. Keys must be subsets of the pattern's
            output dims.
        align: if False (default), shared dims with mismatched
            coords raise a ``CoordMismatch`` error. If True, the
            shared dims are realigned with xarray's default
            inner-join semantics before dispatch.
        **shape_kwargs: dim sizes for any pattern axis whose size
            cannot be inferred from inputs (e.g. block-decomposition
            sub-axes).

    Returns:
        DataArray with dims given by the pattern's output slot, in
        result order. Coords are forwarded from the first input
        carrying each surviving dim; ``coords`` overrides take
        precedence.

    Raises:
        KeyError: a pattern axis names a dim absent from the
            corresponding input.
        CoordMismatch: shared dims have unequal coords and
            ``align=False``.
        ValueError: pattern is malformed or a pattern axis has
            no size after considering inputs + shape_kwargs.

    Example:
        >>> # Project (time, lat, lon) field through a (lat, lon) mask:
        >>> total = xnx.einsum(
        ...     "time lat lon, lat lon -> time",
        ...     field, mask,
        ... )
    """
```

### `rearrange`

```python
def rearrange(
    pattern: str,
    da: xr.DataArray,
    coords: Mapping[str, xr.DataArray] | None = None,
    **shape_kwargs: int,
) -> xr.DataArray:
    """Labeled einx rearrange.

    Performs reshape / transpose / merge / split operations described
    by ``pattern``. Coords on surviving output dims are forwarded;
    new dims created by splitting an input dim share that dim's coord
    sliced to the split sizes (which is rarely what you want — pass
    ``coords=`` to override).

    Example:
        >>> patches = xnx.rearrange(
        ...     field,
        ...     "time (lat_blk lat_in) (lon_blk lon_in) "
        ...     "-> time (lat_blk lon_blk) lat_in lon_in",
        ...     lat_in=4, lon_in=4,
        ... )
    """
```

### `reduce`

```python
def reduce(
    pattern: str,
    da: xr.DataArray,
    *,
    op: Callable | str,
    **shape_kwargs: int,
) -> xr.DataArray:
    """Reduce over named axes.

    Args:
        pattern: einx pattern. Reduced dims appear on the input slot
            and not on the output slot.
        da: input DataArray.
        op: reduction op. Strings 'sum'/'mean'/'min'/'max'/'median'/
            'std'/'var' resolve to the backend's numpy-API
            equivalent. Pass a callable for anything else.

    Example:
        >>> climatology = xnx.reduce(
        ...     sst,
        ...     "time month lat lon -> month lat lon",
        ...     op="mean",
        ... )
    """
```

### `repeat`

```python
def repeat(
    pattern: str,
    da: xr.DataArray,
    coords: Mapping[str, xr.DataArray] | None = None,
    **shape_kwargs: int,
) -> xr.DataArray:
    """Broadcast / tile along named axes.

    Example:
        >>> # Replicate a (lat, lon) climatology across 12 months:
        >>> seasonal = xnx.repeat(
        ...     mean_field,
        ...     "lat lon -> month lat lon",
        ...     month=12,
        ... )
    """
```

### Linear-algebra conveniences

Built on top of `einsum`:

```python
def matmul(a: xr.DataArray, b: xr.DataArray, *, dim: str) -> xr.DataArray:
    """Contract a and b along the shared ``dim``."""


def outer(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """Outer product. Output carries both inputs' dims."""


def batch_matmul(
    a: xr.DataArray, b: xr.DataArray, *, dim: str, batch_dims: Sequence[str] = (),
) -> xr.DataArray:
    """matmul broadcast over ``batch_dims``."""
```

These are explicitly *not* a replacement for `xarray.dot`; they
exist to keep einx-style pipelines self-contained without users
having to write a five-token einsum for a one-line operation.

## Layer-1 operators

Each function has a corresponding `Operator` subclass:

| Function       | Operator       | Notes                                                |
| -------------- | -------------- | ---------------------------------------------------- |
| `einsum`       | `Einsum`       | Multi-input (variadic); supports `Graph` fan-in.     |
| `rearrange`    | `Rearrange`    | Single-input.                                        |
| `reduce`       | `Reduce`       | Single-input; carries `op`.                          |
| `repeat`       | `Repeat`       | Single-input.                                        |
| `matmul`       | `Matmul`       | Two-input.                                           |
| `outer`        | `Outer`        | Two-input.                                           |
| `batch_matmul` | `BatchMatmul`  | Two-input.                                           |

Construction mirrors the function signature: `Einsum(pattern,
**shape_kwargs)`. `_apply` accepts the runtime DataArrays.

## Dataset helpers

```python
def pack_dataset(
    ds: xr.Dataset,
    variables: Sequence[str] | None = None,
    *,
    new_dim: str = "variable",
) -> xr.DataArray:
    """Stack named variables along a new 'variable'-style dim."""


def unpack_dataset(
    da: xr.DataArray,
    *,
    dim: str = "variable",
) -> xr.Dataset:
    """Split a 'variable'-style dim back into a Dataset."""
```

Inverses of one another for compatible inputs (all variables sharing
dims and coords).

## Exception hierarchy

```python
class EinxBridgeError(Exception):
    """Base for xrtoolz.einx errors."""


class CoordMismatch(EinxBridgeError):
    """Shared dims have unequal coords and ``align=False``."""


class PatternError(EinxBridgeError, ValueError):
    """Pattern is malformed or references dims not on the inputs."""
```

Raised in addition to upstream einx / xarray errors when the bridge
itself catches a problem before dispatching.

## What this API explicitly does not include

- **`einx.{vmap, map, dot, get_at, set_at, ...}`** — first cut covers
  the four pattern verbs (`einsum`, `rearrange`, `reduce`, `repeat`)
  + matmul conveniences. The remaining einx surface lands later under
  the same labeled-dim convention.
- **Dask-graph optimisation.** Calls go straight through to einx; if
  the underlying array is dask-backed, we get dask laziness for free.
  We do not rewrite einx patterns into dask blockwise calls.
- **JAX `pmap` / device-placement helpers.** Out of scope; users who
  need them call einx directly with `xnx.unpack_dataset`-style
  helpers as bookends.
