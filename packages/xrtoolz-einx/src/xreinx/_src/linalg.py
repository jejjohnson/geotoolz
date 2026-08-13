"""Linear-algebra conveniences built on :func:`xreinx.einsum`.

These are thin, readable wrappers over a single ``einsum`` call — they
exist so einx-style pipelines stay self-contained without users writing
a five-token pattern for a one-line operation. They are explicitly *not*
a replacement for :func:`xarray.dot`.
"""

from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from xreinx._src.core import einsum
from xreinx._src.errors import PatternError


def matmul(a: xr.DataArray, b: xr.DataArray, *, dim: str) -> xr.DataArray:
    """Contract ``a`` and ``b`` along the shared ``dim``.

    The output carries every dim of ``a`` and ``b`` except ``dim``,
    with ``a``'s remaining dims first.

    Example:
        Project a time series onto a basis — ``(time, k) · (k, mode)``:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> field = xr.DataArray(np.ones((5, 3)), dims=("time", "k"))
        >>> basis = xr.DataArray(np.ones((3, 2)), dims=("k", "mode"))
        >>> scores = matmul(field, basis, dim="k")
        >>> scores.dims, scores.shape
        (('time', 'mode'), (5, 2))

        ```
    """
    if dim not in a.dims or dim not in b.dims:
        raise PatternError(
            f"matmul dim {dim!r} must be present on both inputs; got "
            f"{tuple(a.dims)} and {tuple(b.dims)}."
        )
    a_rest = [str(d) for d in a.dims if d != dim]
    b_rest = [str(d) for d in b.dims if d != dim]
    overlap = set(a_rest) & set(b_rest)
    if overlap:
        raise PatternError(
            f"matmul non-contracted dims must be disjoint; both inputs carry "
            f"{sorted(overlap)}. Use einsum for a custom contraction."
        )
    pattern = (
        f"{' '.join([*a_rest, dim])}, {' '.join([*b_rest, dim])} "
        f"-> {' '.join([*a_rest, *b_rest])}"
    )
    return einsum(pattern, a, b)


def outer(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """Outer product. Output carries ``a``'s dims then ``b``'s dims.

    Example:
        Build a 2-D weight grid from per-axis weights:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> lat_weights = xr.DataArray(np.ones(3), dims="lat")
        >>> lon_weights = xr.DataArray(np.ones(4), dims="lon")
        >>> outer(lat_weights, lon_weights).dims
        ('lat', 'lon')

        ```
    """
    overlap = set(a.dims) & set(b.dims)
    if overlap:
        raise PatternError(
            f"outer requires disjoint dims; inputs share {sorted(overlap)}."
        )
    pattern = (
        f"{' '.join(map(str, a.dims))}, {' '.join(map(str, b.dims))} "
        f"-> {' '.join(map(str, (*a.dims, *b.dims)))}"
    )
    return einsum(pattern, a, b)


def batch_matmul(
    a: xr.DataArray,
    b: xr.DataArray,
    *,
    dim: str,
    batch_dims: Sequence[str] = (),
) -> xr.DataArray:
    """``matmul`` along ``dim`` broadcast over shared ``batch_dims``.

    Example:
        Contract ``k`` while broadcasting over a shared ``ensemble`` dim:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> a = xr.DataArray(np.ones((2, 5, 3)), dims=("ensemble", "time", "k"))
        >>> b = xr.DataArray(np.ones((2, 3, 4)), dims=("ensemble", "k", "mode"))
        >>> out = batch_matmul(a, b, dim="k", batch_dims=["ensemble"])
        >>> out.dims, out.shape
        (('ensemble', 'time', 'mode'), (2, 5, 4))

        ```
    """
    batch = list(batch_dims)
    for name in (dim, *batch):
        if name not in a.dims or name not in b.dims:
            raise PatternError(
                f"batch_matmul requires {name!r} on both inputs; got "
                f"{tuple(a.dims)} and {tuple(b.dims)}."
            )
    a_rest = [str(d) for d in a.dims if d != dim and d not in batch]
    b_rest = [str(d) for d in b.dims if d != dim and d not in batch]
    overlap = set(a_rest) & set(b_rest)
    if overlap:
        raise PatternError(
            f"batch_matmul non-contracted, non-batch dims must be disjoint; "
            f"both inputs carry {sorted(overlap)}."
        )
    pattern = (
        f"{' '.join([*batch, *a_rest, dim])}, {' '.join([*batch, *b_rest, dim])} "
        f"-> {' '.join([*batch, *a_rest, *b_rest])}"
    )
    return einsum(pattern, a, b)


__all__ = ["batch_matmul", "matmul", "outer"]
