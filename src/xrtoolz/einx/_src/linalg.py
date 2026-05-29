"""Linear-algebra conveniences built on :func:`xrtoolz.einx.einsum`.

These are thin, readable wrappers over a single ``einsum`` call — they
exist so einx-style pipelines stay self-contained without users writing
a five-token pattern for a one-line operation. They are explicitly *not*
a replacement for :func:`xarray.dot`.
"""

from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from xrtoolz.einx._src.core import einsum
from xrtoolz.einx._src.errors import PatternError


def matmul(a: xr.DataArray, b: xr.DataArray, *, dim: str) -> xr.DataArray:
    """Contract ``a`` and ``b`` along the shared ``dim``.

    The output carries every dim of ``a`` and ``b`` except ``dim``,
    with ``a``'s remaining dims first.

    Example:
        >>> # (time, k) · (k, mode) -> (time, mode)
        >>> scores = matmul(field, basis, dim="k")
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
        >>> weights = outer(lat_weights, lon_weights)
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
        >>> # contract 'k', broadcast over shared 'ensemble'
        >>> out = batch_matmul(a, b, dim="k", batch_dims=["ensemble"])
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
