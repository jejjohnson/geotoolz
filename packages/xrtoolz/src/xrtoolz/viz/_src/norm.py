"""Shared (vmin, vmax) helper for matched colour scales across panels."""

from __future__ import annotations

import xarray as xr


_FLAT_DIM = "_shared_norm_flat"


def _coerce_to_dataarray(a: xr.DataArray | xr.Dataset) -> xr.DataArray:
    if isinstance(a, xr.Dataset):
        if len(a.data_vars) != 1:
            raise ValueError(
                "shared_norm: Dataset inputs must have exactly one "
                f"data variable; got {list(a.data_vars)}."
            )
        (var,) = a.data_vars
        return a[var]
    return a


def _flatten(da: xr.DataArray) -> xr.DataArray:
    """Reshape ``da`` to a single 1-D dim ``_FLAT_DIM`` so a list of inputs
    with disparate shapes can be combined via :func:`xarray.concat` while
    keeping a dask backend lazy."""
    if not da.dims:
        return da.expand_dims(_FLAT_DIM)
    stacked = da.stack({_FLAT_DIM: da.dims})
    # Drop the multi-index that ``stack`` builds — concat needs a plain
    # dim, and we don't care about provenance once we're computing a
    # scalar quantile.
    return stacked.reset_index(_FLAT_DIM, drop=True)


def shared_norm(
    *arrays: xr.DataArray | xr.Dataset,
    q: tuple[float, float] | None = (0.02, 0.98),
    symmetric: bool = False,
) -> tuple[float, float]:
    """Compute matched ``(vmin, vmax)`` across multiple inputs.

    Useful for multi-panel comparison grids where the eye should not
    pick up colour-scale stretch artefacts as if they were structural
    differences in the data.

    Reductions go through xarray (``.quantile`` / ``.min`` /
    ``.max``) rather than ``np.quantile`` on a materialised
    ``np.concatenate``, so dask-backed inputs stay lazy: only the
    final scalar result is realised.

    Args:
        *arrays: One or more :class:`xr.DataArray` or
            :class:`xr.Dataset`. A single-variable Dataset is auto
            unwrapped; a multi-variable Dataset raises.
        q: ``(low, high)`` quantile pair in ``[0, 1]``. Default
            ``(0.02, 0.98)`` strips outliers. Pass ``None`` for the
            full ``(min, max)`` range.
        symmetric: When ``True``, return symmetric limits ``(-M, +M)``
            with ``M = max(|low|, |high|)``. Useful for divergent
            (signed-error) fields.

    Returns:
        ``(vmin, vmax)`` as Python floats. NaNs are ignored. Returns
        ``(nan, nan)`` if every input is fully NaN.
    """
    if not arrays:
        raise ValueError("shared_norm requires at least one input array.")
    if q is not None and not (0.0 <= q[0] <= q[1] <= 1.0):
        raise ValueError(f"q must satisfy 0 <= q[0] <= q[1] <= 1, got {q!r}.")

    pieces = [_flatten(_coerce_to_dataarray(a)) for a in arrays]
    combined = xr.concat(pieces, dim=_FLAT_DIM)
    # Bail out early when every value is NaN — the quantile / min / max
    # below would otherwise return NaN in a less obvious way.
    if int(combined.count().values) == 0:
        return (float("nan"), float("nan"))

    if q is None:
        lo = float(combined.min(skipna=True).values)
        hi = float(combined.max(skipna=True).values)
    else:
        qs = combined.quantile(list(q), skipna=True)
        lo = float(qs.isel(quantile=0).values)
        hi = float(qs.isel(quantile=1).values)

    if symmetric:
        m = max(abs(lo), abs(hi))
        return (-m, m)
    return (lo, hi)


__all__ = ["shared_norm"]
