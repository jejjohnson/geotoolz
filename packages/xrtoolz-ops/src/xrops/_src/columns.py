"""Run a kernel of columns along one dim of DataArrays.

diagnostix's column functions take plain arrays with the levels last.
`on_columns` names the levels, broadcasts the other dims and puts the levels
back; on dask a column becomes one chunk, and nothing is computed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from xrops.grad._src import cartesian


def positional(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Drop every coordinate on ``dim``, so fields along it align by position."""
    return da.drop_vars([name for name, c in da.coords.items() if dim in c.dims])


def levels_of(da: xr.DataArray, dim: str) -> xr.DataArray:
    """The 1-D coordinate of ``dim``, as floats of the field's precision.

    Raises:
        ValueError: If ``dim`` is missing, or its coordinate is missing, not
            1-D, of fewer than two levels or not strictly monotonic.
    """
    if dim not in da.dims:
        raise ValueError(
            f"Dimension {dim!r} not present on DataArray with dims={da.dims}."
        )
    cartesian._require_coord(da, dim)
    cartesian._require_1d_coord(da[dim])
    values = np.asarray(da[dim].values, dtype=np.float64)
    gaps = np.diff(values)
    if gaps.size == 0 or not (np.all(gaps > 0) or np.all(gaps < 0)):
        raise ValueError(
            f"Coordinate {dim!r} must be strictly monotonic, with two levels or "
            f"more; got {values.size} level(s)."
        )
    dtype = da.dtype if np.issubdtype(da.dtype, np.floating) else np.float64
    return xr.DataArray(values.astype(dtype), dims=(dim,), name=dim)


def on_columns(
    kernel: Callable[..., Any], *fields: xr.DataArray, dim: str, reduces: bool
) -> xr.DataArray:
    """``kernel(*arrays)`` with the levels of each field on its last axis.

    Args:
        kernel: Takes arrays that end in the levels, and returns one array, with
            them or without.
        *fields: The fields, each on ``dim``, where they align by position.
        dim: The dim of the levels.
        reduces: Whether the kernel's result has no levels.

    Returns:
        The result, without a name or attributes, with the dims in the order
        they come in and the levels' coordinates where it has levels.
    """
    first = fields[0]
    out = xr.apply_ufunc(
        kernel,
        *(positional(field, dim) for field in fields),
        input_core_dims=[[dim]] * len(fields),
        output_core_dims=[[] if reduces else [dim]],
        dask="parallelized",
        output_dtypes=[np.result_type(*(f.dtype for f in fields), np.float32)],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    if reduces:
        return out
    order = dict.fromkeys(d for field in fields for d in field.dims)
    on_levels = {k: c for k, c in first.coords.items() if dim in c.dims}
    return out.transpose(*order).assign_coords(on_levels)
