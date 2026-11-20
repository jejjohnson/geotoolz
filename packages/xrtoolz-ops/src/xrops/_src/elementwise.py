"""Apply a geophysix function to DataArrays, lazily on dask and keeping coordinates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr


def elementwise(func: Callable[..., Any], *args: Any, **kwargs: Any) -> xr.DataArray:
    """Apply an elementwise geophysix ``func`` to ``args``, broadcasting like xarray.

    geophysix functions take arrays and return arrays; this wraps one in
    `xarray.apply_ufunc` so a DataArray keeps its dims, coordinates and name,
    and a dask-backed one stays lazy (the function runs once per chunk).
    Attributes are dropped, as in xarray arithmetic.

    Args:
        func: A geophysix function of arrays, such as
            `geophysix.rotation.coriolis`.
        *args: DataArrays and plain numbers, in ``func``'s positional order.
        **kwargs: Keyword arguments for ``func`` (a record such as
            ``earth=``).

    Returns:
        The result, in the floating dtype of the array arguments (float64
        when there are none).
    """
    dtypes = [a.dtype for a in args if hasattr(a, "dtype")]
    dtype = np.result_type(*dtypes) if dtypes else np.dtype(float)
    if dtype.kind != "f":
        dtype = np.dtype(float)
    return xr.apply_ufunc(
        lambda *arrays: func(*arrays, **kwargs),
        *args,
        dask="parallelized",
        output_dtypes=[dtype],
    )
