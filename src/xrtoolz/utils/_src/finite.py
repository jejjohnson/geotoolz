"""Shared finite-value helpers.

Consumed across ``xrtoolz.interpolate``, ``xrtoolz.transforms``, and
any other module that needs to filter or mask non-finite samples.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike


def _finite_mask(*arrays: ArrayLike) -> np.ndarray:
    """Return the elementwise finite mask shared by all arrays."""
    if not arrays:
        raise ValueError("_finite_mask requires at least one array")
    mask = np.ones(np.asarray(arrays[0]).shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(np.asarray(array))
    return mask


def _finite_filter(*arrays: ArrayLike) -> tuple[np.ndarray, ...]:
    """Return arrays filtered to positions where every input is finite."""
    flat = tuple(np.ravel(np.asarray(array)) for array in arrays)
    mask = _finite_mask(*flat)
    return tuple(array[mask] for array in flat)


def _finite_mask_da(da: xr.DataArray) -> xr.DataArray:
    """Return an xarray-aware finite mask."""
    return xr.apply_ufunc(np.isfinite, da, dask="allowed")


def _as_numeric_with_mask(values: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert values to float and return the finite mask."""
    arr = np.asarray(values, dtype=float)
    return arr, np.isfinite(arr)


def _floating_output_dtype(da: xr.DataArray | xr.Dataset) -> np.dtype:
    """Floating output dtype for a per-slice kernel, for a DataArray or Dataset.

    ``apply_ufunc(dask="parallelized")`` needs a single ``output_dtypes`` entry
    even when it maps over a Dataset's variables. Kernels that always yield
    floating output (gap-fill, smoothing) use this to pick the common floating
    type across variables, falling back to ``float64`` for integer inputs.

    Args:
        da: The input the kernel was called with.

    Returns:
        A floating numpy dtype suitable for ``output_dtypes``.
    """
    if isinstance(da, xr.Dataset):
        dtypes = [v.dtype for v in da.data_vars.values()]
        base = np.result_type(*dtypes) if dtypes else np.dtype(np.float64)
    else:
        base = da.dtype
    return base if np.issubdtype(base, np.floating) else np.dtype(np.float64)


__all__ = [
    "_as_numeric_with_mask",
    "_finite_filter",
    "_finite_mask",
    "_finite_mask_da",
    "_floating_output_dtype",
]
