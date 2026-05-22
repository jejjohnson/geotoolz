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


__all__ = [
    "_as_numeric_with_mask",
    "_finite_filter",
    "_finite_mask",
    "_finite_mask_da",
]
