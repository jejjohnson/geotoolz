"""Shared validation helpers.

Used cross-package — ``xr_toolz.interpolate``, ``xr_toolz.transforms``
(morphology), and metrics — for dim/coord presence checks, scalar/int
validation, boolean-mask validation, and IDW hyperparameter validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr


def _is_int_like(value: Any) -> bool:
    """True for built-in and numpy integer scalars, excluding booleans."""
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _require_dims(
    obj: xr.DataArray | xr.Dataset, *dims: str, name: str = "input"
) -> None:
    """Raise if any requested dimensions are absent."""
    missing = [dim for dim in dims if dim not in obj.dims]
    if missing:
        raise ValueError(
            f"{name} must have dims {tuple(dims)!r}; missing {tuple(missing)!r}"
        )


def _require_coords(
    obj: xr.DataArray | xr.Dataset,
    *coords: str,
    name: str = "input",
) -> None:
    """Raise if any requested coordinates are absent."""
    missing = [coord for coord in coords if coord not in obj.coords]
    if missing:
        raise ValueError(
            f"{name} must have coords {tuple(coords)!r}; missing {tuple(missing)!r}"
        )


def _validate_bool_mask(mask: xr.DataArray, lon: str, lat: str) -> None:
    """Validate a boolean mask carrying the requested spatial dimensions."""
    if mask.dtype != bool:
        raise TypeError(f"mask must be boolean, got {mask.dtype}")
    _require_dims(mask, lat, lon, name="mask")


def _validate_positive_int(value: Any, *, name: str) -> int:
    """Validate and normalize a positive integer scalar."""
    if not _is_int_like(value):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    int_value = int(value)
    if int_value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int_value


def _validate_coarsen_factor(factor: Mapping[str, int]) -> dict[str, int]:
    """Validate and normalize per-dimension positive integer factors."""
    factor_dict: dict[str, int] = {}
    for dim, value in factor.items():
        # Accept numpy integer types (np.int64 etc.) by reducing through __index__.
        try:
            int_value = (
                int(value.__index__()) if hasattr(value, "__index__") else int(value)
            )
            int_match = hasattr(value, "__index__")
        except (TypeError, ValueError):
            int_match = False
            int_value = 0
        if not int_match or int_value < 1 or isinstance(value, bool):
            raise ValueError(
                f"coarsen factor for {dim!r} must be a positive integer "
                f"(>= 1), got {value!r}."
            )
        factor_dict[dim] = int_value
    return factor_dict


def _validate_idw_args(
    k: int,
    power: float,
    metric: str,
    max_distance: float | None,
    eps: float,
) -> None:
    """Validate common inverse-distance-weighting hyperparameters."""
    if not _is_int_like(k) or int(k) < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if power < 0:
        raise ValueError(f"power must be non-negative, got {power}")
    if metric not in {"euclidean", "haversine"}:
        raise ValueError(f"metric must be 'euclidean' or 'haversine', got {metric!r}")
    if max_distance is not None and max_distance < 0:
        raise ValueError(f"max_distance must be non-negative, got {max_distance}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")


__all__ = [
    "_is_int_like",
    "_require_coords",
    "_require_dims",
    "_validate_bool_mask",
    "_validate_coarsen_factor",
    "_validate_idw_args",
    "_validate_positive_int",
]
