"""Shared dtype primitives: float promotion and fills a dtype can hold.

Raw sensor data usually arrives as integer digital numbers (``uint16``
for Sentinel-2 / Landsat). Doing band arithmetic in that dtype is wrong
in two ways: unsigned subtraction wraps around (``1000 - 3000`` becomes
``63536``) and signed addition overflows. :func:`as_float` is the one
promotion the arithmetic Tier-A primitives (indices, spectral,
radiometry) apply before touching their input.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from jaxtyping import Float, Num, Shaped
from numpy.typing import DTypeLike


__all__ = ["as_float", "fill_for", "holds_fill"]


def as_float(
    arr: Num[np.ndarray, "*dims"] | Shaped[np.ndarray, "*dims"],
    min_dtype: DTypeLike = np.float32,
) -> Float[np.ndarray, "*dims"]:
    """Promote integer / boolean ``arr`` to floating point before arithmetic.

    Floating-point (and complex) input is returned unchanged, without a
    copy. Anything else is cast to ``np.result_type(arr, min_dtype)``:
    with the default ``float32`` floor, ``(u)int8`` / ``(u)int16`` /
    ``bool`` become ``float32`` and wider integers become ``float64``.

    Args:
        arr: Any numeric array-like.
        min_dtype: Smallest float dtype integer input is promoted to.
            Default ``np.float32``; pass ``np.float64`` where the
            caller documents a ``float64`` result for integer input.

    JAX and dask arrays (an Array API array with a ``dtype``) stay on their
    backend, so a kernel can promote its input under ``jax.jit``; anything
    without a ``dtype`` goes through ``np.asarray`` first.

    Returns:
        A floating-point array with the same shape and values, on the
        input's backend.
    """
    if not hasattr(arr, "dtype") or isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.inexact):
            return arr
        return arr.astype(np.result_type(arr, min_dtype), copy=False)
    if np.issubdtype(arr.dtype, np.inexact):
        return arr
    return arr.astype(np.result_type(arr.dtype, min_dtype))


def holds_fill(dtype: DTypeLike, fill: Any) -> bool:
    """Whether ``dtype`` data can hold the fill value ``fill`` exactly.

    Mirrors the check `geoarray.GeoArray` makes at construction: float and
    complex data hold any fill (``NaN`` included); boolean data holds
    ``0`` / ``1``; integer data holds an integral fill within its range.
    ``None`` (no fill) fits every dtype.

    Example:
        >>> holds_fill(np.uint8, -9999), holds_fill(np.float32, np.nan)
        (False, True)
    """
    if fill is None:
        return True
    dtype = np.dtype(dtype)
    if dtype.kind not in "biu":
        return True
    value = fill.item() if isinstance(fill, np.generic) else fill
    if isinstance(value, float) and (math.isnan(value) or not value.is_integer()):
        return False
    if dtype.kind == "b":
        return value in (0, 1)
    info = np.iinfo(dtype.type)
    return bool(info.min <= value <= info.max)


def fill_for(dtype: DTypeLike, fill: Any) -> Any:
    """``fill`` when ``dtype`` holds it, else the canonical fill for ``dtype``.

    The canonical fills follow the output's meaning: ``False`` for a
    boolean (mask) output, whatever the input's fill, and ``0`` for an
    integer (label / count) output that cannot hold ``fill``. ``None``
    (no fill) stays ``None``.

    Example:
        >>> fill_for(np.uint8, np.nan), fill_for(bool, -9999), fill_for(np.int32, -9)
        (0, False, -9)
    """
    if fill is None:
        return None
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return False
    return fill if holds_fill(dtype, fill) else 0
