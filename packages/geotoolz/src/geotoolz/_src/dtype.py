"""Shared dtype-promotion primitive.

Raw sensor data usually arrives as integer digital numbers (``uint16``
for Sentinel-2 / Landsat). Doing band arithmetic in that dtype is wrong
in two ways: unsigned subtraction wraps around (``1000 - 3000`` becomes
``63536``) and signed addition overflows. :func:`as_float` is the one
promotion the arithmetic Tier-A primitives (indices, spectral,
radiometry) apply before touching their input.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Num, Shaped
from numpy.typing import DTypeLike


__all__ = ["as_float"]


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

    Returns:
        A floating-point ndarray with the same shape and values.
    """
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.inexact):
        return arr
    return arr.astype(np.result_type(arr, min_dtype), copy=False)
