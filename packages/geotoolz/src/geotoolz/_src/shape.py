"""Shared shape-normalisation primitives.

Several operator families (plume, mask, measure, segment, feature, viz)
accept "a single-band map" — either a 2-D ``(H, W)`` array or a
singleton-band ``(1, H, W)`` cube — and need the 2-D view for scipy /
skimage calls. :func:`single_band` is the one implementation of that
squeeze, replacing the per-module copies.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Shaped


__all__ = ["single_band"]


def single_band(
    values: Shaped[np.ndarray, "h w"] | Shaped[np.ndarray, "1 h w"],
    *,
    name: str = "input",
) -> Shaped[np.ndarray, "h w"]:
    """Return the 2-D view of a single-band array.

    Args:
        values: A ``(H, W)`` array, or a ``(1, H, W)`` cube whose leading
            band axis is squeezed away. Any array-like is accepted.
        name: Label used in the error message so callers can attribute
            the failure to their own operator (e.g. ``"Otsu"``).

    Returns:
        The ``(H, W)`` array. No copy is made for ndarray input.

    Raises:
        ValueError: If ``values`` is neither ``(H, W)`` nor ``(1, H, W)``.
    """
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(
        f"{name} expects a single-band map with shape (H, W) or (1, H, W); "
        f"got shape {arr.shape}"
    )
