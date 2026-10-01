"""Shared percentile clip-and-stretch core.

The single NaN-aware percentile stretch behind
``radiometry.percentile_clip`` / ``radiometry.PercentileClip``,
``normalize.HistogramStretch`` and ``viz.StretchToUint8`` (all spelled
``lower`` / ``upper`` / ``reduce_axes``). It owns the ``upper > lower``
validation, so the wrappers delegate without re-checking.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Shaped


__all__ = ["percentile_stretch"]


def percentile_stretch(
    arr: Shaped[np.ndarray, "*dims"],
    lower: float,
    upper: float,
    *,
    reduce_axes: int | tuple[int, ...] | None = (-2, -1),
) -> Float[np.ndarray, "*dims"]:
    """Clip to percentile bounds and stretch the result into ``[0, 1]``.

    NaN-aware: percentiles are computed with ``np.nanpercentile`` so
    fill/NaN pixels don't poison the thresholds. A constant slice
    (``hi == lo``) maps to ``0`` rather than dividing by zero.

    Args:
        arr: Input array of any shape.
        lower: Lower percentile in ``[0, 100]``.
        upper: Upper percentile in ``[0, 100]``; must exceed ``lower``.
        reduce_axes: Axis (or axes) to compute percentiles over. The default
            ``(-2, -1)`` stretches each leading band / time slice
            independently; ``None`` uses one global pair of thresholds.

    Returns:
        Float array of the same shape with values in ``[0, 1]`` (NaNs
        propagate through untouched).

    Raises:
        ValueError: If ``upper <= lower``.
    """
    if upper <= lower:
        raise ValueError(
            f"percentile_stretch requires upper > lower; got {lower=}, {upper=}"
        )
    # `keepdims=True` so lo/hi broadcast back over the reduced axes.
    lo = np.nanpercentile(arr, lower, axis=reduce_axes, keepdims=True)
    hi = np.nanpercentile(arr, upper, axis=reduce_axes, keepdims=True)
    # Guard the constant-slice degenerate case.
    denom = np.where(hi > lo, hi - lo, 1.0)
    return np.clip((arr - lo) / denom, 0.0, 1.0)
