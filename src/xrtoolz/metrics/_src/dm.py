"""Diebold–Mariano predictive-accuracy test."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def dm_test(
    errors_a: ArrayLike,
    errors_b: ArrayLike,
    *,
    h: int = 1,
    alternative: str = "two-sided",
    power: float = 2.0,
    hln_correction: bool = True,
) -> tuple[float, float]:
    """Test equal predictive accuracy for two paired *error* sequences.

    The inputs are raw forecast errors (residuals); the loss function
    is ``|e|**power`` (defaults to squared error). Pass raw errors —
    do not pre-square or pre-abs them, or the transform will be
    applied twice.
    """
    if h < 1:
        raise ValueError("h must be at least 1.")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'.")

    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"errors_a and errors_b must have matching shapes; got {a.shape} and "
            f"{b.shape}."
        )

    d = np.abs(a.ravel()) ** power - np.abs(b.ravel()) ** power
    d = d[np.isfinite(d)]
    n = d.size
    if n <= h:
        raise ValueError("dm_test requires more finite paired samples than h.")

    mean = float(np.mean(d))
    centered = d - mean
    variance = _newey_west_variance(centered, lag=h - 1)
    if variance <= 0.0:
        stat = 0.0 if np.isclose(mean, 0.0) else np.copysign(np.inf, mean)
    else:
        stat = mean / np.sqrt(variance / n)

    if hln_correction:
        factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        stat *= factor
        p_value = _p_value(stat, alternative, distribution=stats.t(df=n - 1))
    else:
        p_value = _p_value(stat, alternative, distribution=stats.norm)
    return float(stat), float(p_value)


def _newey_west_variance(centered: np.ndarray, *, lag: int) -> float:
    n = centered.size
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1)
        variance += 2.0 * weight * gamma
    return variance


def _p_value(stat: float, alternative: str, *, distribution) -> float:
    if alternative == "two-sided":
        return float(2.0 * distribution.sf(abs(stat)))
    if alternative == "greater":
        return float(distribution.sf(stat))
    return float(distribution.cdf(stat))


__all__ = ["dm_test"]
