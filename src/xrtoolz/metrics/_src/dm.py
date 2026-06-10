"""Diebold–Mariano predictive-accuracy test."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
from scipy import stats

from xrtoolz._operator import Operator


_ALTERNATIVES = {"two-sided", "less", "greater"}


def dm_test(
    errors_a: xr.DataArray,
    errors_b: xr.DataArray,
    *,
    dim: str | Sequence[str] | None = None,
    h: int = 1,
    alternative: str = "two-sided",
    power: float = 2.0,
    hln_correction: bool = True,
) -> xr.Dataset:
    """Test equal predictive accuracy for two paired *error* sequences.

    The inputs are raw forecast errors (residuals); the loss function
    is ``|e|**power`` (defaults to squared error). Pass raw errors —
    do not pre-square or pre-abs them, or the transform will be
    applied twice.

    Args:
        errors_a: Forecast errors of method A.
        errors_b: Forecast errors of method B, paired with ``errors_a``.
        dim: Dimension(s) over which to compute the test. When ``None``
            (default), the full array is flattened and a single scalar
            statistic / p-value pair is returned. When given, the test
            is computed independently along ``dim`` and broadcast over
            the remaining dimensions.
        h: Forecast horizon (Newey–West lag truncation is ``h - 1``).
        alternative: ``"two-sided"``, ``"less"``, or ``"greater"``.
        power: Loss exponent applied to ``|e|``.
        hln_correction: Apply the Harvey–Leybourne–Newbold small-sample
            correction and use the Student-t reference distribution.

    Returns:
        Dataset with ``statistic`` and ``p_value`` variables. Both are
        scalars when ``dim`` is ``None``, otherwise DataArrays over the
        non-reduced dimensions.
    """
    if h < 1:
        raise ValueError("h must be at least 1.")
    if alternative not in _ALTERNATIVES:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'.")

    if dim is None:
        # Flatten everything to a single paired sequence. Align first so
        # mismatched coordinate labels or dimension orders are caught (or
        # reconciled) by xarray rather than silently paired positionally.
        aligned_a, aligned_b = xr.align(errors_a, errors_b, join="exact")
        aligned_b = aligned_b.transpose(*aligned_a.dims)
        stat, p_value = _dm_stat_pvalue(
            aligned_a.values,
            aligned_b.values,
            h=h,
            alternative=alternative,
            power=power,
            hln_correction=hln_correction,
        )
        return xr.Dataset({"statistic": stat, "p_value": p_value})

    core = [dim] if isinstance(dim, str) else list(dim)
    stat, p_value = xr.apply_ufunc(
        _dm_stat_pvalue,
        errors_a,
        errors_b,
        input_core_dims=[core, core],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
        dask_gufunc_kwargs={"allow_rechunk": True},
        kwargs={
            "h": h,
            "alternative": alternative,
            "power": power,
            "hln_correction": hln_correction,
        },
    )
    return xr.Dataset({"statistic": stat, "p_value": p_value})


def _dm_stat_pvalue(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    *,
    h: int,
    alternative: str,
    power: float,
    hln_correction: bool,
) -> tuple[float, float]:
    """Diebold–Mariano statistic / p-value for one paired error vector."""
    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)

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


class DieboldMariano(Operator):
    """Layer-1 Diebold–Mariano test on a forecast-error variable.

    Selects ``variable`` from two input Datasets of forecast *errors*
    (residuals) and runs :func:`dm_test`, returning a Dataset with
    ``statistic`` and ``p_value``.

    Args:
        variable: Error variable to select from each input Dataset.
        dim: Dimension(s) to reduce. ``None`` flattens the full array.
        h: Forecast horizon.
        alternative: ``"two-sided"``, ``"less"``, or ``"greater"``.
        power: Loss exponent applied to ``|e|``.
        hln_correction: Apply the Harvey–Leybourne–Newbold correction.

    Example:
        ```pycon
        >>> op = DieboldMariano("error", dim="time")
        >>> result = op(ds_errors_a, ds_errors_b)
        >>> result["statistic"], result["p_value"]
        ```
    """

    def __init__(
        self,
        variable: str,
        *,
        dim: str | Sequence[str] | None = None,
        h: int = 1,
        alternative: str = "two-sided",
        power: float = 2.0,
        hln_correction: bool = True,
    ) -> None:
        self.variable = variable
        self.dim = dim
        self.h = h
        self.alternative = alternative
        self.power = power
        self.hln_correction = hln_correction

    def _apply(self, ds_a: xr.Dataset, ds_b: xr.Dataset) -> xr.Dataset:
        return dm_test(
            ds_a[self.variable],
            ds_b[self.variable],
            dim=self.dim,
            h=self.h,
            alternative=self.alternative,
            power=self.power,
            hln_correction=self.hln_correction,
        )

    def get_config(self) -> dict[str, Any]:
        if self.dim is None or isinstance(self.dim, str):
            dim = self.dim
        else:
            dim = list(self.dim)
        return {
            "variable": self.variable,
            "dim": dim,
            "h": self.h,
            "alternative": self.alternative,
            "power": self.power,
            "hln_correction": self.hln_correction,
        }


__all__ = ["DieboldMariano", "dm_test"]
