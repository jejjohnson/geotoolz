"""Diebold–Mariano predictive-accuracy test."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
from diagnostix import skill
from jaxtyping import Float
from scipy import stats

from xrops._src.operator import Operator


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
        h: Forecast horizon: the errors of an ``h``-step forecast are
            correlated up to lag ``h - 1``, and their autocovariances up to it
            are summed with a weight of one (``diagnostix.skill``). A missing
            pair is dropped for ``h = 1``, and makes the result NaN otherwise.
        alternative: ``"two-sided"``, ``"less"``, or ``"greater"``.
        power: Loss exponent applied to ``|e|``.
        hln_correction: Apply the Harvey–Leybourne–Newbold small-sample
            correction and use the Student-t reference distribution.

    Returns:
        Dataset with ``statistic`` and ``p_value`` variables. Both are
        scalars when ``dim`` is ``None``, otherwise DataArrays over the
        non-reduced dimensions. NaN where the loss differences are all the same,
        or their long-run variance is estimated as negative.
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
    errors_a: Float[np.ndarray, "*shape"],
    errors_b: Float[np.ndarray, "*shape"],
    *,
    h: int,
    alternative: str,
    power: float,
    hln_correction: bool,
) -> tuple[float, float]:
    """Diebold–Mariano statistic / p-value for one paired error vector."""
    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)

    a, b = a.ravel(), b.ravel()
    if h == 1:
        # only a forecast of one step has errors that are not h apart
        paired = np.isfinite(a) & np.isfinite(b)
        a, b = a[paired], b[paired]
    if a.size <= h:
        raise ValueError("dm_test requires more finite paired samples than h.")
    out = skill.diebold_mariano(a, b, horizon=h, power=power, correction=hln_correction)
    distribution = stats.t(df=out.degrees_of_freedom) if hln_correction else stats.norm
    statistic = float(out.statistic)
    return statistic, _p_value(statistic, alternative, distribution=distribution)


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
