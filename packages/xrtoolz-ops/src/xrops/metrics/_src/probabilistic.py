"""Probabilistic / ensemble metrics.

V2.2. Member-dimension convention: by default the ensemble axis is
named ``"member"`` and is settable per call via ``ensemble_dim=``.
The scores are ``diagnostix.skill``'s, and a reliable ensemble of any size has
the value that is said of it. A case where a member or the reference is missing
is left out.

Operators:

- :class:`SpreadSkillRatio` — ``ensemble-std / RMSE(ensemble-mean, ref)``
- :class:`RankHistogram` — Talagrand rank histogram
- :class:`EnsembleCoverage` — fraction of references inside the central
  interval of the ensemble
- :class:`ReliabilityCurve` — observed event frequency vs forecast
  probability bin
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import xarray as xr
from diagnostix import skill
from jaxtyping import Float

from xrops._src.operator import Operator


def _check_ensemble_dim(da: xr.DataArray, dim: str) -> None:
    if dim not in da.dims:
        raise ValueError(
            f"Variable is missing ensemble dim {dim!r}; got dims={tuple(da.dims)}."
        )
    if da.sizes[dim] < 2:
        raise ValueError(
            f"Ensemble dim {dim!r} has size {da.sizes[dim]}; need at least 2."
        )


# ---------- Layer-0 -------------------------------------------------------


def _pooled(
    function: Callable[..., Any],
    ensemble: xr.DataArray,
    ref: xr.DataArray,
    ensemble_dim: str,
    dims: Sequence[str] | None,
    weights: xr.DataArray | None,
) -> xr.DataArray:
    """A number from the complete cases of each ensemble, pooled over ``dims``."""
    _check_ensemble_dim(ensemble, ensemble_dim)
    pooled = [d for d in ensemble.dims if d != ensemble_dim]
    if dims is not None:
        pooled = list(dims)
    case = ensemble.isel({ensemble_dim: 0}, drop=True)
    cases = [ref.broadcast_like(case)]
    if weights is not None:
        cases.append(weights.broadcast_like(case))
    dtype = np.result_type(ensemble.dtype, ref.dtype, np.float32)

    def kernel(x: Any, y: Any, w: Any = None) -> Any:
        complete = (x == x).all(-1) & (y == y)
        axis = tuple(range(-len(pooled), 0))
        out = function(x, y, member_axis=-1, weights=w, valid=complete, axis=axis)
        return out.astype(dtype)

    return xr.apply_ufunc(
        kernel,
        ensemble,
        *cases,
        input_core_dims=[[*pooled, ensemble_dim], *([pooled] * len(cases))],
        dask="parallelized",
        output_dtypes=[dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


def spread_skill_ratio(
    ensemble: xr.DataArray,
    ref: xr.DataArray,
    *,
    ensemble_dim: str = "member",
    dims: Sequence[str] | None = None,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """``ensemble-std / RMSE(ensemble-mean, reference)``, 1 for a reliable ensemble.

    The mean of E members has an error of its own, so the ratio is multiplied by
    ``sqrt((E + 1) / E)``: it is 1 for a reliable ensemble of any size. Values
    < 1 indicate under-dispersion, > 1 over-dispersion.

    Args:
        ensemble: Ensemble DataArray with an ``ensemble_dim`` axis.
        ref: Deterministic reference DataArray.
        ensemble_dim: Member dim.
        dims: Dims to reduce over (e.g. ``("time", "lat", "lon")``).
            ``None`` reduces over every non-ensemble dim.
        weights: Weight of each case, 0 or more, such as the cell areas.
    """
    return _pooled(
        lambda *args, **kw: skill.spread_skill(*args, **kw).ratio,
        ensemble,
        ref,
        ensemble_dim,
        dims,
        weights,
    )


def rank_histogram(
    ensemble: xr.DataArray,
    ref: xr.DataArray,
    *,
    ensemble_dim: str = "member",
) -> xr.Dataset:
    """Talagrand rank histogram.

    For each forecast/observation pair, the number of ensemble members below
    the observation, in ``[0, n_members]``. An observation that equals members
    is shared evenly over the ranks around them, so the counts are floats. A
    perfectly calibrated ensemble produces a flat histogram.

    Args:
        ensemble: Ensemble forecast DataArray with an ``ensemble_dim`` axis.
        ref: Reference DataArray.
        ensemble_dim: Member dim.

    Returns:
        Dataset with a ``"rank_count"`` variable indexed by ``"rank"``
        (size ``n_members + 1``), over the cases without a missing value.
    """
    _check_ensemble_dim(ensemble, ensemble_dim)
    members = ensemble.transpose(..., ensemble_dim)
    case = members.isel({ensemble_dim: 0}, drop=True)
    x, y = members.values, ref.broadcast_like(case).transpose(*case.dims).values
    complete = np.isfinite(x).all(axis=-1) & np.isfinite(y)
    counts = skill.rank_histogram(x, y, member_axis=-1, valid=complete)
    return xr.Dataset(
        {"rank_count": (("rank",), counts)},
        coords={"rank": np.arange(ensemble.sizes[ensemble_dim] + 1)},
    )


def ensemble_coverage(
    ensemble: xr.DataArray,
    ref: xr.DataArray,
    *,
    level: float = 0.9,
    ensemble_dim: str = "member",
    dims: Sequence[str] | None = None,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Fraction of reference values inside the central interval of the ensemble.

    The members cut the line into slots of equal probability, and a reference
    counts by the share of its slot in the interval. So a reliable ensemble of
    any size covers ``level`` (``diagnostix.skill.coverage``); the fraction
    between interpolated quantiles is 0.62 at 0.9 for five members.

    Args:
        ensemble: Ensemble forecast DataArray with an ``ensemble_dim`` axis.
        ref: Reference DataArray.
        level: Probability of the interval, in ``(0, 1]``.
        ensemble_dim: Member dim.
        dims: Dims to reduce over. ``None`` reduces over every non-ensemble dim.
        weights: Weight of each case, 0 or more, such as the cell areas.

    Returns:
        The fraction (0..1), without the reduced dims.
    """
    return _pooled(
        lambda *args, **kw: skill.coverage(*args, level=level, **kw),
        ensemble,
        ref,
        ensemble_dim,
        dims,
        weights,
    )


def reliability_curve(
    probability: xr.DataArray,
    event: xr.DataArray,
    *,
    probability_bins: Float[np.ndarray, "edges"] | None = None,
) -> xr.Dataset:
    """Reliability diagram coordinates.

    Bins the forecast probability of an event and reports the
    observed frequency of the event within each bin. A perfectly
    calibrated forecast lies on the identity line.

    Args:
        probability: DataArray of forecast probabilities in ``[0, 1]``.
        event: DataArray of 0/1 event indicators on the same grid.
        probability_bins: Bin edges in ``[0, 1]``, ascending. Defaults to
            ``np.linspace(0, 1, 11)`` (10 bins). A probability outside them is
            in no bin.

    Returns:
        Dataset with ``forecast_probability``, ``observed_frequency``,
        and ``count`` variables indexed by ``probability_bin``.
    """
    bins = (
        np.linspace(0.0, 1.0, 11)
        if probability_bins is None
        else np.asarray(probability_bins)
    )
    p = np.asarray(probability.values, dtype=float)
    o = np.asarray(event.values, dtype=float)
    valid = np.isfinite(p) & np.isfinite(o)
    out = skill.reliability(p, o, edges=bins, valid=valid)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return xr.Dataset(
        {
            "forecast_probability": (("probability_bin",), out.mean_probability),
            "observed_frequency": (("probability_bin",), out.observed_frequency),
            "count": (("probability_bin",), np.rint(out.weight).astype(np.int64)),
        },
        coords={"probability_bin": centers},
    )


# ---------- Layer-1 -------------------------------------------------------


class _ProbabilisticOp(Operator):
    def __init__(
        self,
        variable: str,
        *,
        ensemble_dim: str = "member",
    ) -> None:
        self.variable = variable
        self.ensemble_dim = ensemble_dim

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable, "ensemble_dim": self.ensemble_dim}


class SpreadSkillRatio(_ProbabilisticOp):
    """``SpreadSkillRatio(variable, ensemble_dim, dims)``."""

    def __init__(
        self,
        variable: str,
        *,
        ensemble_dim: str = "member",
        dims: Sequence[str] | None = None,
    ) -> None:
        super().__init__(variable, ensemble_dim=ensemble_dim)
        self.dims = None if dims is None else list(dims)

    def _apply(self, ds_ensemble: xr.Dataset, ds_ref: xr.Dataset) -> xr.DataArray:
        return spread_skill_ratio(
            ds_ensemble[self.variable],
            ds_ref[self.variable],
            ensemble_dim=self.ensemble_dim,
            dims=self.dims,
        )

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["dims"] = None if self.dims is None else list(self.dims)
        return cfg


class RankHistogram(_ProbabilisticOp):
    """Talagrand rank histogram."""

    def _apply(self, ds_ensemble: xr.Dataset, ds_ref: xr.Dataset) -> xr.Dataset:
        return rank_histogram(
            ds_ensemble[self.variable],
            ds_ref[self.variable],
            ensemble_dim=self.ensemble_dim,
        )


class EnsembleCoverage(_ProbabilisticOp):
    """Fraction of references inside the central ``level`` of the ensemble."""

    def __init__(
        self,
        variable: str,
        *,
        level: float = 0.9,
        ensemble_dim: str = "member",
        dims: Sequence[str] | None = None,
    ) -> None:
        super().__init__(variable, ensemble_dim=ensemble_dim)
        self.level = float(level)
        self.dims = None if dims is None else list(dims)

    def _apply(self, ds_ensemble: xr.Dataset, ds_ref: xr.Dataset) -> xr.DataArray:
        return ensemble_coverage(
            ds_ensemble[self.variable],
            ds_ref[self.variable],
            level=self.level,
            ensemble_dim=self.ensemble_dim,
            dims=self.dims,
        )

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["level"] = self.level
        cfg["dims"] = None if self.dims is None else list(self.dims)
        return cfg


class ReliabilityCurve(Operator):
    """Reliability-diagram operator (probability vs event)."""

    def __init__(
        self,
        variable: str,
        *,
        probability_bins: Float[np.ndarray, "edges"] | None = None,
    ) -> None:
        self.variable = variable
        self.probability_bins = (
            None if probability_bins is None else np.asarray(probability_bins)
        )

    def _apply(self, ds_probability: xr.Dataset, ds_event: xr.Dataset) -> xr.Dataset:
        return reliability_curve(
            ds_probability[self.variable],
            ds_event[self.variable],
            probability_bins=self.probability_bins,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "probability_bins": (
                None
                if self.probability_bins is None
                else self.probability_bins.tolist()
            ),
        }


__all__ = [
    "EnsembleCoverage",
    "RankHistogram",
    "ReliabilityCurve",
    "SpreadSkillRatio",
    "ensemble_coverage",
    "rank_histogram",
    "reliability_curve",
    "spread_skill_ratio",
]
