"""Probabilistic / ensemble metrics.

V2.2. Member-dimension convention: by default the ensemble axis is
named ``"member"`` and is settable per call via ``ensemble_dim=``.
Inputs are :class:`xr.Dataset`; the variable is selected by name.

Operators:

- :class:`SpreadSkillRatio` — ``ensemble-std / RMSE(ensemble-mean, ref)``
- :class:`RankHistogram` — Talagrand rank histogram
- :class:`EnsembleCoverage` — fraction of references inside an
  ensemble quantile envelope
- :class:`ReliabilityCurve` — observed event frequency vs forecast
  probability bin
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from xr_toolz.core import Operator


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


def spread_skill_ratio(
    ds_ensemble: xr.Dataset,
    ds_ref: xr.Dataset,
    *,
    variable: str,
    ensemble_dim: str = "member",
    dims: Sequence[str] | None = None,
) -> xr.DataArray:
    """``ensemble-std / RMSE(ensemble-mean, reference)``.

    A perfectly calibrated ensemble has spread/skill ≈ 1: the
    ensemble standard deviation matches the error of its mean. Values
    < 1 indicate under-dispersion, > 1 over-dispersion.

    Args:
        ds_ensemble: Dataset with the ensemble variable.
        ds_ref: Deterministic reference.
        variable: Variable name.
        ensemble_dim: Member dim.
        dims: Dims to reduce over (e.g. ``("time", "lat", "lon")``).
            ``None`` reduces over every non-ensemble dim.
    """
    da_ens = ds_ensemble[variable]
    da_ref = ds_ref[variable]
    _check_ensemble_dim(da_ens, ensemble_dim)

    if dims is None:
        reduce = [d for d in da_ens.dims if d != ensemble_dim]
    else:
        reduce = list(dims)

    ens_mean = da_ens.mean(dim=ensemble_dim)
    ens_std = da_ens.std(dim=ensemble_dim, ddof=1)
    spread = (ens_std**2).mean(dim=reduce) ** 0.5
    skill = ((ens_mean - da_ref) ** 2).mean(dim=reduce) ** 0.5
    return spread / skill


def rank_histogram(
    ds_ensemble: xr.Dataset,
    ds_ref: xr.Dataset,
    *,
    variable: str,
    ensemble_dim: str = "member",
) -> xr.Dataset:
    """Talagrand rank histogram.

    For each forecast/observation pair, count how many ensemble
    members are strictly less than the observation; the resulting
    rank in ``[0, n_members]`` is binned. A perfectly calibrated
    ensemble produces a flat histogram.

    Args:
        ds_ensemble: Ensemble forecast.
        ds_ref: Reference.
        variable: Variable name.
        ensemble_dim: Member dim.

    Returns:
        Dataset with a ``"rank_count"`` variable indexed by ``"rank"``
        (size ``n_members + 1``).
    """
    da_ens = ds_ensemble[variable]
    da_ref = ds_ref[variable]
    _check_ensemble_dim(da_ens, ensemble_dim)

    n = da_ens.sizes[ensemble_dim]
    ens_arr = da_ens.transpose(ensemble_dim, ...).values
    ref_arr = da_ref.broadcast_like(da_ens.isel({ensemble_dim: 0})).values
    ranks = np.sum(ens_arr < ref_arr[None, ...], axis=0)
    counts = np.bincount(ranks.ravel(), minlength=n + 1)
    return xr.Dataset(
        {"rank_count": (("rank",), counts.astype(np.int64))},
        coords={"rank": np.arange(n + 1)},
    )


def ensemble_coverage(
    ds_ensemble: xr.Dataset,
    ds_ref: xr.Dataset,
    *,
    variable: str,
    q: tuple[float, float] = (0.05, 0.95),
    ensemble_dim: str = "member",
) -> xr.DataArray:
    """Fraction of reference values inside the ensemble ``q`` envelope.

    For ``q=(0.05, 0.95)`` a calibrated ensemble produces ≈ 0.9.

    Args:
        ds_ensemble: Ensemble forecast.
        ds_ref: Reference.
        variable: Variable name.
        q: ``(low, high)`` quantiles in ``[0, 1]``.
        ensemble_dim: Member dim.

    Returns:
        Scalar fraction (0..1) :class:`xr.DataArray`.
    """
    if not 0.0 <= q[0] < q[1] <= 1.0:
        raise ValueError(f"q must be (low, high) with 0 <= low < high <= 1, got {q}.")

    da_ens = ds_ensemble[variable]
    da_ref = ds_ref[variable]
    _check_ensemble_dim(da_ens, ensemble_dim)
    lo = da_ens.quantile(q[0], dim=ensemble_dim).drop_vars("quantile")
    hi = da_ens.quantile(q[1], dim=ensemble_dim).drop_vars("quantile")
    inside = (da_ref >= lo) & (da_ref <= hi)
    return inside.mean()


def reliability_curve(
    ds_probability: xr.Dataset,
    ds_event: xr.Dataset,
    *,
    variable: str,
    probability_bins: np.ndarray | None = None,
) -> xr.Dataset:
    """Reliability diagram coordinates.

    Bins the forecast probability of an event and reports the
    observed frequency of the event within each bin. A perfectly
    calibrated forecast lies on the identity line.

    Args:
        ds_probability: Dataset whose ``variable`` holds forecast
            probabilities in ``[0, 1]``.
        ds_event: Dataset whose ``variable`` is a 0/1 event indicator
            on the same grid.
        variable: Variable name (same in both).
        probability_bins: Bin edges in ``[0, 1]``. Defaults to
            ``np.linspace(0, 1, 11)`` (10 bins).

    Returns:
        Dataset with ``forecast_probability``, ``observed_frequency``,
        and ``count`` variables indexed by ``probability_bin``.
    """
    bins = (
        np.linspace(0.0, 1.0, 11)
        if probability_bins is None
        else np.asarray(probability_bins)
    )
    p = ds_probability[variable].values.ravel()
    o = ds_event[variable].values.ravel()
    valid = np.isfinite(p) & np.isfinite(o)
    p = p[valid]
    o = o[valid]
    idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    n_bins = len(bins) - 1
    fc = np.full(n_bins, np.nan)
    ob = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for k in range(n_bins):
        sel = idx == k
        counts[k] = sel.sum()
        if counts[k] > 0:
            fc[k] = p[sel].mean()
            ob[k] = o[sel].mean()
    centers = 0.5 * (bins[:-1] + bins[1:])
    return xr.Dataset(
        {
            "forecast_probability": (("probability_bin",), fc),
            "observed_frequency": (("probability_bin",), ob),
            "count": (("probability_bin",), counts),
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
            ds_ensemble,
            ds_ref,
            variable=self.variable,
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
            ds_ensemble,
            ds_ref,
            variable=self.variable,
            ensemble_dim=self.ensemble_dim,
        )


class EnsembleCoverage(_ProbabilisticOp):
    """Fraction of references inside the ``q`` ensemble envelope."""

    def __init__(
        self,
        variable: str,
        *,
        q: tuple[float, float] = (0.05, 0.95),
        ensemble_dim: str = "member",
    ) -> None:
        super().__init__(variable, ensemble_dim=ensemble_dim)
        self.q = tuple(q)

    def _apply(self, ds_ensemble: xr.Dataset, ds_ref: xr.Dataset) -> xr.DataArray:
        return ensemble_coverage(
            ds_ensemble,
            ds_ref,
            variable=self.variable,
            q=self.q,
            ensemble_dim=self.ensemble_dim,
        )

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["q"] = list(self.q)
        return cfg


class ReliabilityCurve(Operator):
    """Reliability-diagram operator (probability vs event)."""

    def __init__(
        self, variable: str, *, probability_bins: np.ndarray | None = None
    ) -> None:
        self.variable = variable
        self.probability_bins = (
            None if probability_bins is None else np.asarray(probability_bins)
        )

    def _apply(self, ds_probability: xr.Dataset, ds_event: xr.Dataset) -> xr.Dataset:
        return reliability_curve(
            ds_probability,
            ds_event,
            variable=self.variable,
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
