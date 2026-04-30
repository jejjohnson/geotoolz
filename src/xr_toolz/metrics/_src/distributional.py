"""Distributional metrics — CRPS, energy distance, Wasserstein-1.

V2.3. These metrics compare full distributions rather than point
estimates:

- :func:`crps_ensemble` for ensemble-vs-deterministic-truth (delegates
  to :mod:`xskillscore`).
- :func:`energy_distance` for two-sample distribution comparison
  (pure :mod:`numpy`).
- :func:`wasserstein_1` for 1-D distribution comparison via
  :func:`scipy.stats.wasserstein_distance`.

Member dimension convention follows V2.2: ``ensemble_dim="member"``
(or ``sample_dim_a`` / ``sample_dim_b`` for two-sample metrics).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import xarray as xr
import xskillscore as xs
from scipy.stats import energy_distance as _scipy_energy_distance, wasserstein_distance

from xr_toolz.core import Operator


def _check_member_dim(da: xr.DataArray, dim: str) -> None:
    if dim not in da.dims:
        raise ValueError(
            f"Variable is missing ensemble/sample dim {dim!r}; "
            f"got dims={tuple(da.dims)}."
        )
    if da.sizes[dim] < 2:
        raise ValueError(
            f"Ensemble/sample dim {dim!r} has size {da.sizes[dim]}; need at least 2."
        )


# ---------- CRPS ----------------------------------------------------------


def crps_ensemble(
    ds_ensemble: xr.Dataset,
    ds_ref: xr.Dataset,
    *,
    variable: str,
    ensemble_dim: str = "member",
    dims: Sequence[str] | None = None,
) -> xr.DataArray:
    """Continuous Ranked Probability Score (ensemble vs deterministic truth).

    Delegates to :func:`xskillscore.crps_ensemble`.

    Args:
        ds_ensemble: Dataset containing the ensemble forecast variable
            with an ``ensemble_dim`` axis.
        ds_ref: Dataset containing the deterministic reference.
        variable: Variable name (must be present in both datasets).
        ensemble_dim: Name of the ensemble member dimension. Default
            ``"member"``.
        dims: Optional dims to average over after the per-pixel CRPS
            (e.g. ``("time", "lat", "lon")``); ``None`` keeps full
            shape.
    """
    da_ens = ds_ensemble[variable]
    da_ref = ds_ref[variable]
    _check_member_dim(da_ens, ensemble_dim)
    crps = cast(
        xr.DataArray,
        xs.crps_ensemble(da_ref, da_ens, member_dim=ensemble_dim, dim=[]),
    )
    if dims:
        crps = crps.mean(dim=list(dims))
    return crps


# ---------- Energy distance ----------------------------------------------


def energy_distance(
    ds_a: xr.Dataset,
    ds_b: xr.Dataset,
    *,
    variable: str,
    sample_dim_a: str = "member",
    sample_dim_b: str = "member",
    dims: Sequence[str] | None = None,
) -> xr.DataArray:
    """Two-sample energy distance.

    For 1-D samples ``a ∈ R^n`` and ``b ∈ R^m``::

        E(a, b) = 2 * E|A - B| - E|A - A'| - E|B - B'|

    where ``A, A'`` are i.i.d. draws from ``a`` and ``B, B'`` from ``b``.
    The implementation uses pairwise absolute differences along the
    sample dimensions and is **not** vectorised over additional dims.

    Args:
        ds_a: Dataset with sample variable along ``sample_dim_a``.
        ds_b: Dataset with sample variable along ``sample_dim_b``.
        variable: Variable name in both datasets.
        sample_dim_a: Sample dim in ``ds_a``.
        sample_dim_b: Sample dim in ``ds_b``.
        dims: Optional non-sample dims to average over. ``None`` keeps
            the result un-reduced (per-pixel energy distance).
    """
    da_a = ds_a[variable]
    da_b = ds_b[variable]
    _check_member_dim(da_a, sample_dim_a)
    _check_member_dim(da_b, sample_dim_b)

    # Move sample dims to the leading axis for clarity.
    a = da_a.transpose(sample_dim_a, ...).values  # (n, *rest)
    b = da_b.transpose(sample_dim_b, ...).values  # (m, *rest)
    rest_shape = a.shape[1:]

    a_flat = a.reshape(a.shape[0], -1)  # (n, K)
    b_flat = b.reshape(b.shape[0], -1)  # (m, K)

    # Per-pixel energy distance via scipy: O(n+m) per pixel rather than
    # O((n+m)^2), and uses the unbiased self-distance estimator (excludes
    # i==j pairs) so small ensembles aren't biased downward.
    energy_flat = np.array(
        [
            _scipy_energy_distance(a_flat[:, k], b_flat[:, k])
            for k in range(a_flat.shape[1])
        ]
    )
    energy = energy_flat.reshape(rest_shape) if rest_shape else energy_flat.item()

    rest_dims = tuple(d for d in da_a.dims if d != sample_dim_a)
    coords = {
        name: coord
        for name, coord in da_a.coords.items()
        if set(coord.dims).issubset(set(rest_dims))
    }
    out = xr.DataArray(np.asarray(energy), dims=rest_dims, coords=coords)
    if dims:
        out = out.mean(dim=list(dims))
    return out


# ---------- Wasserstein-1 ------------------------------------------------


def wasserstein_1(
    ds_a: xr.Dataset,
    ds_b: xr.Dataset,
    *,
    variable: str,
    sample_dim_a: str = "member",
    sample_dim_b: str = "member",
    dims: Sequence[str] | None = None,
) -> xr.DataArray:
    """1-D Wasserstein (earth-mover's) distance between two sample sets.

    Uses :func:`scipy.stats.wasserstein_distance` along the sample
    dims; supports broadcasting over the remaining dims.

    Args:
        ds_a: Dataset whose ``variable`` carries samples along
            ``sample_dim_a``.
        ds_b: Same shape (modulo sample dim) as ``ds_a``.
        variable: Variable name.
        sample_dim_a, sample_dim_b: Names of the sample dims.
        dims: Optional non-sample dims to average over.
    """
    da_a = ds_a[variable]
    da_b = ds_b[variable]
    _check_member_dim(da_a, sample_dim_a)
    _check_member_dim(da_b, sample_dim_b)

    a = da_a.transpose(sample_dim_a, ...).values
    b = da_b.transpose(sample_dim_b, ...).values
    rest_shape = a.shape[1:]
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)

    out_flat = np.array(
        [
            wasserstein_distance(a_flat[:, k], b_flat[:, k])
            for k in range(a_flat.shape[1])
        ]
    )
    out_arr = out_flat.reshape(rest_shape) if rest_shape else out_flat.item()

    rest_dims = tuple(d for d in da_a.dims if d != sample_dim_a)
    coords = {
        name: coord
        for name, coord in da_a.coords.items()
        if set(coord.dims).issubset(set(rest_dims))
    }
    out = xr.DataArray(np.asarray(out_arr), dims=rest_dims, coords=coords)
    if dims:
        out = out.mean(dim=list(dims))
    return out


# ---------- Layer-1 -------------------------------------------------------


class _DistributionalOp(Operator):
    _fn: Any = None

    def __init__(
        self,
        variable: str,
        *,
        ensemble_dim: str = "member",
        dims: Sequence[str] | None = None,
    ) -> None:
        self.variable = variable
        self.ensemble_dim = ensemble_dim
        self.dims = None if dims is None else list(dims)

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "ensemble_dim": self.ensemble_dim,
            "dims": None if self.dims is None else list(self.dims),
        }


class CRPS(_DistributionalOp):
    """Two-input CRPS operator.

    Inputs: ``(ds_ensemble, ds_reference)`` where ``ds_ensemble`` has
    an ``ensemble_dim`` axis and ``ds_reference`` is deterministic.
    """

    def _apply(self, ds_ensemble: xr.Dataset, ds_ref: xr.Dataset) -> xr.DataArray:
        return crps_ensemble(
            ds_ensemble,
            ds_ref,
            variable=self.variable,
            ensemble_dim=self.ensemble_dim,
            dims=self.dims,
        )


class _TwoSampleOp(Operator):
    _fn: Any = None

    def __init__(
        self,
        variable: str,
        *,
        sample_dim_a: str = "member",
        sample_dim_b: str = "member",
        dims: Sequence[str] | None = None,
    ) -> None:
        self.variable = variable
        self.sample_dim_a = sample_dim_a
        self.sample_dim_b = sample_dim_b
        self.dims = None if dims is None else list(dims)

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "sample_dim_a": self.sample_dim_a,
            "sample_dim_b": self.sample_dim_b,
            "dims": None if self.dims is None else list(self.dims),
        }


class EnergyDistance(_TwoSampleOp):
    """Two-sample energy-distance operator."""

    def _apply(self, ds_a: xr.Dataset, ds_b: xr.Dataset) -> xr.DataArray:
        return energy_distance(
            ds_a,
            ds_b,
            variable=self.variable,
            sample_dim_a=self.sample_dim_a,
            sample_dim_b=self.sample_dim_b,
            dims=self.dims,
        )


class Wasserstein1(_TwoSampleOp):
    """1-D Wasserstein-distance operator."""

    def _apply(self, ds_a: xr.Dataset, ds_b: xr.Dataset) -> xr.DataArray:
        return wasserstein_1(
            ds_a,
            ds_b,
            variable=self.variable,
            sample_dim_a=self.sample_dim_a,
            sample_dim_b=self.sample_dim_b,
            dims=self.dims,
        )


__all__ = [
    "CRPS",
    "EnergyDistance",
    "Wasserstein1",
    "crps_ensemble",
    "energy_distance",
    "wasserstein_1",
]
