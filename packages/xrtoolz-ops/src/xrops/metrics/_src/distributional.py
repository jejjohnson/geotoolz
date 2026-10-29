"""Distributional metrics — CRPS, energy distance, Wasserstein-1.

V2.3. These metrics compare full distributions rather than point
estimates:

- :func:`crps_ensemble` for ensemble-vs-deterministic-truth (delegates
  to :mod:`xskillscore`; a one-member ensemble scores as the absolute
  error).
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

from xrcore import Operator


def _check_member_dim(da: xr.DataArray, dim: str, *, min_size: int = 2) -> None:
    """Require ``dim`` on ``da`` with at least ``min_size`` entries.

    The two-sample distances need two samples per side; CRPS is defined
    for a single member, so it passes ``min_size=1``.
    """
    if dim not in da.dims:
        raise ValueError(
            f"Variable is missing ensemble/sample dim {dim!r}; "
            f"got dims={tuple(da.dims)}."
        )
    if da.sizes[dim] < min_size:
        raise ValueError(
            f"Ensemble/sample dim {dim!r} has size {da.sizes[dim]}; "
            f"need at least {min_size}."
        )


# ---------- CRPS ----------------------------------------------------------


def crps_ensemble(
    ensemble: xr.DataArray,
    ref: xr.DataArray,
    *,
    ensemble_dim: str = "member",
    dims: Sequence[str] | None = None,
) -> xr.DataArray:
    """Continuous Ranked Probability Score (ensemble vs deterministic truth).

    The ensemble estimator of the CRPS of members ``x_1 … x_M`` against a
    reference ``y`` is::

        CRPS = (1/M) Σ_i |x_i - y| - (1/(2 M²)) Σ_i Σ_j |x_i - x_j|

    With two or more members this delegates to
    :func:`xskillscore.crps_ensemble`. A single member (``M = 1``) is a
    point forecast: the spread term vanishes and the score is exactly the
    absolute error ``|x_1 - y|``, computed directly rather than through
    xskillscore so the identity holds bit for bit. Both paths broadcast,
    order dims and drop attrs the same way, and both treat NaN as
    xskillscore does: a NaN reference gives a NaN score, NaN members are
    skipped, and a pixel whose members are all NaN (with one member: the
    member is NaN) scores NaN. Averaging over ``dims`` is the same
    NaN-skipping mean on both paths.

    Args:
        ensemble: Ensemble forecast DataArray with an ``ensemble_dim`` axis
            of size ≥ 1.
        ref: Deterministic reference DataArray.
        ensemble_dim: Name of the ensemble member dimension. Default
            ``"member"``.
        dims: Optional dims to average over after the per-pixel CRPS
            (e.g. ``("time", "lat", "lon")``); ``None`` keeps full
            shape.

    Returns:
        The per-pixel CRPS (float64, attrs dropped), averaged over
        ``dims`` when given.

    Raises:
        ValueError: If ``ensemble`` has no ``ensemble_dim`` or it is empty.

    Example:
        >>> import numpy as np
        >>> import xarray as xr
        >>> ens = xr.DataArray([[1.0, 4.0]], dims=("member", "x"))
        >>> ref = xr.DataArray([3.0, 2.0], dims=("x",))
        >>> crps_ensemble(ens, ref).values  # one member: |x - y|
        array([2., 2.])
    """
    _check_member_dim(ensemble, ensemble_dim, min_size=1)
    if ensemble.sizes[ensemble_dim] == 1:
        crps = _crps_single_member(ensemble, ref, ensemble_dim)
    else:
        crps = cast(
            xr.DataArray,
            xs.crps_ensemble(ref, ensemble, member_dim=ensemble_dim, dim=[]),
        )
    if dims:
        crps = crps.mean(dim=list(dims))
    return crps


def _crps_single_member(
    ensemble: xr.DataArray, ref: xr.DataArray, ensemble_dim: str
) -> xr.DataArray:
    """CRPS of a one-member ensemble: the absolute error ``|x - y|``.

    Mirrors :func:`xskillscore.crps_ensemble` around the arithmetic: the
    reference is broadcast against the member slice first (so the output
    dims follow the same order), the member dim and its coordinate are
    dropped, attrs are dropped, and the result is float64.
    """
    member = ensemble.isel({ensemble_dim: 0}, drop=True)
    ref_b = ref.broadcast_like(member)
    member = member.broadcast_like(ref_b)
    out: xr.DataArray = xr.apply_ufunc(
        lambda y, x: np.abs(
            np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
        ),
        ref_b,
        member,
        dask="parallelized",
        output_dtypes=[np.float64],
        keep_attrs=False,
    )
    return out


# ---------- Energy distance ----------------------------------------------


def energy_distance(
    a: xr.DataArray,
    b: xr.DataArray,
    *,
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
        a: Sample DataArray with samples along ``sample_dim_a``.
        b: Sample DataArray with samples along ``sample_dim_b``.
        sample_dim_a: Sample dim in ``a``.
        sample_dim_b: Sample dim in ``b``.
        dims: Optional non-sample dims to average over. ``None`` keeps
            the result un-reduced (per-pixel energy distance).
    """
    da_a = a
    da_b = b
    _check_member_dim(da_a, sample_dim_a)
    _check_member_dim(da_b, sample_dim_b)

    # Move sample dims to the leading axis for clarity.
    arr_a = da_a.transpose(sample_dim_a, ...).values  # (n, *rest)
    arr_b = da_b.transpose(sample_dim_b, ...).values  # (m, *rest)
    rest_shape = arr_a.shape[1:]

    a_flat = arr_a.reshape(arr_a.shape[0], -1)  # (n, K)
    b_flat = arr_b.reshape(arr_b.shape[0], -1)  # (m, K)

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
    a: xr.DataArray,
    b: xr.DataArray,
    *,
    sample_dim_a: str = "member",
    sample_dim_b: str = "member",
    dims: Sequence[str] | None = None,
) -> xr.DataArray:
    """1-D Wasserstein (earth-mover's) distance between two sample sets.

    Uses :func:`scipy.stats.wasserstein_distance` along the sample
    dims; supports broadcasting over the remaining dims.

    Args:
        a: Sample DataArray with samples along ``sample_dim_a``.
        b: Sample DataArray with samples along ``sample_dim_b``.
        sample_dim_a, sample_dim_b: Names of the sample dims.
        dims: Optional non-sample dims to average over.
    """
    da_a = a
    da_b = b
    _check_member_dim(da_a, sample_dim_a)
    _check_member_dim(da_b, sample_dim_b)

    arr_a = da_a.transpose(sample_dim_a, ...).values
    arr_b = da_b.transpose(sample_dim_b, ...).values
    rest_shape = arr_a.shape[1:]
    a_flat = arr_a.reshape(arr_a.shape[0], -1)
    b_flat = arr_b.reshape(arr_b.shape[0], -1)

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
    an ``ensemble_dim`` axis (one or more members) and ``ds_reference``
    is deterministic. A one-member ensemble scores as the absolute error
    ``|x - y|``; see :func:`crps_ensemble`.
    """

    def _apply(self, ds_ensemble: xr.Dataset, ds_ref: xr.Dataset) -> xr.DataArray:
        return crps_ensemble(
            ds_ensemble[self.variable],
            ds_ref[self.variable],
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
            ds_a[self.variable],
            ds_b[self.variable],
            sample_dim_a=self.sample_dim_a,
            sample_dim_b=self.sample_dim_b,
            dims=self.dims,
        )


class Wasserstein1(_TwoSampleOp):
    """1-D Wasserstein-distance operator."""

    def _apply(self, ds_a: xr.Dataset, ds_b: xr.Dataset) -> xr.DataArray:
        return wasserstein_1(
            ds_a[self.variable],
            ds_b[self.variable],
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
