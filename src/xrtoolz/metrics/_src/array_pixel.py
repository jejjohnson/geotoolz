"""Tier A — array kernels for pointwise (pixel-level) evaluation metrics.

Pure-array entry points used by Tier B (xarray) wrappers. Signatures
follow D11: ``(prediction, reference, *, axis, **kwargs) -> ndarray``.

NaN handling: matches the Tier B xarray default (``skipna=True`` for
floating-point arrays). All reductions ignore NaNs via
:func:`numpy.nanmean` / :func:`numpy.nansum`. If every element along
``axis`` is NaN, the result for that slice is NaN (NumPy emits a
``RuntimeWarning``).

Backend: numpy. JAX / CuPy variants are out of scope for the pilot.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


Axis = int | tuple[int, ...]


def mse(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Mean squared error along ``axis``. NaN-skipping."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    return np.nanmean((pred - ref) ** 2, axis=axis)


def rmse(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Root mean squared error along ``axis``. NaN-skipping."""
    return np.sqrt(mse(prediction, reference, axis=axis))


def mae(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Mean absolute error along ``axis``. NaN-skipping."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    return np.nanmean(np.abs(pred - ref), axis=axis)


def bias(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Mean bias ``<pred - ref>`` along ``axis``. NaN-skipping."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    return np.nanmean(pred - ref, axis=axis)


def nrmse(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Normalized RMSE: ``1 - RMSE / sqrt(<ref^2>)`` along ``axis``. NaN-skipping."""
    err = rmse(prediction, reference, axis=axis)
    ref = np.asarray(reference)
    scale = np.sqrt(np.nanmean(ref**2, axis=axis))
    return 1.0 - err / scale


def nrmse_score(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Mercator-flavour normalized RMSE skill score: ``1 - RMSE / std(ref)``.

    Distinct from :func:`nrmse` — both are 1 when prediction matches the
    reference exactly, but the normalisation differs:

    - :func:`nrmse` uses ``sqrt(<ref^2>)`` (raw signal magnitude).
    - :func:`nrmse_score` uses ``std(ref)`` (anomaly magnitude).

    For zero-mean references (SLA, SSH-anomaly) the two are equivalent;
    for non-zero-mean references (SST, salinity) they diverge. Adding
    :func:`nrmse_score` lets users reproduce upstream OceanBench
    leaderboard numerics (DC20a / DC21a Gulf Stream eval).

    Edge case — constant reference (``std(ref) == 0``): a perfect
    prediction (``RMSE == 0``) returns ``1`` rather than ``0/0 → NaN``;
    any non-zero error against a constant reference returns ``-inf``
    (infinitely bad skill, since there is no variability against which
    to normalise).
    """
    err = rmse(prediction, reference, axis=axis)
    ref = np.asarray(reference)
    std = np.nanstd(ref, axis=axis)
    # Avoid 0/0 on constant references: branch explicitly on std==0 so
    # the perfect-prediction case yields 1.0 instead of NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(std == 0, np.where(err == 0, 0.0, np.inf), err / std)
    return 1.0 - ratio


def correlation(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Pearson correlation between ``prediction`` and ``reference``. NaN-skipping."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    pred_mean = np.nanmean(pred, axis=axis, keepdims=True)
    ref_mean = np.nanmean(ref, axis=axis, keepdims=True)
    pred_anom = pred - pred_mean
    ref_anom = ref - ref_mean
    num = np.nanmean(pred_anom * ref_anom, axis=axis)
    denom = np.sqrt(
        np.nanmean(pred_anom**2, axis=axis) * np.nanmean(ref_anom**2, axis=axis)
    )
    return num / denom


def r2_score(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Coefficient of determination ``1 - SS_res / SS_tot``. NaN-skipping."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    ss_res = np.nansum((ref - pred) ** 2, axis=axis)
    ref_mean = np.nanmean(ref, axis=axis, keepdims=True)
    ss_tot = np.nansum((ref - ref_mean) ** 2, axis=axis)
    return 1.0 - ss_res / ss_tot


__all__ = [
    "bias",
    "correlation",
    "mae",
    "mse",
    "nrmse",
    "nrmse_score",
    "r2_score",
    "rmse",
]
