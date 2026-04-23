"""Pixel-level and spectral evaluation metrics.

Pixel metrics take ``ds_pred`` (prediction) and ``ds_ref`` (reference)
Datasets, a variable name, and a list of reduction dimensions; they
return a :class:`xr.DataArray` with the remaining dimensions.

Spectral metrics (``psd_*_score``) compare the PSD of the prediction
against the PSD of the reference and return a normalized score plus
helpers to locate the resolved-scale crossover.

Convention: positive ``bias`` means the prediction is larger than the
reference. Correlation is Pearson's ``r``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from xr_toolz.geo._src.spectral import (
    conditional_average,
    psd_isotropic,
    psd_spacetime,
)


Dims = str | Sequence[str]


def mse(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Mean squared error reduced over ``dims``."""
    diff = ds_pred[variable] - ds_ref[variable]
    return (diff**2).mean(dim=dims)


def rmse(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Root mean squared error reduced over ``dims``."""
    return mse(ds_pred, ds_ref, variable, dims) ** 0.5


def nrmse(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Normalized RMSE: ``1 - RMSE / sqrt(<ref^2>)``.

    Returns a score in ``(-inf, 1]`` where 1 means a perfect match and 0
    means the prediction is as wrong as a zero prediction.
    """
    err = rmse(ds_pred, ds_ref, variable, dims)
    scale = (ds_ref[variable] ** 2).mean(dim=dims) ** 0.5
    return 1.0 - err / scale


def mae(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Mean absolute error reduced over ``dims``."""
    return abs(ds_pred[variable] - ds_ref[variable]).mean(dim=dims)


def bias(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Mean bias ``<pred - ref>`` reduced over ``dims``."""
    return (ds_pred[variable] - ds_ref[variable]).mean(dim=dims)


def correlation(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Pearson correlation between prediction and reference over ``dims``."""
    return xr.corr(ds_pred[variable], ds_ref[variable], dim=dims)


def r2_score(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Coefficient of determination: ``1 - SS_res / SS_tot``."""
    ref = ds_ref[variable]
    ss_res = ((ref - ds_pred[variable]) ** 2).sum(dim=dims)
    ss_tot = ((ref - ref.mean(dim=dims)) ** 2).sum(dim=dims)
    return 1.0 - ss_res / ss_tot


# ---------- spectral scores -----------------------------------------------


def psd_error(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    psd_dims: Sequence[str],
    avg_dims: Sequence[str] | None = None,
    isotropic: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """PSD of the prediction error ``pred - ref``.

    Args:
        ds_pred: Prediction dataset.
        ds_ref: Reference dataset.
        variable: Variable to score.
        psd_dims: Dimensions over which to take the PSD.
        avg_dims: Optional dims to conditionally average out after the
            PSD (e.g. average over ``lat`` after computing the lon/time
            spectrum).
        isotropic: If ``True``, use the isotropic power spectrum.
        **kwargs: Forwarded to the underlying PSD function.

    Returns:
        Dataset with a single ``"error"`` variable containing the
        PSD of the error.
    """
    diff = (ds_pred[variable] - ds_ref[variable]).to_dataset(name="error")
    psd_fn = psd_isotropic if isotropic else psd_spacetime
    err = psd_fn(diff, "error", psd_dims, **kwargs)
    if avg_dims is not None:
        err = conditional_average(err, dims=avg_dims, drop=True)
    return err


def psd_score(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    psd_dims: Sequence[str],
    avg_dims: Sequence[str] | None = None,
    isotropic: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """Normalized PSD score ``1 - PSD(err) / PSD(ref)``.

    A score of ``1`` means the error has no power at that scale; ``0``
    means the error has as much power as the reference signal.

    Args:
        ds_pred: Prediction dataset.
        ds_ref: Reference dataset.
        variable: Variable to score.
        psd_dims: Dimensions over which to take the PSD.
        avg_dims: Optional conditional-average dims applied after PSD.
        isotropic: If ``True``, use the isotropic power spectrum.
        **kwargs: Forwarded to the underlying PSD function.

    Returns:
        Dataset with a single ``"score"`` variable.
    """
    err = psd_error(
        ds_pred, ds_ref, variable, psd_dims, avg_dims, isotropic=isotropic, **kwargs
    )
    psd_fn = psd_isotropic if isotropic else psd_spacetime
    ref = psd_fn(ds_ref, variable, psd_dims, **kwargs)
    if avg_dims is not None:
        ref = conditional_average(ref, dims=avg_dims, drop=True)
    score = 1.0 - err["error"] / ref[variable]
    return score.to_dataset(name="score")


def resolved_scale(
    score: xr.DataArray | xr.Dataset,
    frequency: str,
    level: float = 0.5,
) -> float:
    """Wavelength (``1 / frequency``) at which a PSD score crosses ``level``.

    Args:
        score: DataArray of PSD-score values along ``frequency``, or a
            Dataset containing a ``"score"`` variable.
        frequency: Name of the frequency coordinate.
        level: Score threshold (default 0.5 — the commonly-used
            "resolved scale" threshold).

    Returns:
        Scalar wavelength at which the score first crosses ``level``.
    """
    score_da = score["score"] if isinstance(score, xr.Dataset) else score
    freqs = np.asarray(score_da[frequency].values)
    vals = np.asarray(score_da.values)
    positive = freqs > 0
    freqs = freqs[positive]
    vals = vals[positive]
    wavelengths = 1.0 / freqs
    return find_intercept_1D(x=wavelengths, y=vals, level=level)


# ---------- interpolation helpers -----------------------------------------


def find_intercept_1D(
    x: np.ndarray,
    y: np.ndarray,
    level: float = 0.5,
    kind: str = "slinear",
    **kwargs: Any,
) -> float:
    """Invert a 1-D monotone-ish curve at ``y = level`` and return ``x``.

    Uses :class:`scipy.interpolate.interp1d` on ``(y, x)``. Duplicate
    ``y`` values (common for plateaued PSD scores) are collapsed to
    their first occurrence in the sorted order before interpolating,
    because ``interp1d`` requires a strictly monotone x-axis.
    Extrapolates silently when ``level`` falls outside the range of the
    deduplicated ``y``.
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    order = np.argsort(y_arr)
    y_sorted = y_arr[order]
    x_sorted = x_arr[order]

    # Collapse runs of duplicate y to the first-seen x so interp1d has
    # a strictly increasing x-axis. `np.unique` returns the first index
    # of each unique value; we sort those indices to preserve order.
    _, first_idx = np.unique(y_sorted, return_index=True)
    first_idx.sort()
    y_unique = y_sorted[first_idx]
    x_unique = x_sorted[first_idx]

    if y_unique.size < 2:
        # Can't interpolate a single point; return it unconditionally.
        return float(x_unique.item()) if x_unique.size else float("nan")

    f = interp1d(
        y_unique,
        x_unique,
        fill_value=kwargs.pop("fill_value", "extrapolate"),
        kind=kind,
        **kwargs,
    )
    try:
        return float(np.asarray(f(level)).item())
    except ValueError:
        warnings.warn(
            f"level={level} outside range of y — returning edge value.",
            stacklevel=2,
        )
        y_min, y_max = float(y_unique.min()), float(y_unique.max())
        edge = y_min if level < y_min else y_max
        return float(np.asarray(f(edge)).item())
