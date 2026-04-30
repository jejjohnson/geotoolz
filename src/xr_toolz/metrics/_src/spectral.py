"""Spectral evaluation metrics.

Spectral metrics (``psd_*``) compare the PSD of the prediction against
the PSD of the reference and return a normalized score plus helpers to
locate the resolved-scale crossover.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from xr_toolz.core import Operator
from xr_toolz.transforms._src.fourier import (
    drop_negative_frequencies,
    power_spectrum,
)


# ---------- Layer-0 (xarray) ----------------------------------------------


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
    diff = (ds_pred[variable] - ds_ref[variable]).rename("error")
    err = power_spectrum(diff, dim=list(psd_dims), isotropic=isotropic, **kwargs)
    err_ds = err.rename("error").to_dataset()
    if avg_dims is not None:
        err_ds = drop_negative_frequencies(err_ds, dims=avg_dims, drop=True)
    return err_ds


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
    ref_psd = power_spectrum(
        ds_ref[variable], dim=list(psd_dims), isotropic=isotropic, **kwargs
    )
    ref_ds = ref_psd.rename(variable).to_dataset()
    if avg_dims is not None:
        ref_ds = drop_negative_frequencies(ref_ds, dims=avg_dims, drop=True)
    score = 1.0 - err["error"] / ref_ds[variable]
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

    _, first_idx = np.unique(y_sorted, return_index=True)
    first_idx.sort()
    y_unique = y_sorted[first_idx]
    x_unique = x_sorted[first_idx]

    if y_unique.size < 2:
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


# ---------- Layer-1 (Operator wrappers) -----------------------------------


class PSDScore(Operator):
    """Two-input PSD score operator."""

    def __init__(
        self,
        variable: str,
        psd_dims: Sequence[str],
        avg_dims: Sequence[str] | None = None,
        isotropic: bool = False,
        **kwargs: Any,
    ):
        self.variable = variable
        self.psd_dims = list(psd_dims)
        self.avg_dims = None if avg_dims is None else list(avg_dims)
        self.isotropic = isotropic
        self.kwargs = dict(kwargs)

    def _apply(self, ds_pred, ds_ref):
        return psd_score(
            ds_pred,
            ds_ref,
            self.variable,
            self.psd_dims,
            avg_dims=self.avg_dims,
            isotropic=self.isotropic,
            **self.kwargs,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "psd_dims": list(self.psd_dims),
            "avg_dims": None if self.avg_dims is None else list(self.avg_dims),
            "isotropic": self.isotropic,
            **self.kwargs,
        }


__all__ = [
    "PSDScore",
    "find_intercept_1D",
    "psd_error",
    "psd_score",
    "resolved_scale",
]
