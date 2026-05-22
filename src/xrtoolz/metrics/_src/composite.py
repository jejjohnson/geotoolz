"""Thin composition helpers built on existing metric primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import xarray as xr

from xrtoolz.metrics._src.pixel import nrmse, rmse
from xrtoolz.metrics._src.spectral import psd_score, resolved_scale_2d


def rmse_skill_scores(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    *,
    variable: str,
    space_dims: Sequence[str] = ("lat", "lon"),
    time_dim: str = "time",
) -> xr.Dataset:
    """Bundle the canonical RMSE-based skill diagnostics.

    Args:
        ds_pred: Prediction dataset.
        ds_ref: Reference dataset.
        variable: Variable to score.
        space_dims: Spatial dimensions reduced for the per-time skill.
        time_dim: Temporal dimension reduced for the per-cell RMSE map.

    Returns:
        Dataset with ``rmse_t`` over ``time_dim``, ``rmse_xy`` over
        ``space_dims``, plus scalar ``leaderboard_rmse`` and
        ``error_stability``.

    Examples:
        >>> rmse_skill_scores(pred, ref, variable="ssh")

    ``error_stability`` uses xarray's default ``std(..., ddof=0)`` to
    match the upstream OSSE report.
    """
    space_dims_t = tuple(space_dims)
    if time_dim in space_dims_t:
        raise ValueError(
            f"time_dim={time_dim!r} must not appear in space_dims={space_dims_t!r};"
            " duplicate core dims are rejected by xr.apply_ufunc."
        )
    rmse_t = nrmse(ds_pred, ds_ref, variable, dims=space_dims_t).rename("rmse_t")
    rmse_xy = rmse(ds_pred, ds_ref, variable, dims=time_dim).rename("rmse_xy")
    leaderboard_rmse = nrmse(
        ds_pred,
        ds_ref,
        variable,
        dims=(time_dim, *space_dims_t),
    ).rename("leaderboard_rmse")
    # Pin ddof=0 explicitly to match the OSSE-report convention; relying
    # on xarray's default could shift quietly if it ever changes.
    error_stability = rmse_t.std(dim=time_dim, ddof=0).rename("error_stability")
    return xr.Dataset(
        {
            "rmse_t": rmse_t,
            "rmse_xy": rmse_xy,
            "leaderboard_rmse": leaderboard_rmse,
            "error_stability": error_stability,
        }
    )


def psd_score_spacetime(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    *,
    variable: str,
    space_dim: str = "lon",
    time_dim: str = "time",
    avg_dims: Sequence[str] | None = ("lat",),
    level: float = 0.5,
    **xrft_kwargs: Any,
) -> tuple[xr.Dataset, dict[str, float]]:
    """Compute a 2-D space-time PSD score and resolved-scale summary.

    Args:
        ds_pred: Prediction dataset.
        ds_ref: Reference dataset.
        variable: Variable to score.
        space_dim: Spatial dimension used in the PSD.
        time_dim: Temporal dimension used in the PSD.
        avg_dims: Optional dimensions averaged out after the PSD.
        level: Threshold used by :func:`resolved_scale_2d`.
        **xrft_kwargs: Extra keyword arguments forwarded to
            :func:`psd_score`.

    Returns:
        Tuple ``(score, summary)`` where ``score`` is a Dataset with a
        ``"score"`` variable on ``(freq_<space_dim>, freq_<time_dim>)``
        and ``summary`` contains the min/max resolved wavelengths.

    Examples:
        >>> score, summary = psd_score_spacetime(pred, ref, variable="ssh")
    """
    if xrft_kwargs.get("isotropic"):
        # Isotropic mode collapses the two spatial axes into a single
        # radial frequency, so the freq_<space_dim>/freq_<time_dim>
        # plane this function indexes into below would not exist.
        raise ValueError(
            "psd_score_spacetime requires the 2-D (space, time) PSD; "
            "isotropic=True is not supported here. Use psd_score "
            "directly for isotropic spectra."
        )
    freq_space_dim = f"freq_{space_dim}"
    freq_time_dim = f"freq_{time_dim}"
    score = psd_score(
        ds_pred,
        ds_ref,
        variable,
        psd_dims=(space_dim, time_dim),
        avg_dims=avg_dims,
        **xrft_kwargs,
    )
    positive = (score[freq_space_dim] > 0.0) & (score[freq_time_dim] > 0.0)
    score = score.where(positive, drop=True).transpose(freq_space_dim, freq_time_dim)
    summary = resolved_scale_2d(
        score,
        level=level,
        space_dim=freq_space_dim,
        time_dim=freq_time_dim,
    )
    return score, summary
