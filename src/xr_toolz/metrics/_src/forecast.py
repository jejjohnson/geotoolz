"""Forecast-skill metrics — skill broken down by forecast lead time.

This module implements the lead-time slice of validation.md §1
("Scales of Evaluation"). The :class:`SkillByLeadTime` operator
applies any inner metric (an :class:`xr_toolz.core.Operator`) per
lead-time slice, returning skill as a function of forecast horizon.

The inner-metric type is :class:`~xr_toolz.core.Operator` rather than
a bare callable so :meth:`get_config` can introspect the full skill
configuration (per V1.1 acceptance criteria).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from xr_toolz.core import Operator


# ---------- Layer-0 (xarray) ----------------------------------------------


def skill_by_lead_time(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    *,
    metric: Operator,
    lead_dim: str = "lead_time",
) -> xr.DataArray | xr.Dataset:
    """Apply ``metric`` per lead-time slice and stack along ``lead_dim``.

    Args:
        ds_pred: Prediction dataset with a ``lead_dim`` axis.
        ds_ref: Reference dataset. May share or omit the ``lead_dim``
            axis; if omitted the same reference is used for every lead.
        metric: Inner :class:`Operator` taking ``(pred_ds, ref_ds)``
            and returning an :class:`xr.DataArray` or :class:`xr.Dataset`.
        lead_dim: Name of the lead-time dimension (default ``"lead_time"``).

    Returns:
        The metric stacked along ``lead_dim``; a :class:`xr.DataArray`
        if the inner metric returns one, otherwise a :class:`xr.Dataset`.
    """
    if lead_dim not in ds_pred.dims:
        raise ValueError(
            f"Prediction is missing lead dim {lead_dim!r}; "
            f"got dims={tuple(ds_pred.dims)}."
        )

    leads = ds_pred[lead_dim].values
    ref_has_lead = lead_dim in ds_ref.dims
    if ref_has_lead:
        if ds_ref.sizes[lead_dim] != ds_pred.sizes[lead_dim]:
            raise ValueError(
                f"Reference {lead_dim!r} size {ds_ref.sizes[lead_dim]} does "
                f"not match prediction size {ds_pred.sizes[lead_dim]}."
            )
        if (
            lead_dim in ds_ref.coords
            and lead_dim in ds_pred.coords
            and not np.array_equal(ds_ref[lead_dim].values, leads)
        ):
            raise ValueError(
                f"Reference {lead_dim!r} coordinate values differ from "
                f"prediction; align them with `xr.align(...)` upstream."
            )

    pieces: list[xr.DataArray | xr.Dataset] = []
    for i in range(len(leads)):
        pred_i = ds_pred.isel({lead_dim: i})
        ref_i = ds_ref.isel({lead_dim: i}) if ref_has_lead else ds_ref
        pieces.append(metric(pred_i, ref_i))

    return xr.concat(pieces, dim=ds_pred[lead_dim])


# ---------- Layer-1 (Operator) --------------------------------------------


class SkillByLeadTime(Operator):
    """Apply an inner metric per lead-time slice.

    Args:
        metric: Inner :class:`Operator` (e.g. ``RMSE("ssh", dims="time")``)
            evaluated on each ``lead_dim`` slice independently.
        lead_dim: Name of the lead-time dimension on the prediction
            dataset. Defaults to ``"lead_time"``.

    Example:
        >>> from xr_toolz.metrics import RMSE, SkillByLeadTime
        >>> op = SkillByLeadTime(RMSE("ssh", dims=("lat", "lon")))
        >>> skill = op(pred_ds, ref_ds)  # DataArray indexed by lead_time
    """

    def __init__(self, metric: Operator, *, lead_dim: str = "lead_time") -> None:
        if not isinstance(metric, Operator):
            raise TypeError(
                f"SkillByLeadTime requires an Operator, got "
                f"{type(metric).__name__!r}; pass an instantiated metric like "
                "RMSE(...) so its config is introspectable."
            )
        self.metric = metric
        self.lead_dim = lead_dim

    def _apply(
        self, ds_pred: xr.Dataset, ds_ref: xr.Dataset
    ) -> xr.DataArray | xr.Dataset:
        return skill_by_lead_time(
            ds_pred, ds_ref, metric=self.metric, lead_dim=self.lead_dim
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "metric": {
                "class": type(self.metric).__name__,
                "config": self.metric.get_config(),
            },
            "lead_dim": self.lead_dim,
        }


__all__ = ["SkillByLeadTime", "skill_by_lead_time"]
