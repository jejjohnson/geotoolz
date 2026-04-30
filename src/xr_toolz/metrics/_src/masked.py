"""Masked-metric wrappers — apply any inner metric on the valid pixels.

V2.4. The :class:`MaskedMetric` operator wraps any pixel / spectral /
structural metric and applies it after masking the inputs with a
boolean (or NaN-aware float) mask. Eliminates the
``pred.where(mask), ref.where(mask)`` boilerplate that proliferates
upstream of every metric call.

This is **not** per-region breakdown — that is :class:`EvaluateByRegion`
in V1.2. ``MaskedMetric`` is single-mask filtering, strictly orthogonal.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from xr_toolz.core import Operator


# ---------- Layer-0 -------------------------------------------------------


def masked_metric(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    *,
    metric: Operator,
    mask: xr.DataArray,
) -> xr.DataArray | xr.Dataset:
    """Apply ``metric`` to ``(pred, ref)`` after masking with ``mask``.

    Args:
        ds_pred: Prediction dataset.
        ds_ref: Reference dataset.
        metric: Inner :class:`Operator` (any two-input metric).
        mask: Boolean :class:`xr.DataArray` (or NaN-aware float).
            Pixels where ``mask`` is ``False`` / ``NaN`` are dropped.
            Float masks are coerced via ``mask.fillna(False).astype(bool)``
            so NaN explicitly maps to False (rather than relying on
            ``where(NaN)`` truthiness, which is implementation-defined).
    """
    mask_bool = mask.fillna(False).astype(bool)
    return metric(ds_pred.where(mask_bool), ds_ref.where(mask_bool))


# ---------- Layer-1 -------------------------------------------------------


class MaskedMetric(Operator):
    """Wrap any inner metric so it operates only on masked-valid pixels.

    Args:
        metric: Inner two-input :class:`Operator` (e.g. ``RMSE(...)``).
        mask: Optional default mask. If ``None``, a per-call mask must
            be supplied via ``__call__(pred, ref, mask=...)``.

    Example:
        >>> from xr_toolz.metrics import RMSE, MaskedMetric
        >>> op = MaskedMetric(RMSE("ssh", ("lat", "lon")), mask=ocean)
        >>> op(pred_ds, ref_ds)
        >>>
        >>> op2 = MaskedMetric(RMSE("sst", ("lat", "lon")))
        >>> op2(pred_ds, ref_ds, mask=cloud_free_today)
    """

    def __init__(self, metric: Operator, mask: xr.DataArray | None = None) -> None:
        if not isinstance(metric, Operator):
            raise TypeError(
                f"MaskedMetric requires an Operator, got {type(metric).__name__!r}."
            )
        self.metric = metric
        self.mask = mask

    def _apply(
        self,
        ds_pred: xr.Dataset,
        ds_ref: xr.Dataset,
        *,
        mask: xr.DataArray | None = None,
    ) -> xr.DataArray | xr.Dataset:
        chosen = mask if mask is not None else self.mask
        if chosen is None:
            raise ValueError(
                "MaskedMetric requires a mask either at construction "
                "(MaskedMetric(metric, mask=...)) or per call "
                "(op(pred, ref, mask=...))."
            )
        return masked_metric(ds_pred, ds_ref, metric=self.metric, mask=chosen)

    def get_config(self) -> dict[str, Any]:
        return {
            "metric": {
                "class": type(self.metric).__name__,
                "config": self.metric.get_config(),
            },
            "mask": "<DataArray>" if self.mask is not None else None,
        }


__all__ = ["MaskedMetric", "masked_metric"]
