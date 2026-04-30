"""Multiscale evaluation — skill broken down by region.

Implements the regional slice of validation.md §1. The
:class:`EvaluateByRegion` operator applies any inner metric per
region and returns the per-region scores stacked along a ``region``
axis.

Region inputs are normalized internally so callers can pass any of:

* a ``regionmask.Regions`` object (lazy-imported only when used),
* an integer-encoded mask :class:`xr.DataArray` (one label per
  region, NaN/-1 outside),
* a ``dict[str, xr.DataArray[bool]]`` of named boolean masks.

The normalization helper :func:`normalize_regions` is exposed so a
caller can pre-build the mask once and reuse across many metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from xr_toolz.core import Operator


def _import_regionmask() -> Any:
    try:
        import regionmask
    except ImportError as exc:
        raise ImportError(
            "EvaluateByRegion received a regionmask.Regions input but "
            "`regionmask` is not installed. Install with "
            "`pip install regionmask` or pass an integer mask / dict "
            "of boolean masks instead."
        ) from exc
    return regionmask


def normalize_regions(
    regions: Any,
    template: xr.Dataset | xr.DataArray,
) -> tuple[xr.DataArray, dict[int, str]]:
    """Normalize an arbitrary region spec to ``(int_mask, names)``.

    Args:
        regions: One of: ``regionmask.Regions``, integer-mask
            :class:`xr.DataArray`, or ``dict[str, DataArray[bool]]``.
        template: Dataset / DataArray supplying the lat/lon grid that
            the mask must align with (used only when ``regions`` is a
            ``regionmask.Regions``).

    Returns:
        ``(mask, names)`` where ``mask`` is an integer DataArray
        (``-1`` outside any region) and ``names`` maps integer labels
        to human-readable region names.
    """
    if isinstance(regions, xr.DataArray):
        # Float masks are allowed to use NaN for "outside any region"
        # (matching the docstring); fill NaNs with -1 before casting
        # so they don't silently turn into a garbage int64 value.
        mask = regions
        if np.issubdtype(mask.dtype, np.floating):
            non_nan = mask.values[~np.isnan(mask.values)]
            if non_nan.size and not np.allclose(non_nan, non_nan.astype(np.int64)):
                raise ValueError(
                    "Integer-mask DataArray inputs must encode region labels as "
                    "whole numbers (NaN allowed for outside); got non-integer "
                    "values."
                )
            mask = xr.where(mask.notnull(), mask, np.int64(-1))
        labels = np.unique(mask.values)
        labels_list = [int(lbl) for lbl in labels if int(lbl) >= 0]
        names = {lbl: f"region_{lbl}" for lbl in labels_list}
        return mask.astype("int64"), names

    if isinstance(regions, Mapping):
        if not regions:
            raise ValueError("dict-of-masks regions must be non-empty.")
        names = {}
        out: xr.DataArray | None = None
        for i, (name, m) in enumerate(regions.items()):
            if not isinstance(m, xr.DataArray):
                raise TypeError(
                    f"region {name!r}: expected boolean xr.DataArray, "
                    f"got {type(m).__name__}"
                )
            names[i] = name
            contrib = xr.where(m, np.int64(i), np.int64(-1))
            out = contrib if out is None else xr.where(m, np.int64(i), out)
        assert out is not None
        return out.astype("int64"), names

    # Detect a regionmask.Regions object by module name *without* importing
    # regionmask — that way unsupported types raise TypeError below even
    # when regionmask isn't installed.
    if type(regions).__module__.startswith("regionmask"):
        regionmask = _import_regionmask()
        if isinstance(regions, regionmask.Regions):
            lat = template["lat"] if "lat" in template.coords else template["latitude"]
            lon = template["lon"] if "lon" in template.coords else template["longitude"]
            mask = regions.mask(lon, lat)
            names = {int(i): str(n) for i, n in enumerate(regions.names)}
            mask = xr.where(mask.notnull(), mask, np.int64(-1)).astype("int64")
            return mask, names

    raise TypeError(
        f"Unsupported regions type {type(regions).__name__!r}; expected "
        "regionmask.Regions, int-mask DataArray, or dict[str, DataArray]."
    )


# ---------- Layer-0 (xarray) ----------------------------------------------


def evaluate_by_region(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    *,
    metric: Operator,
    regions: Any,
) -> xr.Dataset:
    """Apply ``metric`` per region and return per-region scores.

    Args:
        ds_pred: Prediction dataset.
        ds_ref: Reference dataset, aligned on the same grid as
            ``ds_pred``.
        metric: Inner :class:`Operator` taking ``(pred_ds, ref_ds)``.
        regions: Region spec (see :func:`normalize_regions`).

    Returns:
        Dataset indexed by ``region`` with one variable per metric
        output (a single ``"score"`` variable when the inner metric
        returns a :class:`xr.DataArray`).
    """
    mask, names = normalize_regions(regions, ds_pred)
    region_ids = sorted(names)

    # Build a NaN template for empty regions. We do this by calling the
    # metric on the first non-empty region and then multiplying by NaN —
    # the operator's contract for empty regions becomes deterministic
    # rather than dependent on the inner metric's NaN behaviour.
    slots: list[xr.Dataset | None] = [None] * len(region_ids)
    nan_template: xr.Dataset | None = None
    for idx, rid in enumerate(region_ids):
        sel = mask == rid
        if not bool(sel.any()):
            continue
        pred_r = ds_pred.where(sel)
        ref_r = ds_ref.where(sel)
        result = metric(pred_r, ref_r)
        if isinstance(result, xr.DataArray):
            result = result.to_dataset(name=result.name or "score")
        slots[idx] = result
        if nan_template is None:
            nan_template = result * np.nan

    if nan_template is None:
        raise ValueError(
            "evaluate_by_region: every region selected zero pixels — cannot "
            "build a NaN-template result. Check that the mask intersects the "
            "data grid."
        )
    pieces: list[xr.Dataset] = [p if p is not None else nan_template for p in slots]

    out = xr.concat(pieces, dim="region")
    out = out.assign_coords(region=("region", [names[i] for i in region_ids]))
    return out


# ---------- Layer-1 (Operator) --------------------------------------------


class EvaluateByRegion(Operator):
    """Apply an inner metric per region.

    Args:
        metric: Inner :class:`Operator` evaluated per region.
        regions: One of:
            * a ``regionmask.Regions`` instance (requires the optional
              ``regionmask`` dep),
            * an integer-encoded mask :class:`xr.DataArray`,
            * a ``dict[str, xr.DataArray[bool]]``.

    Pre-normalize once and reuse via :func:`normalize_regions` if you
    intend to evaluate multiple metrics over the same regions:

        >>> mask, names = normalize_regions(natural_earth, ds)
        >>> EvaluateByRegion(RMSE(...), regions=mask)
    """

    def __init__(self, metric: Operator, *, regions: Any) -> None:
        if not isinstance(metric, Operator):
            raise TypeError(
                f"EvaluateByRegion requires an Operator, got {type(metric).__name__!r}."
            )
        self.metric = metric
        self.regions = regions

    def _apply(self, ds_pred: xr.Dataset, ds_ref: xr.Dataset) -> xr.Dataset:
        return evaluate_by_region(
            ds_pred, ds_ref, metric=self.metric, regions=self.regions
        )

    def get_config(self) -> dict[str, Any]:
        if isinstance(self.regions, xr.DataArray):
            regions_repr = "<DataArray>"
        elif isinstance(self.regions, Mapping):
            regions_repr = list(self.regions.keys())
        else:
            regions_repr = type(self.regions).__name__
        return {
            "metric": {
                "class": type(self.metric).__name__,
                "config": self.metric.get_config(),
            },
            "regions": regions_repr,
        }


__all__ = [
    "EvaluateByRegion",
    "evaluate_by_region",
    "normalize_regions",
]
