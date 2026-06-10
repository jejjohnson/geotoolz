"""Residual diagnostics for along-track evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from jaxtyping import Float
from scipy.stats import binned_statistic_2d

from xrtoolz._operator import Operator


def bin_residuals_2d(
    ds_track: xr.Dataset,
    *,
    var_ref: str,
    var_pred: str,
    lon_bins: Sequence[float],
    lat_bins: Sequence[float],
    lon: str = "lon",
    lat: str = "lat",
    statistics: Sequence[str] = ("mean", "std", "count", "rmse"),
) -> xr.Dataset:
    """Bin along-track residuals onto a 2-D latitude/longitude grid."""
    lon_edges = np.asarray(lon_bins, dtype=float)
    lat_edges = np.asarray(lat_bins, dtype=float)
    # Broadcast residual + coords against each other so extra dims on
    # var_ref/var_pred (e.g. an ensemble axis) line up with lon/lat
    # rather than producing length-mismatched ravels.
    residual_da = ds_track[var_pred] - ds_track[var_ref]
    residual_da, lon_da, lat_da = xr.broadcast(
        residual_da, ds_track[lon], ds_track[lat]
    )
    residual = residual_da.values.ravel()
    lon_values = lon_da.values.ravel()
    lat_values = lat_da.values.ravel()
    valid = np.isfinite(residual) & np.isfinite(lon_values) & np.isfinite(lat_values)

    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    for statistic in statistics:
        if statistic == "rmse":
            values = residual[valid] ** 2
            scipy_stat = "mean"
        elif statistic in {"mean", "std", "count", "median", "min", "max"}:
            values = residual[valid]
            scipy_stat = statistic
        else:
            raise ValueError(f"Unsupported binned residual statistic {statistic!r}.")

        stat, _, _, _ = binned_statistic_2d(
            lon_values[valid],
            lat_values[valid],
            values,
            statistic=scipy_stat,
            bins=[lon_edges, lat_edges],
        )
        out = stat.T
        if statistic == "rmse":
            out = np.sqrt(out)
        data_vars[statistic] = (("lat_bin", "lon_bin"), out)

    return xr.Dataset(
        data_vars,
        coords={
            "lat_bin": 0.5 * (lat_edges[:-1] + lat_edges[1:]),
            "lon_bin": 0.5 * (lon_edges[:-1] + lon_edges[1:]),
        },
    )


def scores_by_region(
    ds_track: xr.Dataset,
    *,
    var_ref: str,
    var_pred: str,
    regions: xr.DataArray | regionmask.Regions,
    lon: str = "lon",
    lat: str = "lat",
    metrics: Sequence[str] = ("rmse", "bias", "correlation", "explained_variance"),
    region_dim: str = "region",
) -> xr.Dataset:
    """Compute residual scores stratified by geographic or categorical region.

    Points whose region label is NaN are dropped before metric reductions.
    """
    mask, names = _region_labels(
        ds_track,
        regions,
        var_ref=var_ref,
        lon=lon,
        lat=lat,
    )
    ref = ds_track[var_ref].values.ravel()
    pred = ds_track[var_pred].values.ravel()
    labels = mask.values.ravel()
    valid = ~pd.isna(labels) & np.isfinite(ref) & np.isfinite(pred)
    labels_valid = labels[valid]
    ref_valid = ref[valid]
    pred_valid = pred[valid]

    region_values = list(names) if names else list(pd.unique(labels_valid))
    out: dict[str, list[float]] = {metric: [] for metric in metrics}
    for region in region_values:
        sel = labels_valid == region
        ref_region = ref_valid[sel]
        pred_region = pred_valid[sel]
        for metric in metrics:
            out[metric].append(_score_region(ref_region, pred_region, metric))

    return xr.Dataset(
        {
            metric: (region_dim, np.asarray(values, dtype=float))
            for metric, values in out.items()
        },
        coords={region_dim: region_values},
    )


class BinnedResiduals2D(Operator):
    """Layer-1 wrapper for :func:`bin_residuals_2d`."""

    def __init__(
        self,
        *,
        var_ref: str,
        var_pred: str,
        lon_bins: Sequence[float],
        lat_bins: Sequence[float],
        lon: str = "lon",
        lat: str = "lat",
        statistics: Sequence[str] = ("mean", "std", "count", "rmse"),
    ) -> None:
        self.var_ref = var_ref
        self.var_pred = var_pred
        self.lon_bins = list(lon_bins)
        self.lat_bins = list(lat_bins)
        self.lon = lon
        self.lat = lat
        self.statistics = tuple(statistics)

    def _apply(self, ds_track: xr.Dataset) -> xr.Dataset:
        return bin_residuals_2d(
            ds_track,
            var_ref=self.var_ref,
            var_pred=self.var_pred,
            lon=self.lon,
            lat=self.lat,
            lon_bins=self.lon_bins,
            lat_bins=self.lat_bins,
            statistics=self.statistics,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "var_ref": self.var_ref,
            "var_pred": self.var_pred,
            "lon": self.lon,
            "lat": self.lat,
            "lon_bins": list(self.lon_bins),
            "lat_bins": list(self.lat_bins),
            "statistics": list(self.statistics),
        }


class RegionScores(Operator):
    """Layer-1 wrapper for :func:`scores_by_region`."""

    def __init__(
        self,
        *,
        var_ref: str,
        var_pred: str,
        regions: xr.DataArray | regionmask.Regions,
        lon: str = "lon",
        lat: str = "lat",
        metrics: Sequence[str] = ("rmse", "bias", "correlation", "explained_variance"),
        region_dim: str = "region",
    ) -> None:
        self.var_ref = var_ref
        self.var_pred = var_pred
        self.regions = regions
        self.lon = lon
        self.lat = lat
        self.metrics = tuple(metrics)
        self.region_dim = region_dim

    def _apply(self, ds_track: xr.Dataset) -> xr.Dataset:
        return scores_by_region(
            ds_track,
            var_ref=self.var_ref,
            var_pred=self.var_pred,
            regions=self.regions,
            lon=self.lon,
            lat=self.lat,
            metrics=self.metrics,
            region_dim=self.region_dim,
        )

    def get_config(self) -> dict[str, Any]:
        # ``regions`` is an opaque object (xr.DataArray or
        # regionmask.Regions) and is not JSON-serializable. Mirror the
        # pattern used by RegridLike: emit a stable summary so the config
        # is JSON-safe, and document that callers must re-supply the
        # actual regions object when reconstructing the operator.
        if isinstance(self.regions, xr.DataArray):
            regions_summary: dict[str, Any] = {
                "kind": "DataArray",
                "name": self.regions.name,
                "dims": list(self.regions.dims),
            }
        elif isinstance(self.regions, regionmask.Regions):
            regions_summary = {
                "kind": "regionmask.Regions",
                "name": self.regions.name,
                "regions": list(self.regions.names),
            }
        else:
            regions_summary = {"kind": type(self.regions).__name__}
        return {
            "var_ref": self.var_ref,
            "var_pred": self.var_pred,
            "regions": regions_summary,
            "lon": self.lon,
            "lat": self.lat,
            "metrics": list(self.metrics),
            "region_dim": self.region_dim,
        }


def _region_labels(
    ds_track: xr.Dataset,
    regions: xr.DataArray | regionmask.Regions,
    *,
    var_ref: str,
    lon: str,
    lat: str,
) -> tuple[xr.DataArray, list[str]]:
    if isinstance(regions, xr.DataArray):
        return regions.broadcast_like(ds_track[var_ref]), []

    if type(regions).__module__.startswith("regionmask"):
        import warnings

        if isinstance(regions, regionmask.Regions):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                mask = regions.mask(ds_track[lon], ds_track[lat], method="shapely")
            # Broadcast the regionmask result against the data so extra
            # dims on var_ref (e.g. ensemble) line up with region labels.
            mask = mask.broadcast_like(ds_track[var_ref])
            number_to_name = {
                number: str(name)
                for number, name in zip(regions.numbers, regions.names, strict=True)
            }
            names = list(number_to_name.values())
            renamed = xr.full_like(mask, "", dtype=object)
            for number, name in number_to_name.items():
                renamed = xr.where(mask == number, name, renamed)
            return renamed.where(mask.notnull()), names

    raise TypeError(
        f"Unsupported regions type {type(regions).__name__!r}; expected "
        "regionmask.Regions or a categorical xr.DataArray."
    )


def _score_region(
    ref: Float[np.ndarray, "n"], pred: Float[np.ndarray, "n"], metric: str
) -> float:
    if ref.size == 0:
        # ``count`` is a sample size, so an empty selection means zero
        # matched points — not a missing measurement. All other metrics
        # are undefined on an empty sample and stay NaN.
        return 0.0 if metric == "count" else np.nan
    residual = pred - ref
    if metric == "rmse":
        return float(np.sqrt(np.mean(residual**2)))
    if metric == "bias":
        return float(np.mean(residual))
    if metric == "mae":
        return float(np.mean(np.abs(residual)))
    if metric == "correlation":
        var_ref = np.var(ref)
        var_pred = np.var(pred)
        if ref.size < 2 or np.isclose(var_ref, 0.0) or np.isclose(var_pred, 0.0):
            return np.nan
        return float(np.corrcoef(pred, ref)[0, 1])
    if metric == "r2":
        denom = np.sum((ref - np.mean(ref)) ** 2)
        return (
            np.nan
            if np.isclose(denom, 0.0)
            else float(1.0 - np.sum(residual**2) / denom)
        )
    if metric == "explained_variance":
        denom = np.var(ref)
        return (
            np.nan if np.isclose(denom, 0.0) else float(1.0 - np.var(residual) / denom)
        )
    if metric == "count":
        return float(ref.size)
    raise ValueError(f"Unsupported region score metric {metric!r}.")


__all__ = [
    "BinnedResiduals2D",
    "RegionScores",
    "bin_residuals_2d",
    "scores_by_region",
]
