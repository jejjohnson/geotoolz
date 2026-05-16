"""Carrier-aware normalization operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from geotoolz.core import Operator
from geotoolz.normalize._src.array import (
    asinh_scale,
    histogram_match,
    log_scale,
    minmax_scale,
    per_band_stats,
    percentile_clip,
    power_scale,
    robust_scale,
    standard_scale,
    stat_axes,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


def _jsonable(value: Any) -> float | list[Any] | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


class PerBandStats(Operator):
    """Compute NaN-aware per-band statistics and cache them on the operator."""

    def __init__(self, *, percentiles: list[float] | None = None) -> None:
        self.percentiles = [1.0, 99.0] if percentiles is None else list(percentiles)
        self.stats: dict[str, Any] = {}

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        stats = per_band_stats(arr, percentiles=self.percentiles, axis=stat_axes(arr))
        self.stats = {key: _jsonable(value) for key, value in stats.items()}
        return gt

    def get_config(self) -> dict[str, Any]:
        return {"percentiles": self.percentiles}


class StandardScaler(Operator):
    """Apply per-band z-score scaling using provided or fitted statistics."""

    def __init__(
        self,
        *,
        mean: np.ndarray | list[float] | float | None = None,
        std: np.ndarray | list[float] | float | None = None,
        fit_on_call: bool = False,
    ) -> None:
        self.mean = _array_or_none(mean)
        self.std = _array_or_none(std)
        self.fit_on_call = fit_on_call

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        axis = stat_axes(arr)
        if self.fit_on_call and (self.mean is None or self.std is None):
            self.mean = np.nanmean(arr, axis=axis)
            self.std = np.nanstd(arr, axis=axis)
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler requires mean/std or fit_on_call=True")
        return gt.array_as_geotensor(
            standard_scale(arr, self.mean, self.std, axis=axis)
        )

    def inverse(self, gt: GeoTensor) -> GeoTensor:
        """Invert a previously applied standard scaling."""
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler must be fitted before inverse()")
        arr = np.asarray(gt, dtype=float)
        axis = stat_axes(arr)
        from geotoolz.normalize._src.array import reshape_stat

        mean = reshape_stat(self.mean, arr, axis)
        std = reshape_stat(self.std, arr, axis)
        scale = np.where(std != 0, std, 1.0)
        return gt.array_as_geotensor(arr * scale + mean)

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": _jsonable(self.mean),
            "std": _jsonable(self.std),
            "fit_on_call": self.fit_on_call,
        }


class RobustScaler(Operator):
    """Apply per-band median/IQR scaling using provided or fitted statistics."""

    def __init__(
        self,
        *,
        median: np.ndarray | list[float] | float | None = None,
        iqr: np.ndarray | list[float] | float | None = None,
        fit_on_call: bool = False,
    ) -> None:
        self.median = _array_or_none(median)
        self.iqr = _array_or_none(iqr)
        self.fit_on_call = fit_on_call

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        axis = stat_axes(arr)
        if self.fit_on_call and (self.median is None or self.iqr is None):
            q25, q50, q75 = np.nanpercentile(arr, [25.0, 50.0, 75.0], axis=axis)
            self.median = q50
            self.iqr = q75 - q25
        if self.median is None or self.iqr is None:
            raise ValueError("RobustScaler requires median/iqr or fit_on_call=True")
        return gt.array_as_geotensor(
            robust_scale(arr, self.median, self.iqr, axis=axis)
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "median": _jsonable(self.median),
            "iqr": _jsonable(self.iqr),
            "fit_on_call": self.fit_on_call,
        }


class MinMaxScaler(Operator):
    """Apply per-band min-max scaling into a requested output range."""

    def __init__(
        self,
        *,
        vmin: np.ndarray | list[float] | float | None = None,
        vmax: np.ndarray | list[float] | float | None = None,
        out_range: tuple[float, float] = (0.0, 1.0),
        fit_on_call: bool = False,
    ) -> None:
        self.vmin = _array_or_none(vmin)
        self.vmax = _array_or_none(vmax)
        self.out_range = tuple(out_range)
        self.fit_on_call = fit_on_call

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        axis = stat_axes(arr)
        if self.fit_on_call and (self.vmin is None or self.vmax is None):
            self.vmin = np.nanmin(arr, axis=axis)
            self.vmax = np.nanmax(arr, axis=axis)
        if self.vmin is None or self.vmax is None:
            raise ValueError("MinMaxScaler requires vmin/vmax or fit_on_call=True")
        out = minmax_scale(
            arr, self.vmin, self.vmax, out_range=self.out_range, axis=axis
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "vmin": _jsonable(self.vmin),
            "vmax": _jsonable(self.vmax),
            "out_range": self.out_range,
            "fit_on_call": self.fit_on_call,
        }


class PercentileClip(Operator):
    """Clip percentile tails and stretch the remaining values into ``[0, 1]``."""

    def __init__(
        self, *, lower: float = 1.0, upper: float = 99.0, per_band: bool = True
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.per_band = per_band

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        out = percentile_clip(
            arr,
            lower=self.lower,
            upper=self.upper,
            axis=stat_axes(arr, per_band=self.per_band),
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"lower": self.lower, "upper": self.upper, "per_band": self.per_band}


class HistogramStretch(Operator):
    """Percentile stretch into ``out_range`` for visualisation."""

    def __init__(
        self,
        *,
        out_range: tuple[float, float] = (0.0, 1.0),
        lower: float = 2.0,
        upper: float = 98.0,
    ) -> None:
        self.out_range = tuple(out_range)
        self.lower = lower
        self.upper = upper

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        clipped = percentile_clip(
            arr, lower=self.lower, upper=self.upper, axis=stat_axes(arr)
        )
        out_min, out_max = self.out_range
        if out_max <= out_min:
            raise ValueError(f"out_range must be increasing; got {self.out_range}")
        return gt.array_as_geotensor(clipped * (out_max - out_min) + out_min)

    def get_config(self) -> dict[str, Any]:
        return {"out_range": self.out_range, "lower": self.lower, "upper": self.upper}


class HistogramMatch(Operator):
    """Match a GeoTensor histogram to a reference GeoTensor."""

    forbid_in_yaml = True

    def __init__(self, *, reference: GeoTensor) -> None:
        self.reference = reference

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = histogram_match(
            np.asarray(gt, dtype=float), np.asarray(self.reference, dtype=float)
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"reference_shape": tuple(self.reference.shape)}


class LogScale(Operator):
    """Apply logarithmic scaling with an epsilon offset for zeros."""

    def __init__(self, *, base: float = 10.0, eps: float = 1e-6) -> None:
        self.base = base
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            log_scale(np.asarray(gt, dtype=float), base=self.base, eps=self.eps)
        )

    def get_config(self) -> dict[str, Any]:
        return {"base": self.base, "eps": self.eps}


class AsinhScale(Operator):
    """Apply ``asinh(x / a)`` scaling."""

    def __init__(self, *, a: float = 1.0) -> None:
        self.a = a

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(asinh_scale(np.asarray(gt, dtype=float), a=self.a))

    def get_config(self) -> dict[str, Any]:
        return {"a": self.a}


class PowerScale(Operator):
    """Apply non-negative power scaling."""

    def __init__(self, *, gamma: float = 0.5) -> None:
        self.gamma = gamma

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            power_scale(np.asarray(gt, dtype=float), gamma=self.gamma)
        )

    def get_config(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class Normalize(StandardScaler):
    """Convenience alias for a stateless standard scaler."""

    def __init__(
        self,
        *,
        mean: np.ndarray | list[float] | float,
        std: np.ndarray | list[float] | float,
    ) -> None:
        super().__init__(mean=mean, std=std, fit_on_call=False)


class ZeroOne(Operator):
    """Scale the current scene extent into ``[0, 1]``."""

    def __init__(self, *, per_band: bool = True) -> None:
        self.per_band = per_band

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        axis = stat_axes(arr, per_band=self.per_band)
        out = minmax_scale(
            arr,
            np.nanmin(arr, axis=axis),
            np.nanmax(arr, axis=axis),
            out_range=(0.0, 1.0),
            axis=axis,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"per_band": self.per_band}
