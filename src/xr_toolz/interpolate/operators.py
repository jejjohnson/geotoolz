"""Layer-1 ``Operator`` wrappers around :mod:`xr_toolz.interpolate._src`.

Each class is a thin adapter: store configuration, implement
``_apply``, return a JSON-serializable ``get_config``. They all inherit
from :class:`xr_toolz.core.Operator`, so they compose with
:class:`~xr_toolz.core.Sequential`, the ``|`` pipe, and the functional
:class:`~xr_toolz.core.Graph` API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from xr_toolz.core import Operator
from xr_toolz.interpolate._src import (
    binning as _binning,
    coord_remap as _coord_remap,
    downscale as _downscale,
    gap_fill as _gap_fill,
    grid_to_grid as _grid_to_grid,
    points_to_grid as _points_to_grid,
    resample as _resample,
    smooth as _smooth,
)


# ---------- gap fill -------------------------------------------------------


class FillNaNSpatial(Operator):
    """Wrap :func:`xr_toolz.interpolate.fillnan_spatial`."""

    def __init__(self, method: str = "linear", lon: str = "lon", lat: str = "lat"):
        self.method = method
        self.lon = lon
        self.lat = lat

    def _apply(self, da):
        return _gap_fill.fillnan_spatial(
            da, method=self.method, lon=self.lon, lat=self.lat
        )

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "lon": self.lon, "lat": self.lat}


class FillNaNTemporal(Operator):
    """Wrap :func:`xr_toolz.interpolate.fillnan_temporal`."""

    def __init__(
        self,
        method: str = "linear",
        time: str = "time",
        max_gap: Any = None,
    ):
        self.method = method
        self.time = time
        self.max_gap = max_gap

    def _apply(self, ds):
        return _gap_fill.fillnan_temporal(
            ds, method=self.method, time=self.time, max_gap=self.max_gap
        )

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "time": self.time, "max_gap": self.max_gap}


class FillNaNRBF(Operator):
    """Wrap :func:`xr_toolz.interpolate.fillnan_rbf`."""

    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        neighbors: int | None = 32,
        lon: str = "lon",
        lat: str = "lat",
    ):
        self.kernel = kernel
        self.neighbors = neighbors
        self.lon = lon
        self.lat = lat

    def _apply(self, da):
        return _gap_fill.fillnan_rbf(
            da,
            kernel=self.kernel,
            neighbors=self.neighbors,
            lon=self.lon,
            lat=self.lat,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "neighbors": self.neighbors,
            "lon": self.lon,
            "lat": self.lat,
        }


# ---------- resample -------------------------------------------------------


class ResampleTime(Operator):
    """Wrap :func:`xr_toolz.interpolate.resample_time`."""

    def __init__(self, freq: str = "1D", method: str = "mean", time: str = "time"):
        self.freq = freq
        self.method = method
        self.time = time

    def _apply(self, ds):
        return _resample.resample_time(
            ds, freq=self.freq, method=self.method, time=self.time
        )

    def get_config(self) -> dict[str, Any]:
        return {"freq": self.freq, "method": self.method, "time": self.time}


# ---------- grid-to-grid ---------------------------------------------------


class Coarsen(Operator):
    """Wrap :func:`xr_toolz.interpolate.coarsen`."""

    def __init__(
        self,
        factor: dict[str, int],
        method: str = "mean",
        boundary: str = "trim",
    ):
        self.factor = dict(factor)
        self.method = method
        self.boundary = boundary

    def _apply(self, ds):
        return _grid_to_grid.coarsen(
            ds, factor=self.factor, method=self.method, boundary=self.boundary
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "factor": dict(self.factor),
            "method": self.method,
            "boundary": self.boundary,
        }


class Refine(Operator):
    """Wrap :func:`xr_toolz.interpolate.refine`."""

    def __init__(self, factor: dict[str, int], method: str = "linear"):
        self.factor = dict(factor)
        self.method = method

    def _apply(self, ds):
        return _grid_to_grid.refine(ds, factor=self.factor, method=self.method)

    def get_config(self) -> dict[str, Any]:
        return {"factor": dict(self.factor), "method": self.method}


# ---------- binning --------------------------------------------------------


class Bin2D(Operator):
    """Wrap :func:`xr_toolz.interpolate.bin_2d`."""

    def __init__(
        self,
        grid: _binning.Grid,
        statistic: str = "mean",
        lon: str = "lon",
        lat: str = "lat",
    ):
        self.grid = grid
        self.statistic = statistic
        self.lon = lon
        self.lat = lat

    def _apply(self, da):
        return _binning.bin_2d(
            da,
            grid=self.grid,
            statistic=self.statistic,
            lon=self.lon,
            lat=self.lat,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "grid": "<Grid>",
            "statistic": self.statistic,
            "lon": self.lon,
            "lat": self.lat,
        }


class Histogram2D(Operator):
    """Wrap :func:`xr_toolz.interpolate.histogram_2d`."""

    def __init__(self, grid: _binning.Grid, lon: str = "lon", lat: str = "lat"):
        self.grid = grid
        self.lon = lon
        self.lat = lat

    def _apply(self, da):
        return _binning.histogram_2d(da, grid=self.grid, lon=self.lon, lat=self.lat)

    def get_config(self) -> dict[str, Any]:
        return {"grid": "<Grid>", "lon": self.lon, "lat": self.lat}


# ---------- points → grid --------------------------------------------------


class PointsToGrid(Operator):
    """Wrap :func:`xr_toolz.interpolate.points_to_grid`.

    Expects a 3-tuple ``(lons, lats, values)`` as input.
    """

    def __init__(self, grid: _binning.Grid, statistic: str = "mean"):
        self.grid = grid
        self.statistic = statistic

    def _apply(self, payload):
        lons, lats, values = payload
        return _points_to_grid.points_to_grid(
            lons, lats, values, grid=self.grid, statistic=self.statistic
        )

    def get_config(self) -> dict[str, Any]:
        return {"grid": "<Grid>", "statistic": self.statistic}


# ---------- smoothers ------------------------------------------------------


class MovingAverage(Operator):
    """Wrap :func:`xr_toolz.interpolate._src.smooth.moving_average`."""

    def __init__(
        self,
        dim: str,
        window: int,
        *,
        center: bool = True,
        min_periods: int | None = None,
    ):
        if not isinstance(window, int) or isinstance(window, bool):
            raise TypeError(f"window must be an int, got {type(window).__name__}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if min_periods is not None and (
            not isinstance(min_periods, int) or min_periods < 0
        ):
            raise ValueError(
                f"min_periods must be a non-negative int or None, got {min_periods!r}"
            )
        self.dim = dim
        self.window = window
        self.center = bool(center)
        self.min_periods = min_periods

    def _apply(self, ds):
        return _smooth.moving_average(
            ds,
            dim=self.dim,
            window=self.window,
            center=self.center,
            min_periods=self.min_periods,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "window": self.window,
            "center": self.center,
            "min_periods": self.min_periods,
        }


class GaussianSmooth(Operator):
    """Wrap :func:`xr_toolz.interpolate._src.smooth.gaussian_smooth`."""

    def __init__(self, dim: str, sigma: float, *, truncate: float = 4.0):
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.dim = dim
        self.sigma = float(sigma)
        self.truncate = float(truncate)

    def _apply(self, ds):
        return _smooth.gaussian_smooth(
            ds, dim=self.dim, sigma=self.sigma, truncate=self.truncate
        )

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim, "sigma": self.sigma, "truncate": self.truncate}


class LowpassFilter(Operator):
    """Wrap :func:`xr_toolz.interpolate._src.smooth.lowpass_filter`.

    For ``btype`` in ``{"low", "high", "lowpass", "highpass"}`` ``cutoff``
    is a scalar in ``(0, 1)``. For ``btype`` in
    ``{"bandpass", "bandstop"}`` it is a length-2 sequence. Validation
    is delegated to the Tier A kernel.
    """

    def __init__(
        self,
        dim: str,
        cutoff: Any,
        *,
        order: int = 4,
        btype: str = "low",
    ):
        if not isinstance(order, int) or isinstance(order, bool):
            raise TypeError(f"order must be an int, got {type(order).__name__}")
        self.dim = dim
        if np.isscalar(cutoff):
            self.cutoff: Any = float(cutoff)
        else:
            pair = tuple(float(v) for v in cutoff)
            if len(pair) != 2:
                raise ValueError(f"cutoff sequence must have length 2, got {len(pair)}")
            self.cutoff = pair
        self.order = order
        self.btype = btype

    def _apply(self, ds):
        return _smooth.lowpass_filter(
            ds,
            dim=self.dim,
            cutoff=self.cutoff,
            order=self.order,
            btype=self.btype,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "cutoff": (
                list(self.cutoff) if isinstance(self.cutoff, tuple) else self.cutoff
            ),
            "order": self.order,
            "btype": self.btype,
        }


# ---------- coord remap ----------------------------------------------------


class RemapAxis(Operator):
    """Generic axis remapping (D12).

    Replaces the ``source_axis`` dimension in the input Dataset with a
    new dimension whose coordinate values are ``target_axis``. Every
    numeric variable that carries ``source_axis`` is interpolated onto
    the target axis.

    Parameters
    ----------
    source_axis
        Name of the existing dimension to remap.
    target_axis
        Target coordinate values. If an :class:`xr.DataArray`, its
        ``.name`` becomes the new dim name; otherwise the new dim name
        defaults to ``target_name`` or ``source_axis``.
    target_name
        Optional explicit new dim name.
    method
        ``"linear"`` or ``"nearest"``.
    """

    def __init__(
        self,
        source_axis: str,
        target_axis: xr.DataArray | np.ndarray | list,
        *,
        target_name: str | None = None,
        method: str = "linear",
    ):
        self.source_axis = source_axis
        if isinstance(target_axis, xr.DataArray):
            self._target_da = target_axis
            self._target_values: np.ndarray = np.asarray(
                target_axis.values, dtype=float
            )
            self._inferred_name = target_axis.name
        else:
            self._target_da = None
            self._target_values = np.asarray(target_axis, dtype=float)
            self._inferred_name = None
        self.target_name = target_name
        self.method = method

    def _apply(self, ds):
        target = self._target_da if self._target_da is not None else self._target_values
        return _coord_remap.remap_axis(
            ds,
            source_dim=self.source_axis,
            target_coords=target,
            target_name=self.target_name,
            method=self.method,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "source_axis": self.source_axis,
            "target_axis": self._target_values.tolist(),
            "target_name": self.target_name or self._inferred_name,
            "method": self.method,
        }


# Vertical presets — thin specializations that pin convention names. The
# user supplies the target coordinate values; the preset names the new
# dim and the source dim conventionally.


class _VerticalPreset(RemapAxis):
    """Common base for vertical-axis presets — pins ``target_name`` only."""

    _DEFAULT_TARGET_NAME: str = ""
    _DEFAULT_SOURCE: str = "depth"

    def __init__(
        self,
        target_axis: xr.DataArray | np.ndarray | list,
        *,
        source_axis: str | None = None,
        target_name: str | None = None,
        method: str = "linear",
    ):
        super().__init__(
            source_axis=source_axis or self._DEFAULT_SOURCE,
            target_axis=target_axis,
            target_name=target_name or self._DEFAULT_TARGET_NAME,
            method=method,
        )


class ToSigma(_VerticalPreset):
    """Remap a depth axis to terrain-following ``sigma`` values."""

    _DEFAULT_TARGET_NAME = "sigma"
    _DEFAULT_SOURCE = "depth"


class FromSigma(_VerticalPreset):
    """Remap a ``sigma`` axis back to a fixed depth grid."""

    _DEFAULT_TARGET_NAME = "depth"
    _DEFAULT_SOURCE = "sigma"


class ToIsopycnal(_VerticalPreset):
    """Remap a depth axis to potential-density (isopycnal) levels."""

    _DEFAULT_TARGET_NAME = "sigma_theta"
    _DEFAULT_SOURCE = "depth"


class ToPressureLevels(_VerticalPreset):
    """Remap a height/depth axis to standard pressure levels."""

    _DEFAULT_TARGET_NAME = "pressure"
    _DEFAULT_SOURCE = "level"


class ToHeight(_VerticalPreset):
    """Remap a pressure or hybrid axis to geometric height."""

    _DEFAULT_TARGET_NAME = "height"
    _DEFAULT_SOURCE = "level"


class ToPhase(Operator):
    """Fold a time axis onto a phase axis by binning + averaging.

    Phase is computed as ``((t - epoch) / period) mod 1`` and binned
    into ``n_bins`` evenly-spaced bins on ``[0, 1)``.

    Parameters
    ----------
    time_dim
        Name of the time dimension.
    period
        Length of one cycle, in the same units as the time coordinate.
    n_bins
        Number of phase bins.
    epoch
        Reference time at which phase = 0.
    """

    def __init__(
        self,
        time_dim: str,
        period: float,
        n_bins: int,
        *,
        epoch: float = 0.0,
    ):
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        self.time_dim = time_dim
        self.period = float(period)
        self.n_bins = int(n_bins)
        self.epoch = float(epoch)

    def _apply(self, ds):
        return _coord_remap.to_phase(
            ds,
            time_dim=self.time_dim,
            period=self.period,
            n_bins=self.n_bins,
            epoch=self.epoch,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "time_dim": self.time_dim,
            "period": self.period,
            "n_bins": self.n_bins,
            "epoch": self.epoch,
        }


# ---------- learned resolution change --------------------------------------

# Re-export Downscale / Upscale from _src.downscale so all Layer-1 Operators
# are reachable from xr_toolz.interpolate.operators.
Downscale = _downscale.Downscale
Upscale = _downscale.Upscale


__all__ = [
    "Bin2D",
    "Coarsen",
    "Downscale",
    "FillNaNRBF",
    "FillNaNSpatial",
    "FillNaNTemporal",
    "FromSigma",
    "GaussianSmooth",
    "Histogram2D",
    "LowpassFilter",
    "MovingAverage",
    "PointsToGrid",
    "Refine",
    "RemapAxis",
    "ResampleTime",
    "ToHeight",
    "ToIsopycnal",
    "ToPhase",
    "ToPressureLevels",
    "ToSigma",
    "Upscale",
]
