"""Layer-1 ``Operator`` wrappers around :mod:`xr_toolz.interpolate._src`.

Each class is a thin adapter: store configuration, implement
``_apply``, return a JSON-serializable ``get_config``. They all inherit
from :class:`xr_toolz.core.Operator`, so they compose with
:class:`~xr_toolz.core.Sequential`, the ``|`` pipe, and the functional
:class:`~xr_toolz.core.Graph` API.
"""

from __future__ import annotations

from typing import Any

from xr_toolz.core import Operator
from xr_toolz.interpolate._src import (
    binning as _binning,
    gap_fill as _gap_fill,
    grid_to_grid as _grid_to_grid,
    points_to_grid as _points_to_grid,
    resample as _resample,
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


__all__ = [
    "Bin2D",
    "Coarsen",
    "FillNaNRBF",
    "FillNaNSpatial",
    "FillNaNTemporal",
    "Histogram2D",
    "PointsToGrid",
    "Refine",
    "ResampleTime",
]
