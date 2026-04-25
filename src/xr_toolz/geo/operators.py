"""Layer-1 ``Operator`` wrappers around the :mod:`xr_toolz.geo._src` primitives.

Each class is a thin adapter: store configuration, implement
``_apply``, return a JSON-serializable ``get_config``. They all inherit
from :class:`xr_toolz.core.Operator`, so they compose with
:class:`~xr_toolz.core.Sequential`, the ``|`` pipe, and the functional
:class:`~xr_toolz.core.Graph` API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xr_toolz.core import Operator
from xr_toolz.geo._src import (
    detrend as _detrend,
    interpolate as _interpolate,
    masks as _masks,
    metrics as _metrics,
    subset as _subset,
    validation as _validation,
)


# ---------- validation -----------------------------------------------------


class ValidateLongitude(Operator):
    """Wrap :func:`xr_toolz.geo.validate_longitude`."""

    def _apply(self, ds):
        return _validation.validate_longitude(ds)


class ValidateLatitude(Operator):
    """Wrap :func:`xr_toolz.geo.validate_latitude`."""

    def _apply(self, ds):
        return _validation.validate_latitude(ds)


class ValidateCoords(Operator):
    """Apply longitude and latitude validation in one pass."""

    def _apply(self, ds):
        ds = _validation.validate_longitude(ds)
        return _validation.validate_latitude(ds)


class RenameCoords(Operator):
    """Wrap :func:`xr_toolz.geo.rename_coords`."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)

    def _apply(self, ds):
        return _validation.rename_coords(ds, self.mapping)

    def get_config(self) -> dict[str, Any]:
        return {"mapping": self.mapping}


# ---------- subset ---------------------------------------------------------


class SubsetBBox(Operator):
    def __init__(
        self,
        lon_bnds: tuple[float, float],
        lat_bnds: tuple[float, float],
        lon: str = "lon",
        lat: str = "lat",
    ):
        self.lon_bnds = tuple(lon_bnds)
        self.lat_bnds = tuple(lat_bnds)
        self.lon = lon
        self.lat = lat

    def _apply(self, ds):
        return _subset.subset_bbox(
            ds,
            lon_bnds=self.lon_bnds,
            lat_bnds=self.lat_bnds,
            lon=self.lon,
            lat=self.lat,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "lon_bnds": list(self.lon_bnds),
            "lat_bnds": list(self.lat_bnds),
            "lon": self.lon,
            "lat": self.lat,
        }


class SubsetTime(Operator):
    def __init__(self, time_min: str, time_max: str, time: str = "time"):
        self.time_min = time_min
        self.time_max = time_max
        self.time = time

    def _apply(self, ds):
        return _subset.subset_time(
            ds, time_min=self.time_min, time_max=self.time_max, time=self.time
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "time_min": self.time_min,
            "time_max": self.time_max,
            "time": self.time,
        }


class SelectVariables(Operator):
    def __init__(self, variables: str | Sequence[str]):
        self.variables = [variables] if isinstance(variables, str) else list(variables)

    def _apply(self, ds):
        return _subset.select_variables(ds, self.variables)

    def get_config(self) -> dict[str, Any]:
        return {"variables": list(self.variables)}


# ---------- detrend --------------------------------------------------------


class CalculateClimatology(Operator):
    """Return a climatology at ``freq`` from the input dataset."""

    def __init__(self, freq: str = "day", time: str = "time"):
        self.freq = freq
        self.time = time

    def _apply(self, ds):
        return _detrend.calculate_climatology(ds, freq=self.freq, time=self.time)

    def get_config(self) -> dict[str, Any]:
        return {"freq": self.freq, "time": self.time}


class CalculateClimatologySmoothed(Operator):
    def __init__(self, window: int = 60, time: str = "time"):
        self.window = window
        self.time = time

    def _apply(self, ds):
        return _detrend.calculate_climatology_smoothed(
            ds, window=self.window, time=self.time
        )

    def get_config(self) -> dict[str, Any]:
        return {"window": self.window, "time": self.time}


class RemoveClimatology(Operator):
    """Subtract a precomputed climatology from the input dataset."""

    def __init__(self, climatology, time: str = "time"):
        self.climatology = climatology
        self.time = time

    def _apply(self, ds):
        return _detrend.remove_climatology(ds, self.climatology, time=self.time)

    def get_config(self) -> dict[str, Any]:
        # climatology is rich state — referenced rather than serialized
        return {"climatology": "<xr object>", "time": self.time}


class AddClimatology(Operator):
    """Inverse of :class:`RemoveClimatology`."""

    def __init__(self, climatology, time: str = "time"):
        self.climatology = climatology
        self.time = time

    def _apply(self, ds):
        return _detrend.add_climatology(ds, self.climatology, time=self.time)

    def get_config(self) -> dict[str, Any]:
        return {"climatology": "<xr object>", "time": self.time}


# ---------- masks ----------------------------------------------------------


class AddLandMask(Operator):
    def __init__(self, name: str = "land_mask"):
        self.name = name

    def _apply(self, ds):
        return _masks.add_land_mask(ds, name=self.name)

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}


class AddOceanMask(Operator):
    def __init__(self, ocean: str = "global", name: str = "ocean_mask"):
        self.ocean = ocean
        self.name = name

    def _apply(self, ds):
        return _masks.add_ocean_mask(ds, ocean=self.ocean, name=self.name)

    def get_config(self) -> dict[str, Any]:
        return {"ocean": self.ocean, "name": self.name}


class AddCountryMask(Operator):
    def __init__(self, country: str, name: str = "country_mask"):
        self.country = country
        self.name = name

    def _apply(self, ds):
        return _masks.add_country_mask(ds, country=self.country, name=self.name)

    def get_config(self) -> dict[str, Any]:
        return {"country": self.country, "name": self.name}


class ApplyMask(Operator):
    def __init__(self, mask, drop: bool = False):
        self.mask = mask
        self.drop = drop

    def _apply(self, ds):
        return _masks.apply_mask(ds, self.mask, drop=self.drop)

    def get_config(self) -> dict[str, Any]:
        mask_repr = self.mask if isinstance(self.mask, str) else "<DataArray>"
        return {"mask": mask_repr, "drop": self.drop}


# ---------- interpolate ----------------------------------------------------


class FillNaNSpatial(Operator):
    def __init__(self, method: str = "linear", lon: str = "lon", lat: str = "lat"):
        self.method = method
        self.lon = lon
        self.lat = lat

    def _apply(self, da):
        return _interpolate.fillnan_spatial(
            da, method=self.method, lon=self.lon, lat=self.lat
        )

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "lon": self.lon, "lat": self.lat}


class FillNaNTemporal(Operator):
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
        return _interpolate.fillnan_temporal(
            ds, method=self.method, time=self.time, max_gap=self.max_gap
        )

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "time": self.time, "max_gap": self.max_gap}


class ResampleTime(Operator):
    def __init__(self, freq: str = "1D", method: str = "mean", time: str = "time"):
        self.freq = freq
        self.method = method
        self.time = time

    def _apply(self, ds):
        return _interpolate.resample_time(
            ds, freq=self.freq, method=self.method, time=self.time
        )

    def get_config(self) -> dict[str, Any]:
        return {"freq": self.freq, "method": self.method, "time": self.time}


# ---------- pixel metrics (multi-input) -----------------------------------


class _PixelMetricOp(Operator):
    """Base class for two-input pixel metrics."""

    _fn: Any = None

    def __init__(self, variable: str, dims: str | Sequence[str]):
        self.variable = variable
        self.dims = dims if isinstance(dims, str) else list(dims)

    def _apply(self, ds_pred, ds_ref):
        return self.__class__._fn(ds_pred, ds_ref, self.variable, self.dims)

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "dims": self.dims if isinstance(self.dims, str) else list(self.dims),
        }


class MSE(_PixelMetricOp):
    _fn = staticmethod(_metrics.mse)


class RMSE(_PixelMetricOp):
    _fn = staticmethod(_metrics.rmse)


class NRMSE(_PixelMetricOp):
    _fn = staticmethod(_metrics.nrmse)


class MAE(_PixelMetricOp):
    _fn = staticmethod(_metrics.mae)


class Bias(_PixelMetricOp):
    _fn = staticmethod(_metrics.bias)


class Correlation(_PixelMetricOp):
    _fn = staticmethod(_metrics.correlation)


class R2Score(_PixelMetricOp):
    _fn = staticmethod(_metrics.r2_score)


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
        return _metrics.psd_score(
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
    "MAE",
    "MSE",
    "NRMSE",
    "RMSE",
    "AddClimatology",
    "AddCountryMask",
    "AddLandMask",
    "AddOceanMask",
    "ApplyMask",
    "Bias",
    "CalculateClimatology",
    "CalculateClimatologySmoothed",
    "Correlation",
    "FillNaNSpatial",
    "FillNaNTemporal",
    "PSDScore",
    "R2Score",
    "RemoveClimatology",
    "RenameCoords",
    "ResampleTime",
    "SelectVariables",
    "SubsetBBox",
    "SubsetTime",
    "ValidateCoords",
    "ValidateLatitude",
    "ValidateLongitude",
]
