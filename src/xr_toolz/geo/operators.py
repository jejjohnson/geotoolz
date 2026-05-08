"""Layer-1 ``Operator`` wrappers around the :mod:`xr_toolz.geo._src` primitives.

Each class is a thin adapter: store configuration, implement
``_apply``, return a JSON-serializable ``get_config``. They all inherit
from :class:`xr_toolz.core.Operator`, so they compose with
:class:`~xr_toolz.core.Sequential`, the ``|`` pipe, and the functional
:class:`~xr_toolz.core.Graph` API.

Metric operators (``MSE``, ``RMSE``, …, ``PSDScore``) moved to
:mod:`xr_toolz.metrics.operators`. They remain importable from this
module for one release with a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

from xr_toolz.core import Operator, Signature
from xr_toolz.geo._src import (
    along_track as _along_track,
    detrend as _detrend,
    masks as _masks,
    subset as _subset,
    validation as _validation,
)


# ---------- validation -----------------------------------------------------


def _validate_signature(
    input_signature: Signature, *, aliases: tuple[str, ...], canonical: str
) -> Signature:
    """Mirror the runtime rename done by ``validate_longitude`` /
    ``validate_latitude``: if any alias is present in the input dims,
    rename it to ``canonical``. Shape-preserving otherwise."""
    for alias in aliases:
        if alias in input_signature.dims:
            return input_signature.rename_dims({alias: canonical})
    return input_signature


class ValidateLongitude(Operator):
    """Wrap :func:`xr_toolz.geo.validate_longitude`."""

    def _apply(self, ds):
        return _validation.validate_longitude(ds)

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return _validate_signature(
            input_signature,
            aliases=_validation._LONGITUDE_ALIASES,
            canonical="lon",
        )


class ValidateLatitude(Operator):
    """Wrap :func:`xr_toolz.geo.validate_latitude`."""

    def _apply(self, ds):
        return _validation.validate_latitude(ds)

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return _validate_signature(
            input_signature,
            aliases=_validation._LATITUDE_ALIASES,
            canonical="lat",
        )


class ValidateCoords(Operator):
    """Apply longitude and latitude validation in one pass."""

    def _apply(self, ds):
        ds = _validation.validate_longitude(ds)
        return _validation.validate_latitude(ds)

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        signature = _validate_signature(
            input_signature,
            aliases=_validation._LONGITUDE_ALIASES,
            canonical="lon",
        )
        return _validate_signature(
            signature,
            aliases=_validation._LATITUDE_ALIASES,
            canonical="lat",
        )


class DecodeCFTime(Operator):
    """Wrap :func:`xr_toolz.geo.decode_cf_time`."""

    def __init__(self, *, time: str = "time", units: str | None = None):
        self.time = time
        self.units = units

    def _apply(self, ds):
        return _validation.decode_cf_time(ds, time=self.time, units=self.units)

    def get_config(self) -> dict[str, Any]:
        return {"time": self.time, "units": self.units}


class ValidateTime(Operator):
    """Wrap :func:`xr_toolz.geo.validate_time`."""

    def __init__(
        self,
        *,
        time: str = "time",
        unit: str | None = None,
        origin: str = "unix",
    ):
        self.time = time
        self.unit = unit
        self.origin = origin

    def _apply(self, ds):
        return _validation.validate_time(
            ds, time=self.time, unit=self.unit, origin=self.origin
        )

    def get_config(self) -> dict[str, Any]:
        return {"time": self.time, "unit": self.unit, "origin": self.origin}


class RenameCoords(Operator):
    """Wrap :func:`xr_toolz.geo.rename_coords`."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)

    def _apply(self, ds):
        return _validation.rename_coords(ds, self.mapping)

    def get_config(self) -> dict[str, Any]:
        return {"mapping": self.mapping}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature.rename_dims(self.mapping)


class RenameVariables(Operator):
    """Wrap :func:`xr_toolz.geo.rename_variables` (data-var renames)."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)

    def _apply(self, ds):
        return _validation.rename_variables(ds, self.mapping)

    def get_config(self) -> dict[str, Any]:
        return {"mapping": self.mapping}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature


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

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature.replace_dims({self.lon: None, self.lat: None})


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

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature.replace_dims({self.time: None})


class SelectVariables(Operator):
    def __init__(self, variables: str | Sequence[str]):
        self.variables = [variables] if isinstance(variables, str) else list(variables)

    def _apply(self, ds):
        return _subset.select_variables(ds, self.variables)

    def get_config(self) -> dict[str, Any]:
        return {"variables": list(self.variables)}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature


# ---------- detrend --------------------------------------------------------


class CalculateClimatology(Operator):
    """Return a climatology at ``freq`` from the input dataset."""

    def __init__(self, freq: str = "day", time: str = "time"):
        if freq not in _detrend.CLIMATOLOGY_DIMS:
            raise ValueError(
                f"Unsupported climatology frequency {freq!r}; expected one of "
                f"{sorted(_detrend.CLIMATOLOGY_DIMS)}."
            )
        self.freq = freq
        self.time = time

    def _apply(self, ds):
        return _detrend.calculate_climatology(ds, freq=self.freq, time=self.time)

    def get_config(self) -> dict[str, Any]:
        return {"freq": self.freq, "time": self.time}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        dim = _detrend.CLIMATOLOGY_DIMS[self.freq]
        return _replace_dim(input_signature, old=self.time, new=dim, size=None)


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

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        # Smoothed climatology always groups by day-of-year; pull the
        # canonical dim name from the same source the runtime uses
        # rather than hard-coding the literal here.
        new_dim = _detrend.CLIMATOLOGY_DIMS["day"]
        return _replace_dim(input_signature, old=self.time, new=new_dim, size=None)


class RemoveMean(Operator):
    """Subtract the mean over ``dims`` (cheap anomaly without climatology)."""

    def __init__(self, dims: str | tuple[str, ...]):
        self.dims = (dims,) if isinstance(dims, str) else tuple(dims)

    def _apply(self, ds):
        return _detrend.remove_mean(ds, self.dims)

    def get_config(self) -> dict[str, Any]:
        return {"dims": list(self.dims)}


# ---------- generic transforms --------------------------------------------


class FillNaN(Operator):
    """Replace NaN values with a constant.

    Promotes the inline op used in the V1.5 PSD demo to a library
    primitive. ``xr_toolz.geo._src.interpolation`` provides
    *interpolating* NaN fillers (``FillNaNSpatial`` / ``FillNaNTemporal``);
    this op is the constant-fill counterpart — useful for zeroing land
    cells before PSD scoring so they contribute no spectral energy.

    Args:
        value: Constant to substitute for NaN. Default ``0.0``.
    """

    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def _apply(self, ds):
        return ds.fillna(self.value)

    def get_config(self) -> dict[str, Any]:
        return {"value": self.value}


_REDUCE_OPS = ("mean", "sum", "median", "min", "max", "std", "var")


class Reduce(Operator):
    """Reduce a Dataset over one or more dims with a named aggregation.

    Wraps the corresponding ``xr.Dataset`` reducer (``mean`` / ``sum`` /
    …). Promotes the inline ``MeanOverDim`` from the V1.5 PSD demo so
    notebooks no longer redefine a one-method subclass per reduction.

    Args:
        op: Aggregation name. One of ``"mean"``, ``"sum"``, ``"median"``,
            ``"min"``, ``"max"``, ``"std"``, ``"var"``.
        dim: Dim or tuple of dims to reduce over.
        keepdims: Whether to keep reduced dims as length-1 (forwarded to
            the underlying xarray reducer's ``keepdims`` kwarg).
    """

    def __init__(
        self,
        op: str = "mean",
        dim: str | tuple[str, ...] = ("time",),
        *,
        keepdims: bool = False,
    ) -> None:
        if op not in _REDUCE_OPS:
            raise ValueError(
                f"Unknown reduce op {op!r}; expected one of {_REDUCE_OPS}."
            )
        self.op = op
        self.dim = (dim,) if isinstance(dim, str) else tuple(dim)
        self.keepdims = bool(keepdims)

    def _apply(self, ds):
        reducer = getattr(ds, self.op)
        return reducer(dim=self.dim, keepdims=self.keepdims)

    def get_config(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "dim": list(self.dim),
            "keepdims": self.keepdims,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        if self.keepdims:
            return input_signature.replace_dims({dim: 1 for dim in self.dim})
        return input_signature.drop_dims(self.dim)


class BandpassWavelength(Operator):
    """Wrap :func:`xr_toolz.geo.bandpass_wavelength`."""

    def __init__(
        self,
        *,
        dim: str,
        lambda_min_km: float | None = None,
        lambda_max_km: float | None = None,
        spacing_km: float | None = None,
        method: str = "lanczos",
        num_taps: int | None = None,
        attenuation_db: float | None = None,
        lon: str = "lon",
        lat: str = "lat",
    ) -> None:
        self.dim = dim
        self.lambda_min_km = lambda_min_km
        self.lambda_max_km = lambda_max_km
        self.spacing_km = spacing_km
        self.method = method
        self.num_taps = num_taps
        self.attenuation_db = attenuation_db
        self.lon = lon
        self.lat = lat

    def _apply(self, ds):
        return _along_track.bandpass_wavelength(
            ds,
            dim=self.dim,
            lambda_min_km=self.lambda_min_km,
            lambda_max_km=self.lambda_max_km,
            spacing_km=self.spacing_km,
            method=self.method,
            num_taps=self.num_taps,
            attenuation_db=self.attenuation_db,
            lon=self.lon,
            lat=self.lat,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "lambda_min_km": self.lambda_min_km,
            "lambda_max_km": self.lambda_max_km,
            "spacing_km": self.spacing_km,
            "method": self.method,
            "num_taps": self.num_taps,
            "attenuation_db": self.attenuation_db,
            "lon": self.lon,
            "lat": self.lat,
        }


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

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        if not self.drop:
            return input_signature
        return Signature(
            {name: None for name in input_signature.dims},
            dtype=input_signature.dtype,
        )


# ---------- deprecated metric ops -----------------------------------------

# Moved to xr_toolz.metrics.operators. Re-export lazily for one release
# so existing user imports continue to work with a deprecation warning.
_DEPRECATED_METRIC_OPS = {
    "MSE",
    "RMSE",
    "NRMSE",
    "MAE",
    "Bias",
    "Correlation",
    "R2Score",
    "PSDScore",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_METRIC_OPS:
        from importlib import import_module

        warnings.warn(
            f"xr_toolz.geo.operators.{name} is deprecated; "
            f"import from xr_toolz.metrics.operators instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name == "PSDScore":
            module = import_module("xr_toolz.metrics._src.spectral")
        else:
            module = import_module("xr_toolz.metrics._src.pixel")
        return getattr(module, name)
    raise AttributeError(f"module 'xr_toolz.geo.operators' has no attribute {name!r}")


__all__ = [
    "AddClimatology",
    "AddCountryMask",
    "AddLandMask",
    "AddOceanMask",
    "ApplyMask",
    "BandpassWavelength",
    "CalculateClimatology",
    "CalculateClimatologySmoothed",
    "FillNaN",
    "Reduce",
    "RemoveClimatology",
    "RemoveMean",
    "RenameCoords",
    "RenameVariables",
    "SelectVariables",
    "SubsetBBox",
    "SubsetTime",
    "ValidateCoords",
    "ValidateLatitude",
    "ValidateLongitude",
]


def _replace_dim(
    input_signature: Signature,
    *,
    old: str,
    new: str,
    size: int | None,
) -> Signature:
    dims = {}
    for name, dim_size in input_signature.dims.items():
        if name == old:
            dims[new] = size
        else:
            dims[name] = dim_size
    return Signature(dims, dtype=input_signature.dtype)
