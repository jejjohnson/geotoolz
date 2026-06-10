"""Layer-1 ``Operator`` wrappers around the :mod:`xrtoolz.geo._src` primitives.

Each class is a thin adapter: store configuration, implement
``_apply``, return a JSON-serializable ``get_config``. They all inherit
from :class:`pipekit.Operator`, so they compose with
:class:`~pipekit.Sequential`, the ``|`` pipe, and the functional
:class:`~pipekit.Graph` API.

Metric operators (``MSE``, ``RMSE``, …, ``PSDScore``) moved to
:mod:`xrtoolz.metrics.operators`. They remain importable from this
module for one release with a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import regionmask
import xarray as xr

from xrtoolz._operator import Operator
from xrtoolz.geo._src import (
    along_track as _along_track,
    detrend as _detrend,
    masks as _masks,
    regions as _regions,
    subset as _subset,
    validation as _validation,
    wavelet as _wavelet,
    wavelet1d as _wavelet1d,
)
from xrtoolz.signature import Signature


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
    """Validate and CF-normalise the longitude coordinate.

    Renames a recognised longitude alias to ``lon`` and checks the values
    lie in range, raising on malformed input.

    Returns:
        The input dataset with a validated ``lon`` coordinate.
    """

    def _apply(self, ds):
        return _validation.validate_longitude(ds)

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return _validate_signature(
            input_signature,
            aliases=_validation._LONGITUDE_ALIASES,
            canonical="lon",
        )


class ValidateLatitude(Operator):
    """Validate and CF-normalise the latitude coordinate.

    Renames a recognised latitude alias to ``lat`` and checks the values
    lie in ``[-90, 90]``, raising on malformed input.

    Returns:
        The input dataset with a validated ``lat`` coordinate.
    """

    def _apply(self, ds):
        return _validation.validate_latitude(ds)

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return _validate_signature(
            input_signature,
            aliases=_validation._LATITUDE_ALIASES,
            canonical="lat",
        )


class ValidateCoords(Operator):
    """Validate both longitude and latitude in one pass.

    Applies :class:`ValidateLongitude` then :class:`ValidateLatitude`.

    Returns:
        The input dataset with validated ``lon`` and ``lat`` coordinates.
    """

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
    """Decode a CF-encoded time coordinate to ``datetime64``.

    Args:
        time: Name of the time coordinate to decode.
        units: CF time units (e.g. ``"days since 1990-01-01"``); ``None``
            reads them from the coordinate's ``units`` attribute.

    Returns:
        The input dataset with ``time`` decoded to ``datetime64``.
    """

    def __init__(self, *, time: str = "time", units: str | None = None):
        self.time = time
        self.units = units

    def _apply(self, ds):
        return _validation.decode_cf_time(ds, time=self.time, units=self.units)

    def get_config(self) -> dict[str, Any]:
        return {"time": self.time, "units": self.units}


class ValidateTime(Operator):
    """Validate and standardise a time coordinate.

    Args:
        time: Name of the time coordinate.
        unit: Expected time unit; ``None`` infers it.
        origin: Reference epoch for numeric times (default ``"unix"``).

    Returns:
        The input dataset with a validated ``time`` coordinate.
    """

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
    """Rename coordinates by an explicit mapping.

    Args:
        mapping: ``{old_name: new_name}`` coordinate renames.

    Returns:
        The input dataset with the coordinates renamed.
    """

    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)

    def _apply(self, ds):
        return _validation.rename_coords(ds, self.mapping)

    def get_config(self) -> dict[str, Any]:
        return {"mapping": self.mapping}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature.rename_dims(self.mapping)


class RenameVariables(Operator):
    """Rename data variables by an explicit mapping.

    Args:
        mapping: ``{old_name: new_name}`` data-variable renames.

    Returns:
        The input dataset with the data variables renamed.
    """

    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)

    def _apply(self, ds):
        return _validation.rename_variables(ds, self.mapping)

    def get_config(self) -> dict[str, Any]:
        return {"mapping": self.mapping}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature


class RenameToCFStandardNames(Operator):
    """Wrap :func:`xrtoolz.geo.rename_to_cf_standard_names`.

    Renames each variable / coord that carries a ``standard_name`` attr
    to that attr value. Raises ``ValueError`` on collision (two source
    vars mapping to the same ``standard_name``).

    Args:
        include_coords: If ``True`` (default), rename coords too.
    """

    def __init__(self, *, include_coords: bool = True) -> None:
        self.include_coords = include_coords

    def _apply(self, ds):
        return _validation.rename_to_cf_standard_names(
            ds, include_coords=self.include_coords
        )

    def get_config(self) -> dict[str, Any]:
        return {"include_coords": self.include_coords}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature


class RenameFromCFStandardNames(Operator):
    """Wrap :func:`xrtoolz.geo.rename_from_cf_standard_names`.

    Renames CF ``standard_name``-shaped variables to their xrtoolz
    canonical names using the :mod:`xrreader.types.Variable` registry.

    Args:
        fallback: ``"passthrough"`` (default) leaves unrecognized names
            unchanged. ``"raise"`` raises ``KeyError`` on unknown names.
        include_coords: If ``True`` (default), rename coords too.
    """

    def __init__(
        self,
        *,
        fallback: Literal["passthrough", "raise"] = "passthrough",
        include_coords: bool = True,
    ) -> None:
        self.fallback = fallback
        self.include_coords = include_coords

    def _apply(self, ds):
        return _validation.rename_from_cf_standard_names(
            ds,
            fallback=self.fallback,
            include_coords=self.include_coords,
        )

    def get_config(self) -> dict[str, Any]:
        return {"fallback": self.fallback, "include_coords": self.include_coords}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature


# ---------- subset ---------------------------------------------------------


class SubsetBBox(Operator):
    """Subset a dataset to a lon/lat bounding box.

    Args:
        lon_bnds: ``(min, max)`` longitude bounds.
        lat_bnds: ``(min, max)`` latitude bounds.
        lon: Name of the longitude coordinate.
        lat: Name of the latitude coordinate.

    Returns:
        The dataset restricted to the bounding box.
    """

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


class SubsetToRegion(Operator):
    """Subset a dataset to a named or custom geographic region.

    Masks to the region and trims to its bounding box.

    Args:
        region: Region specification — a registry name, a
            :class:`~xrtoolz.geo._src.regions.RegionSpec`, a
            ``regionmask.Regions``, or the dict form from ``get_config()``.
        lon: Name of the longitude coordinate.
        lat: Name of the latitude coordinate.
        validate: Whether to validate coordinates before masking.

    Returns:
        The dataset restricted to the region.
    """

    def __init__(
        self,
        region: str | _regions.RegionSpec | regionmask.Regions | dict[str, Any],
        *,
        lon: str = "lon",
        lat: str = "lat",
        validate: bool = True,
    ):
        # Accept the dict form emitted by ``get_config`` so the standard
        # ``cls(**op.get_config())`` round-trip used by ApplyToEach works
        # for custom regions, not just registry strings.
        if isinstance(region, dict):
            region = _regions.region_from_dict(region)
        self.region = region
        self.lon = lon
        self.lat = lat
        self.validate = validate

    def _apply(self, ds):
        return _subset.subset_to_region(
            ds,
            self.region,
            lon=self.lon,
            lat=self.lat,
            validate=self.validate,
        )

    def get_config(self) -> dict[str, Any]:
        region: str | dict[str, Any]
        if isinstance(self.region, str):
            region = self.region
        else:
            region = _regions.region_to_dict(self.region)
        return {
            "region": region,
            "lon": self.lon,
            "lat": self.lat,
            "validate": self.validate,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature.replace_dims({self.lon: None, self.lat: None})


class SubsetTime(Operator):
    """Subset a dataset to an inclusive ``[time_min, time_max]`` window.

    Args:
        time_min: Start of the time window (ISO string).
        time_max: End of the time window (ISO string).
        time: Name of the time coordinate.

    Returns:
        The dataset restricted to the time window.
    """

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
    """Select a subset of data variables.

    Args:
        variables: A variable name or sequence of names to keep.

    Returns:
        The dataset containing only the selected variables.
    """

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
    """Compute a climatology grouped at ``freq`` from the time axis.

    Args:
        freq: Climatology grouping — e.g. ``"day"`` (day-of-year),
            ``"month"``, ``"season"`` (one of ``CLIMATOLOGY_DIMS``).
        time: Name of the time coordinate.

    Returns:
        A dataset of the per-group climatological means, with ``time``
        replaced by the grouping dimension.
    """

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
    """Compute a day-of-year climatology smoothed with a rolling window.

    Args:
        window: Width (in days) of the centred rolling-mean smoothing
            applied around the day-of-year cycle.
        time: Name of the time coordinate.

    Returns:
        A smoothed day-of-year climatology dataset.
    """

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
    """Subtract the mean over ``dims`` (a cheap anomaly without a climatology).

    Args:
        dims: Dimension or dimensions to average over before subtracting.

    Returns:
        The input dataset with the per-``dims`` mean removed.
    """

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
    primitive. ``xrtoolz.geo._src.interpolation`` provides
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
    """Band-pass a field by wavelength with a windowed-sinc FIR filter.

    Keeps along-``dim`` structure with wavelengths between ``lambda_min_km``
    and ``lambda_max_km``. When ``spacing_km`` is omitted it is derived from
    the ``lon``/``lat`` ride-along coordinates.

    Args:
        dim: Dimension to filter along (e.g. an along-track index).
        lambda_min_km: Shortest wavelength to keep (km); ``None`` for a
            low-pass.
        lambda_max_km: Longest wavelength to keep (km); ``None`` for a
            high-pass.
        spacing_km: Sample spacing (km); ``None`` derives it from lon/lat.
        method: FIR window — ``"lanczos"`` or ``"kaiser"``.
        num_taps: Filter length; ``None`` chooses a default.
        attenuation_db: Stop-band attenuation for the Kaiser window.
        lon: Longitude coordinate name (used when ``spacing_km`` is None).
        lat: Latitude coordinate name (used when ``spacing_km`` is None).

    Returns:
        The band-pass-filtered dataset (each numeric variable along ``dim``).
    """

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
        # Resolve the lon/lat ride-along DataArrays once. They are only
        # needed when ``spacing_km`` is omitted, but doing the lookup
        # here keeps the loop body small.
        lon_da: xr.DataArray | None = None
        lat_da: xr.DataArray | None = None
        if self.spacing_km is None and isinstance(ds, xr.Dataset):
            lon_da = ds[self.lon]
            lat_da = ds[self.lat]
        elif self.spacing_km is None and isinstance(ds, xr.DataArray):
            # DataArray inputs only carry coords, not arbitrary
            # variables; pull lon/lat from there.
            if self.lon in ds.coords:
                lon_da = ds.coords[self.lon]
            if self.lat in ds.coords:
                lat_da = ds.coords[self.lat]

        def _fn(da: xr.DataArray) -> xr.DataArray:
            return _along_track.bandpass_wavelength(
                da,
                dim=self.dim,
                lambda_min_km=self.lambda_min_km,
                lambda_max_km=self.lambda_max_km,
                spacing_km=self.spacing_km,
                method=self.method,
                num_taps=self.num_taps,
                attenuation_db=self.attenuation_db,
                lon=lon_da,
                lat=lat_da,
            )

        if isinstance(ds, xr.DataArray):
            return _fn(ds)

        if self.dim not in ds.dims:
            # The pre-flip Dataset-flavoured primitive raised on a missing
            # dim; preserve that behaviour at the Operator boundary so a
            # misspelled ``dim`` doesn't silently pass every variable
            # through unchanged.
            raise ValueError(f"dim {self.dim!r} not in Dataset dims {tuple(ds.dims)}")

        out_vars: dict[str, xr.DataArray] = {}
        for name, da in ds.data_vars.items():
            if self.dim not in da.dims or not np.issubdtype(da.dtype, np.number):
                out_vars[str(name)] = da
                continue
            out_vars[str(name)] = _fn(da)
        return xr.Dataset(out_vars, coords=ds.coords, attrs=dict(ds.attrs))

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


class WaveletPowerSpectrum(Operator):
    """2-D Morlet wavelet power spectrum of one variable.

    Args:
        var: Name of the variable to transform.
        scales: Wavelet scales (a sequence or DataArray).
        dim: The two spatial dims ``(y, x)``.
        x0: Reference length scale (m).
        ntheta: Number of orientation angles.
        k0: Morlet central wavenumber.
        isotropic: Average over orientation when ``True``.
        output_var: Name for the output variable (default ``"<var>_wpsd"``).

    Returns:
        The input dataset with the wavelet power-spectrum variable added.
    """

    def __init__(
        self,
        var: str,
        scales: Sequence[float] | xr.DataArray,
        *,
        dim: tuple[str, str] = ("y", "x"),
        x0: float = 50e3,
        ntheta: int = 16,
        k0: float = 1.0,
        isotropic: bool = True,
        output_var: str | None = None,
    ) -> None:
        self.var = var
        self.scales = scales
        self.dim = tuple(dim)
        self.x0 = float(x0)
        self.ntheta = int(ntheta)
        self.k0 = float(k0)
        self.isotropic = bool(isotropic)
        self.output_var = output_var

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        if self.var not in ds.data_vars:
            raise KeyError(f"Dataset missing variable {self.var!r}")
        out_name = self.output_var or f"{self.var}_wpsd"
        spectrum = _wavelet.wvlt_power_spectrum(
            ds[self.var],
            self.scales,
            dim=self.dim,
            x0=self.x0,
            ntheta=self.ntheta,
            k0=self.k0,
            isotropic=self.isotropic,
        ).rename(out_name)
        return ds.assign({out_name: spectrum})

    def get_config(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "scales": "<xr object>",
            "dim": list(self.dim),
            "x0": self.x0,
            "ntheta": self.ntheta,
            "k0": self.k0,
            "isotropic": self.isotropic,
            "output_var": self.output_var,
        }


class WaveletScalogram(Operator):
    """Compute a 1-D wavelet scalogram for one variable.

    Adds six derived variables under a common prefix:
    ``<prefix>_wave``, ``<prefix>_power``, ``<prefix>_power_rect``,
    ``<prefix>_scalogram``, ``<prefix>_coi``, ``<prefix>_coi_mask``.
    The prefix defaults to ``var`` and is overridable via ``output_prefix``.
    """

    def __init__(
        self,
        var: str,
        *,
        dim: str = "time",
        mother: str = "morlet",
        param: float | None = None,
        s0: float | None = None,
        dj: float = 0.25,
        j_max: int | None = None,
        rectify: bool = True,
        output_prefix: str | None = None,
    ) -> None:
        self.var = var
        self.dim = dim
        self.mother = mother
        self.param = param
        self.s0 = s0
        self.dj = float(dj)
        self.j_max = j_max
        self.rectify = bool(rectify)
        self.output_prefix = output_prefix

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        if self.var not in ds.data_vars:
            raise KeyError(f"Dataset missing variable {self.var!r}")
        prefix = self.output_prefix or self.var
        out = _wavelet1d.cwt1d(
            ds[self.var],
            dim=self.dim,
            mother=self.mother,
            param=self.param,
            s0=self.s0,
            dj=self.dj,
            j_max=self.j_max,
        )
        power_name = "power_rect" if self.rectify else "power"
        return ds.assign(
            {
                f"{prefix}_wave": out["wave"].rename(f"{prefix}_wave"),
                f"{prefix}_power": out["power"].rename(f"{prefix}_power"),
                f"{prefix}_power_rect": out["power_rect"].rename(
                    f"{prefix}_power_rect"
                ),
                f"{prefix}_scalogram": out[power_name].rename(f"{prefix}_scalogram"),
                f"{prefix}_coi": out["coi"].rename(f"{prefix}_coi"),
                f"{prefix}_coi_mask": out["coi_mask"].rename(f"{prefix}_coi_mask"),
            }
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "dim": self.dim,
            "mother": self.mother,
            "param": self.param,
            "s0": self.s0,
            "dj": self.dj,
            "j_max": self.j_max,
            "rectify": self.rectify,
            "output_prefix": self.output_prefix,
        }


class WaveletSignificance(Operator):
    """Torrence-Compo significance mask for a wavelet-power variable.

    Args:
        var: Name of the wavelet-power variable to test.
        dim_time: Time dimension name.
        dim_scale: Scale dimension name.
        null: Null spectrum — ``"red"`` (AR(1)) or ``"white"`` noise.
        alpha: Lag-1 autocorrelation; ``None`` uses the stored value.
        confidence: Chi-square confidence level.
        mother: Mother wavelet name.
        param: Optional mother parameter.
        output_var: Name for the mask variable (default
            ``"<var>_signif_mask"``).

    Returns:
        The input dataset with a boolean significance-mask variable added.
    """

    def __init__(
        self,
        var: str,
        *,
        dim_time: str = "time",
        dim_scale: str = "scale",
        null: Literal["red", "white"] = "red",
        alpha: float | None = None,
        confidence: float = 0.95,
        mother: str = "morlet",
        param: float | None = None,
        output_var: str | None = None,
    ) -> None:
        self.var = var
        self.dim_time = dim_time
        self.dim_scale = dim_scale
        self.null = null
        self.alpha = alpha
        self.confidence = float(confidence)
        self.mother = mother
        self.param = param
        self.output_var = output_var

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        if self.var not in ds.data_vars:
            raise KeyError(f"Dataset missing variable {self.var!r}")
        out_name = self.output_var or f"{self.var}_signif_mask"
        mask = _wavelet1d.wavelet_significance(
            ds[self.var],
            dim_time=self.dim_time,
            dim_scale=self.dim_scale,
            null=self.null,
            alpha=self.alpha,
            confidence=self.confidence,
            mother=self.mother,
            param=self.param,
        ).rename(out_name)
        return ds.assign({out_name: mask})

    def get_config(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "dim_time": self.dim_time,
            "dim_scale": self.dim_scale,
            "null": self.null,
            "alpha": self.alpha,
            "confidence": self.confidence,
            "mother": self.mother,
            "param": self.param,
            "output_var": self.output_var,
        }


class RemoveClimatology(Operator):
    """Subtract a precomputed climatology to form anomalies.

    Args:
        climatology: Climatology dataset (e.g. from
            :class:`CalculateClimatology`) to subtract.
        time: Name of the time coordinate.

    Returns:
        The anomaly dataset (input minus the matched climatology).
    """

    def __init__(self, climatology, time: str = "time"):
        self.climatology = climatology
        self.time = time

    def _apply(self, ds):
        return _detrend.remove_climatology(ds, self.climatology, time=self.time)

    def get_config(self) -> dict[str, Any]:
        # climatology is rich state — referenced rather than serialized
        return {"climatology": "<xr object>", "time": self.time}


class AddClimatology(Operator):
    """Add a climatology back to anomalies (inverse of :class:`RemoveClimatology`).

    Args:
        climatology: Climatology dataset to add.
        time: Name of the time coordinate.

    Returns:
        The reconstructed dataset (anomaly plus climatology).
    """

    def __init__(self, climatology, time: str = "time"):
        self.climatology = climatology
        self.time = time

    def _apply(self, ds):
        return _detrend.add_climatology(ds, self.climatology, time=self.time)

    def get_config(self) -> dict[str, Any]:
        return {"climatology": "<xr object>", "time": self.time}


# ---------- masks ----------------------------------------------------------


class AddLandMask(Operator):
    """Add a boolean land mask derived from ``regionmask``.

    Args:
        name: Name of the mask variable to add.

    Returns:
        The input dataset with a boolean land-mask variable.
    """

    def __init__(self, name: str = "land_mask"):
        self.name = name

    def _apply(self, ds):
        return _masks.add_land_mask(ds, name=self.name)

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}


class AddOceanMask(Operator):
    """Add a boolean ocean/basin mask derived from ``regionmask``.

    Args:
        ocean: Ocean/basin selector (e.g. ``"global"``).
        name: Name of the mask variable to add.

    Returns:
        The input dataset with a boolean ocean-mask variable.
    """

    def __init__(self, ocean: str = "global", name: str = "ocean_mask"):
        self.ocean = ocean
        self.name = name

    def _apply(self, ds):
        return _masks.add_ocean_mask(ds, ocean=self.ocean, name=self.name)

    def get_config(self) -> dict[str, Any]:
        return {"ocean": self.ocean, "name": self.name}


class AddCountryMask(Operator):
    """Add a boolean country mask derived from ``regionmask``.

    Args:
        country: Country name/selector to mask.
        name: Name of the mask variable to add.

    Returns:
        The input dataset with a boolean country-mask variable.
    """

    def __init__(self, country: str, name: str = "country_mask"):
        self.country = country
        self.name = name

    def _apply(self, ds):
        return _masks.add_country_mask(ds, country=self.country, name=self.name)

    def get_config(self) -> dict[str, Any]:
        return {"country": self.country, "name": self.name}


class ApplyMask(Operator):
    """Apply a boolean mask to a dataset.

    Args:
        mask: A boolean DataArray, or the name of a mask variable already
            in the dataset.
        drop: If ``True``, drop fully-masked slices
            (``where(..., drop=True)``); otherwise keep them as NaN.

    Returns:
        The masked dataset.
    """

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

# Moved to xrtoolz.metrics.operators. Re-export lazily for one release
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
            f"xrtoolz.geo.operators.{name} is deprecated; "
            f"import from xrtoolz.metrics.operators instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name == "PSDScore":
            module = import_module("xrtoolz.metrics._src.spectral")
        else:
            module = import_module("xrtoolz.metrics._src.pixel")
        return getattr(module, name)
    raise AttributeError(f"module 'xrtoolz.geo.operators' has no attribute {name!r}")


__all__ = [
    "AddClimatology",
    "AddCountryMask",
    "AddLandMask",
    "AddOceanMask",
    "ApplyMask",
    "BandpassWavelength",
    "CalculateClimatology",
    "CalculateClimatologySmoothed",
    "DecodeCFTime",
    "FillNaN",
    "Reduce",
    "RemoveClimatology",
    "RemoveMean",
    "RenameCoords",
    "RenameFromCFStandardNames",
    "RenameToCFStandardNames",
    "RenameVariables",
    "SelectVariables",
    "SubsetBBox",
    "SubsetTime",
    "SubsetToRegion",
    "ValidateCoords",
    "ValidateLatitude",
    "ValidateLongitude",
    "ValidateTime",
    "WaveletPowerSpectrum",
    "WaveletScalogram",
    "WaveletSignificance",
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
