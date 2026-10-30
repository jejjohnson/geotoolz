"""Layer-1 ``Operator`` wrappers around the :mod:`xrops.wavelets._src` primitives.

Each class is a thin adapter: store configuration, implement
``_apply``, return a JSON-serializable ``get_config``. They all inherit
from :class:`pipekit.Operator`, so they compose with
:class:`~pipekit.Sequential`, the ``|`` pipe, and the functional
:class:`~pipekit.Graph` API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr

from xrops._src.operator import Operator
from xrops.wavelets._src import (
    along_track as _along_track,
    wavelet as _wavelet,
    wavelet1d as _wavelet1d,
)


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
        variable: Name of the variable to transform.
        scales: Wavelet scales (a sequence or DataArray).
        dim: The two spatial dims ``(y, x)``.
        x0: Reference length scale (m).
        ntheta: Number of orientation angles.
        k0: Morlet central wavenumber.
        isotropic: Average over orientation when ``True``.
        output_var: Name for the output variable (default
            ``"<variable>_wpsd"``).

    Returns:
        The input dataset with the wavelet power-spectrum variable added.
    """

    def __init__(
        self,
        variable: str,
        scales: Sequence[float] | xr.DataArray,
        *,
        dim: tuple[str, str] = ("y", "x"),
        x0: float = 50e3,
        ntheta: int = 16,
        k0: float = 1.0,
        isotropic: bool = True,
        output_var: str | None = None,
    ) -> None:
        self.variable = variable
        self.scales = scales
        self.dim = tuple(dim)
        self.x0 = float(x0)
        self.ntheta = int(ntheta)
        self.k0 = float(k0)
        self.isotropic = bool(isotropic)
        self.output_var = output_var

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        if self.variable not in ds.data_vars:
            raise KeyError(f"Dataset missing variable {self.variable!r}")
        out_name = self.output_var or f"{self.variable}_wpsd"
        spectrum = _wavelet.wvlt_power_spectrum(
            ds[self.variable],
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
            "variable": self.variable,
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
    The prefix defaults to ``variable`` and is overridable via
    ``output_prefix``.
    """

    def __init__(
        self,
        variable: str,
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
        self.variable = variable
        self.dim = dim
        self.mother = mother
        self.param = param
        self.s0 = s0
        self.dj = float(dj)
        self.j_max = j_max
        self.rectify = bool(rectify)
        self.output_prefix = output_prefix

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        if self.variable not in ds.data_vars:
            raise KeyError(f"Dataset missing variable {self.variable!r}")
        prefix = self.output_prefix or self.variable
        out = _wavelet1d.cwt1d(
            ds[self.variable],
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
            "variable": self.variable,
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
        variable: Name of the wavelet-power variable to test.
        dim_time: Time dimension name.
        dim_scale: Scale dimension name.
        null: Null spectrum — ``"red"`` (AR(1)) or ``"white"`` noise.
        alpha: Lag-1 autocorrelation; ``None`` uses the stored value.
        confidence: Chi-square confidence level.
        mother: Mother wavelet name.
        param: Optional mother parameter.
        output_var: Name for the mask variable (default
            ``"<variable>_signif_mask"``).

    Returns:
        The input dataset with a boolean significance-mask variable added.
    """

    def __init__(
        self,
        variable: str,
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
        self.variable = variable
        self.dim_time = dim_time
        self.dim_scale = dim_scale
        self.null = null
        self.alpha = alpha
        self.confidence = float(confidence)
        self.mother = mother
        self.param = param
        self.output_var = output_var

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        if self.variable not in ds.data_vars:
            raise KeyError(f"Dataset missing variable {self.variable!r}")
        out_name = self.output_var or f"{self.variable}_signif_mask"
        mask = _wavelet1d.wavelet_significance(
            ds[self.variable],
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
            "variable": self.variable,
            "dim_time": self.dim_time,
            "dim_scale": self.dim_scale,
            "null": self.null,
            "alpha": self.alpha,
            "confidence": self.confidence,
            "mother": self.mother,
            "param": self.param,
            "output_var": self.output_var,
        }


__all__ = [
    "BandpassWavelength",
    "WaveletPowerSpectrum",
    "WaveletScalogram",
    "WaveletSignificance",
]
