"""Tier-B Operators — carrier-aware radiometric transforms.

Each Operator wraps a primitive in
:mod:`geotoolz.radiometry._src.array`. Most are ufunc-pure (arithmetic,
power, clipping) so the carrier's ``__array_ufunc__`` round-trips
``transform`` / ``crs`` / ``fill_value_default`` for free; the
non-ufunc cases (``PercentileClip``) explicitly wrap via
``array_as_geotensor``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from georeader.reflectance import (
    compute_sza,
    earth_sun_distance_correction_factor,
    integrated_irradiance,
    load_thuillier_irradiance,
    radiance_to_reflectance,
    reflectance_to_radiance,
    srf,
    transform_to_srf,
)

from geotoolz.core import Operator
from geotoolz.radiometry._src.array import (
    _broadcast_to_band_axis,
    bt_from_radiance,
    dn_to_radiance,
    dn_to_reflectance,
    dos1,
    gamma_correct,
    min_max_normalize,
    percentile_clip,
    radiance_to_dn,
)


def _coef_as_jsonable(coef: Any) -> float | list[float]:
    """Coerce a gain / offset coefficient into a JSON-safe scalar or list.

    Keeps scalars as Python ``float``; converts numpy scalars, ndarrays
    and any sequence (list / tuple) of numerics into ``list[float]``.
    Used by `DNToRadiance` / `DNToReflectance`'s ``get_config()`` so
    Hydra / YAML round-trips don't choke on ndarray leaves.
    """
    arr = np.asarray(coef)
    if arr.ndim == 0:
        return float(arr)
    return [float(v) for v in arr.ravel()]


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


class ToFloat32(Operator):
    """Cast the carrier's values to ``float32``.

    The first stop on most radiometric pipelines: sensor DN are usually
    ``uint16``, and downstream arithmetic (division for indices, gain
    application, etc.) wants float. ``float32`` is the right default for
    imagery — half the memory of ``float64`` with plenty of dynamic
    range for reflectance.

    Pure ufunc — `GeoTensor.__array_ufunc__` preserves transform / CRS /
    fill-value automatically.

    Examples:
        >>> import geotoolz as gz
        >>> # Standard first stage of a TOA-reflectance pipeline:
        >>> pipe = (
        ...     gz.radiometry.ToFloat32()
        ...     | gz.radiometry.DNToReflectance(scale=1e-4)
        ... )
        >>> reflectance = pipe(uint16_dn_geotensor)
    """

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        # `astype` on a GeoTensor returns a GeoTensor with metadata
        # preserved via __array_finalize__.
        return gt.astype(np.float32)

    def get_config(self) -> dict[str, Any]:
        return {}


class DNToRadiance(Operator):
    r"""Convert DN to at-sensor radiance.

    .. math::

        L \;=\; \text{gain} \cdot DN + \text{offset}

    Pure linear decode; gain and offset come from sensor metadata. Pass
    scalars for uniform per-pixel coefficients, or 1-D sequences with
    one entry per band to apply different coefficients along the band
    axis.

    See :func:`~geotoolz.radiometry._src.array.dn_to_radiance` for the
    physics. The Operator additionally reshapes 1-D coefficients so
    they broadcast along the configured ``axis``.

    Args:
        gain: Slope of the DN→L decode. Scalar or per-band sequence.
        offset: Intercept. Default ``0.0``.
        axis: Position of the band axis when ``gain`` / ``offset`` are
            per-band sequences. Default ``0``.

    Examples:
        >>> import numpy as np
        >>> from geotoolz.radiometry import DNToRadiance
        >>> # Landsat-8 OLI per-band radiance multiplicative + additive
        >>> # constants from the MTL file (just illustrative numbers):
        >>> gains   = np.array([0.012, 0.013, 0.011, 0.009])
        >>> offsets = np.array([-60.0, -61.0, -55.0, -45.0])
        >>> dn_to_rad = DNToRadiance(gain=gains, offset=offsets)
        >>> radiance = dn_to_rad(dn_geotensor)  # shape preserved
    """

    def __init__(
        self,
        *,
        gain: float | np.ndarray | list,
        offset: float | np.ndarray | list = 0.0,
        scale: float | np.ndarray | list = 1.0,
        axis: int = 0,
    ) -> None:
        self.gain = gain
        self.offset = offset
        self.scale = scale
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt)
        n_bands = arr.shape[self.axis] if arr.ndim > 2 else 1
        gain = _broadcast_to_band_axis(self.gain, n_bands, self.axis, arr.ndim)
        offset = _broadcast_to_band_axis(self.offset, n_bands, self.axis, arr.ndim)
        scale = _broadcast_to_band_axis(self.scale, n_bands, self.axis, arr.ndim)
        out = dn_to_radiance(arr, gain, offset, scale)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "gain": _coef_as_jsonable(self.gain),
            "offset": _coef_as_jsonable(self.offset),
            "scale": _coef_as_jsonable(self.scale),
            "axis": self.axis,
        }


class RadianceToDN(Operator):
    """Convert at-sensor radiance back to DN with `DNToRadiance`'s inverse."""

    def __init__(
        self,
        *,
        gain: float | np.ndarray | list,
        offset: float | np.ndarray | list = 0.0,
        scale: float | np.ndarray | list = 1.0,
        axis: int = 0,
    ) -> None:
        self.gain = gain
        self.offset = offset
        self.scale = scale
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt)
        n_bands = arr.shape[self.axis] if arr.ndim > 2 else 1
        gain = _broadcast_to_band_axis(self.gain, n_bands, self.axis, arr.ndim)
        offset = _broadcast_to_band_axis(self.offset, n_bands, self.axis, arr.ndim)
        scale = _broadcast_to_band_axis(self.scale, n_bands, self.axis, arr.ndim)
        out = radiance_to_dn(arr, gain, offset, scale)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "gain": _coef_as_jsonable(self.gain),
            "offset": _coef_as_jsonable(self.offset),
            "scale": _coef_as_jsonable(self.scale),
            "axis": self.axis,
        }


class DNToReflectance(Operator):
    r"""Convert DN to TOA / surface reflectance via a linear affine decode.

    .. math::

        \rho \;=\; \text{scale} \cdot DN + \text{offset}

    ``scale`` is the slope (reflectance per DN unit) and ``offset`` is
    the *reflectance-units* intercept — the canonical
    ``y = m * x + b`` form. Matches Landsat Collection-2 SR verbatim
    and absorbs the Sentinel-2 L1C ``RADIO_ADD_OFFSET`` after
    multiplying it through the scale.

    For sensors without a pre-scaled-reflectance product (raw radiance
    only), use `DNToRadiance` then call
    `georeader.reflectance.radiance_to_reflectance` (which handles
    solar geometry properly).

    Args:
        scale: Quantification slope (reflectance per DN unit). Scalar
            or per-band 1-D sequence.
        offset: Reflectance-units intercept. Default ``0.0``.
        axis: Band axis when ``scale`` / ``offset`` are per-band.

    Examples:
        >>> from geotoolz.radiometry import DNToReflectance
        >>> # Sentinel-2 L1C pre-2022: a single global scale, no offset.
        >>> op = DNToReflectance(scale=1e-4)
        >>> reflectance = op(s2_l1c_dn_geotensor)
        >>>
        >>> # Post-2022 S2 L1C: RADIO_ADD_OFFSET=-1000 in DN units
        >>> # collapses to -0.1 in reflectance units (-1000 * 1e-4).
        >>> op_v2 = DNToReflectance(scale=1e-4, offset=-0.1)
        >>> reflectance = op_v2(s2_l1c_modern_dn_geotensor)
        >>>
        >>> # Landsat-8/9 Collection-2 surface reflectance.
        >>> op_l8 = DNToReflectance(scale=2.75e-5, offset=-0.2)
        >>> reflectance = op_l8(landsat_c2_sr_geotensor)
    """

    def __init__(
        self,
        *,
        scale: float | np.ndarray | list,
        offset: float | np.ndarray | list = 0.0,
        axis: int = 0,
    ) -> None:
        self.scale = scale
        self.offset = offset
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt)
        n_bands = arr.shape[self.axis] if arr.ndim > 2 else 1
        scale = _broadcast_to_band_axis(self.scale, n_bands, self.axis, arr.ndim)
        offset = _broadcast_to_band_axis(self.offset, n_bands, self.axis, arr.ndim)
        out = dn_to_reflectance(arr, scale, offset)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "scale": _coef_as_jsonable(self.scale),
            "offset": _coef_as_jsonable(self.offset),
            "axis": self.axis,
        }


def _observation_date_corr_factor(
    acquisition_date: datetime,
    sza_deg: float | None,
) -> float | None:
    if sza_deg is None:
        return None
    d = earth_sun_distance_correction_factor(acquisition_date)
    return float(np.pi * (d**2) / np.cos(np.deg2rad(sza_deg)))


class RadianceToReflectance(Operator):
    """Convert at-sensor radiance to TOA reflectance with solar geometry."""

    def __init__(
        self,
        *,
        solar_irradiance: np.ndarray | list,
        acquisition_date: datetime,
        center_coords: tuple[float, float] | None = None,
        sza_deg: float | None = None,
        crs_coords: str | None = None,
        units: str = "W/m2/sr/nm",
    ) -> None:
        self.solar_irradiance = solar_irradiance
        self.acquisition_date = acquisition_date
        self.center_coords = center_coords
        self.sza_deg = sza_deg
        self.crs_coords = crs_coords
        self.units = units

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = radiance_to_reflectance(
            gt,
            solar_irradiance=self.solar_irradiance,
            date_of_acquisition=self.acquisition_date,
            center_coords=self.center_coords,
            crs_coords=self.crs_coords,
            observation_date_corr_factor=_observation_date_corr_factor(
                self.acquisition_date, self.sza_deg
            ),
            units=self.units,
        )
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "solar_irradiance": _coef_as_jsonable(self.solar_irradiance),
            "acquisition_date": self.acquisition_date,
            "center_coords": self.center_coords,
            "sza_deg": self.sza_deg,
            "crs_coords": self.crs_coords,
            "units": self.units,
        }


class ReflectanceToRadiance(Operator):
    """Convert TOA reflectance back to at-sensor radiance."""

    def __init__(
        self,
        *,
        solar_irradiance: np.ndarray | list,
        acquisition_date: datetime,
        center_coords: tuple[float, float] | None = None,
        sza_deg: float | None = None,
        crs_coords: str | None = None,
    ) -> None:
        self.solar_irradiance = solar_irradiance
        self.acquisition_date = acquisition_date
        self.center_coords = center_coords
        self.sza_deg = sza_deg
        self.crs_coords = crs_coords

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = reflectance_to_radiance(
            gt,
            solar_irradiance=self.solar_irradiance,
            date_of_acquisition=self.acquisition_date,
            center_coords=self.center_coords,
            crs_coords=self.crs_coords,
            observation_date_corr_factor=_observation_date_corr_factor(
                self.acquisition_date, self.sza_deg
            ),
        )
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "solar_irradiance": _coef_as_jsonable(self.solar_irradiance),
            "acquisition_date": self.acquisition_date,
            "center_coords": self.center_coords,
            "sza_deg": self.sza_deg,
            "crs_coords": self.crs_coords,
        }


class EarthSunDistanceCorrection(Operator):
    """Compute the Earth-Sun distance correction factor for an acquisition date."""

    def __init__(self, *, acquisition_date: datetime) -> None:
        self.acquisition_date = acquisition_date

    def _apply(self, _gt: Any | None = None) -> float:
        return float(earth_sun_distance_correction_factor(self.acquisition_date))

    def get_config(self) -> dict[str, Any]:
        return {"acquisition_date": self.acquisition_date}


class ComputeSZA(Operator):
    """Compute solar zenith angle in degrees for a location and acquisition date."""

    def __init__(
        self,
        *,
        center_coords: tuple[float, float],
        acquisition_date: datetime,
        crs_coords: str | None = None,
    ) -> None:
        self.center_coords = center_coords
        self.acquisition_date = acquisition_date
        self.crs_coords = crs_coords

    def _apply(self, _gt: Any | None = None) -> float:
        return float(
            compute_sza(
                self.center_coords,
                self.acquisition_date,
                crs_coords=self.crs_coords,
            )
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "center_coords": self.center_coords,
            "acquisition_date": self.acquisition_date,
            "crs_coords": self.crs_coords,
        }


class IntegratedIrradiance(Operator):
    """Compute band-integrated solar irradiance from an SRF table."""

    def __init__(
        self,
        *,
        srf: pd.DataFrame,
        solar_irradiance: pd.DataFrame | None = None,
        epsilon_srf: float = 1e-4,
    ) -> None:
        self.srf = srf
        self.solar_irradiance = solar_irradiance
        self.epsilon_srf = epsilon_srf

    def _apply(self, _gt: Any | None = None) -> np.ndarray:
        solar_irradiance = (
            load_thuillier_irradiance()
            if self.solar_irradiance is None
            else self.solar_irradiance
        )
        return integrated_irradiance(
            self.srf,
            solar_irradiance=solar_irradiance,
            epsilon_srf=self.epsilon_srf,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "srf": self.srf,
            "solar_irradiance": self.solar_irradiance,
            "epsilon_srf": self.epsilon_srf,
        }


class ApplySRF(Operator):
    """Convolve hyperspectral data to target Gaussian spectral response bands."""

    def __init__(
        self,
        *,
        target_center_wavelengths: np.ndarray | list,
        target_fwhm: np.ndarray | list,
        source_wavelengths: np.ndarray | list,
        epsilon_srf: float = 1e-4,
        extrapolate: bool = False,
    ) -> None:
        self.target_center_wavelengths = target_center_wavelengths
        self.target_fwhm = target_fwhm
        self.source_wavelengths = source_wavelengths
        self.epsilon_srf = epsilon_srf
        self.extrapolate = extrapolate

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        source_wavelengths = np.asarray(self.source_wavelengths, dtype=float)
        wavelengths = np.arange(
            np.floor(source_wavelengths.min()),
            np.ceil(source_wavelengths.max()) + 1,
        )
        srf_values = srf(
            self.target_center_wavelengths,
            self.target_fwhm,
            wavelengths,
        )
        srf_df = pd.DataFrame(srf_values, index=wavelengths)
        out = transform_to_srf(
            np.asarray(gt),
            srf_df,
            source_wavelengths.tolist(),
            fill_value_default=gt.fill_value_default,
            epsilon_srf=self.epsilon_srf,
            extrapolate=self.extrapolate,
        )
        fill_value = gt.fill_value_default
        if fill_value is not None:
            missing = np.any(np.asarray(gt) == fill_value, axis=0)
            out[:, missing] = fill_value
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "target_center_wavelengths": _coef_as_jsonable(
                self.target_center_wavelengths
            ),
            "target_fwhm": _coef_as_jsonable(self.target_fwhm),
            "source_wavelengths": _coef_as_jsonable(self.source_wavelengths),
            "epsilon_srf": self.epsilon_srf,
            "extrapolate": self.extrapolate,
        }


class BTFromRadiance(Operator):
    """Convert thermal radiance to brightness temperature in Kelvin."""

    def __init__(
        self,
        *,
        K1: float | np.ndarray | list,
        K2: float | np.ndarray | list,
        axis: int = 0,
    ) -> None:
        self.K1 = K1
        self.K2 = K2
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt)
        n_bands = arr.shape[self.axis] if arr.ndim > 2 else 1
        k1 = _broadcast_to_band_axis(self.K1, n_bands, self.axis, arr.ndim)
        k2 = _broadcast_to_band_axis(self.K2, n_bands, self.axis, arr.ndim)
        fill_value = gt.fill_value_default
        work = arr
        if fill_value is not None:
            work = np.where(arr == fill_value, np.nan, arr)
        out = bt_from_radiance(work, k1, k2)
        if fill_value is not None:
            out[arr == fill_value] = fill_value
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "K1": _coef_as_jsonable(self.K1),
            "K2": _coef_as_jsonable(self.K2),
            "axis": self.axis,
        }


class DOS1(Operator):
    """Apply a Chavez-style DOS1 dark-object subtraction approximation."""

    def __init__(self, *, dark_percentile: float = 1.0) -> None:
        self.dark_percentile = dark_percentile

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        fill_value = gt.fill_value_default
        valid = None if fill_value is None else arr != fill_value
        work = arr if valid is None else np.where(valid, arr, np.nan)
        out = dos1(work, dark_percentile=self.dark_percentile, axis=(-2, -1))
        if valid is not None:
            out[~valid] = fill_value
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"dark_percentile": self.dark_percentile}


class SimpleAtmosphericCorrection(Operator):
    """Dispatch simple BOA approximations; currently supports ``method='dos1'``."""

    def __init__(
        self,
        *,
        method: str = "dos1",
        dark_percentile: float = 1.0,
        aod: float | None = None,
    ) -> None:
        self.method = method
        self.dark_percentile = dark_percentile
        self.aod = aod

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        if self.method.lower() != "dos1":
            raise NotImplementedError(
                "SimpleAtmosphericCorrection currently supports only method='dos1'"
            )
        return DOS1(dark_percentile=self.dark_percentile)(gt)

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "dark_percentile": self.dark_percentile,
            "aod": self.aod,
        }


class MinMax(Operator):
    r"""Linear contrast stretch into ``[0, 1]``.

    .. math::

        y \;=\; \frac{x - v_{\min}}{v_{\max} - v_{\min}}

    Display-prep — no physics; just a fixed-bound rescale. See
    `PercentileClip` for the more robust per-scene variant.

    Args:
        vmin: Lower bound (maps to 0).
        vmax: Upper bound (maps to 1). Must be strictly greater than
            ``vmin``.
        clip: Whether to clamp output to ``[0, 1]``. Default ``True``.

    Examples:
        >>> from geotoolz.radiometry import MinMax
        >>> # Stretch reflectance in [0, 0.3] to display range.
        >>> op = MinMax(vmin=0.0, vmax=0.3)
        >>> display_ready = op(reflectance_geotensor)
    """

    def __init__(self, *, vmin: float, vmax: float, clip: bool = True) -> None:
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = min_max_normalize(np.asarray(gt), self.vmin, self.vmax, clip=self.clip)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"vmin": self.vmin, "vmax": self.vmax, "clip": self.clip}


class PercentileClip(Operator):
    r"""Per-band robust contrast stretch using percentile thresholds.

    Computes :math:`v_{lo} = P_{p_{\min}}(\text{arr})` and
    :math:`v_{hi} = P_{p_{\max}}(\text{arr})` over the configured
    ``axis`` and rescales each slice into ``[0, 1]``.

    Robust against bright outliers (cumulus, specular reflection, sensor
    saturation): a tiny sub-percent population of bright pixels won't
    crush the rest of the histogram the way a true min/max would.

    Args:
        p_min: Lower percentile. Default ``2.0``.
        p_max: Upper percentile. Default ``98.0``.
        axis: Axis (or tuple) to compute percentiles over.
            ``(-2, -1)`` is per-band/-time. ``None`` is global.

    Examples:
        >>> from geotoolz.radiometry import PercentileClip
        >>> # Standard "satellite RGB" stretch -- per-band 2-98 %.
        >>> op = PercentileClip(p_min=2.0, p_max=98.0)
        >>> rgb = op(reflectance_geotensor)
    """

    def __init__(
        self,
        *,
        p_min: float = 2.0,
        p_max: float = 98.0,
        axis: int | tuple[int, ...] | None = (-2, -1),
    ) -> None:
        self.p_min = p_min
        self.p_max = p_max
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = percentile_clip(
            np.asarray(gt), p_min=self.p_min, p_max=self.p_max, axis=self.axis
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"p_min": self.p_min, "p_max": self.p_max, "axis": self.axis}


class Gamma(Operator):
    r"""Power-law gamma correction.

    .. math::

        y \;=\; x^{1/\gamma}

    Display-prep — ``g > 1`` brightens midtones, ``g < 1`` darkens
    them. A gentle ``g = 1.2`` is the workhorse "satellite RGB pop"
    default; sRGB encoding uses ``g ≈ 2.2``.

    Args:
        g: Gamma factor (must be strictly positive). Default ``1.2``.

    Examples:
        >>> import geotoolz as gz
        >>> # Classic display pipeline: stretch, then gamma-brighten.
        >>> pipe = (
        ...     gz.radiometry.PercentileClip()
        ...     | gz.radiometry.Gamma(g=1.4)
        ... )
        >>> rgb = pipe(reflectance_geotensor)
    """

    def __init__(self, *, g: float = 1.2) -> None:
        self.g = g

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = gamma_correct(np.asarray(gt), g=self.g)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"g": self.g}
