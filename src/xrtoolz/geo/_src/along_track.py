"""Along-track geodesic helpers and wavelength-domain filtering."""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from pyproj import Geod

from xrtoolz.interpolate import fir_filter


_GEOD = Geod(ellps="WGS84")


def median_dx_km(lon: ArrayLike, lat: ArrayLike) -> float:
    """Median geodesic spacing between consecutive points in kilometres.

    The inputs must be 1-D — a flattened multi-track array would mix
    endpoints from unrelated tracks into the spacing estimate, biasing
    the inferred FIR cutoff. For multi-track datasets, derive spacing
    per track or pass ``spacing_km`` to :func:`bandpass_wavelength`
    explicitly.

    Args:
        lon: 1-D longitude samples in degrees.
        lat: 1-D latitude samples in degrees, same shape as ``lon``.

    Returns:
        Median WGS-84 geodesic segment length in kilometres.
    """
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    if lon_arr.shape != lat_arr.shape:
        raise ValueError(
            f"lon and lat must have the same shape; got {lon_arr.shape} and "
            f"{lat_arr.shape}"
        )
    if lon_arr.ndim != 1:
        raise ValueError(
            "median_dx_km expects 1-D inputs along a single track; got shape "
            f"{lon_arr.shape}. Pass spacing_km explicitly for multi-track data."
        )
    if lon_arr.size < 2:
        raise ValueError("at least two lon/lat points are required")

    distances_m = np.asarray(_GEOD.line_lengths(lon_arr, lat_arr), dtype=float)
    finite = distances_m[np.isfinite(distances_m)]
    if finite.size == 0:
        raise ValueError("no finite geodesic segment lengths found")
    return float(np.median(finite) / 1000.0)


def _cutoff_from_wavelength(lambda_km: float, spacing_km: float) -> float:
    if lambda_km <= 0:
        raise ValueError(f"wavelength must be > 0 km, got {lambda_km}")
    cutoff = 2.0 * spacing_km / lambda_km
    if not (0.0 < cutoff < 1.0):
        raise ValueError(
            f"wavelength {lambda_km} km is below the Nyquist limit for spacing "
            f"{spacing_km} km; expected lambda_km > {2.0 * spacing_km}"
        )
    return cutoff


def bandpass_wavelength(
    da: xr.DataArray,
    *,
    dim: str,
    lambda_min_km: float | None = None,
    lambda_max_km: float | None = None,
    spacing_km: float | None = None,
    method: str = "lanczos",
    num_taps: int | None = None,
    attenuation_db: float | None = None,
    lon: xr.DataArray | None = None,
    lat: xr.DataArray | None = None,
) -> xr.DataArray:
    """Filter an along-track DataArray using wavelength cutoffs in kilometres.

    Per the PR γ primitive-flip (``docs/design/xarray-native-primitives.md``),
    this Layer-0 primitive operates on a single ``DataArray``. The
    Layer-1 ``BandpassWavelength`` operator handles Dataset selection
    (including per-variable iteration and string ``lon``/``lat`` name
    lookup) and forwards the resulting DataArrays here.

    Args:
        da: Input DataArray. Must carry ``dim`` and a numeric dtype.
        dim: Along-track dimension to filter.
        lambda_min_km: Short-wavelength edge in kilometres. With
            ``lambda_max_km`` this becomes the high-frequency band edge.
        lambda_max_km: Long-wavelength edge in kilometres. With
            ``lambda_min_km`` this becomes the low-frequency band edge.
        spacing_km: Optional along-track sample spacing. If omitted, it is
            derived as the median WGS-84 spacing from ``lon`` / ``lat``.
        method: FIR window family, ``"lanczos"`` or ``"kaiser"``.
        num_taps: Optional odd FIR tap count.
        attenuation_db: Kaiser attenuation target in decibels.
        lon: 1-D longitude DataArray along ``dim``, used to derive
            ``spacing_km`` when it is not supplied explicitly.
        lat: 1-D latitude DataArray along ``dim``, used to derive
            ``spacing_km`` when it is not supplied explicitly.

    Returns:
        DataArray filtered along ``dim``, same shape as ``da``.
    """
    if lambda_min_km is None and lambda_max_km is None:
        raise ValueError("at least one of lambda_min_km or lambda_max_km is required")
    if (
        lambda_min_km is not None
        and lambda_max_km is not None
        and lambda_min_km >= lambda_max_km
    ):
        raise ValueError(
            f"lambda_min_km must be < lambda_max_km; got {lambda_min_km} >= "
            f"{lambda_max_km}"
        )
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not in DataArray dims {tuple(da.dims)}")
    if spacing_km is None:
        if lon is None or lat is None:
            raise ValueError(
                "spacing_km must be provided when lon / lat DataArrays are not "
                "supplied for spacing inference"
            )
        # Restrict spacing inference to the filter dim only; otherwise a
        # 2-D coord (multi-track grid) would mix unrelated endpoints into
        # the median.
        if lon.dims != (dim,) or lat.dims != (dim,):
            raise ValueError(
                "Cannot infer spacing: lon/lat are not 1-D along "
                f"{dim!r} (got dims {lon.dims} / {lat.dims}). Pass "
                "spacing_km explicitly."
            )
        spacing_km = median_dx_km(lon.values, lat.values)
    if spacing_km <= 0:
        raise ValueError(f"spacing_km must be > 0, got {spacing_km}")

    if lambda_min_km is None:
        # Only an upper wavelength bound is set — keep wavelengths shorter
        # than lambda_max_km, i.e. high-pass with cutoff at the long edge.
        btype = "high"
        cutoff: float | tuple[float, float] = _cutoff_from_wavelength(
            lambda_max_km, spacing_km
        )
    elif lambda_max_km is None:
        # Only a lower wavelength bound is set — keep wavelengths longer
        # than lambda_min_km, i.e. low-pass with cutoff at the short edge.
        btype = "low"
        cutoff = _cutoff_from_wavelength(lambda_min_km, spacing_km)
    else:
        btype = "bandpass"
        cutoff = (
            _cutoff_from_wavelength(lambda_max_km, spacing_km),
            _cutoff_from_wavelength(lambda_min_km, spacing_km),
        )

    return fir_filter(
        da,
        dim=dim,
        cutoff=cutoff,
        method=method,
        btype=btype,
        num_taps=num_taps,
        attenuation_db=attenuation_db,
    )


__all__ = ["bandpass_wavelength", "median_dx_km"]
