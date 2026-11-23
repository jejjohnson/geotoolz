"""Solar geometry of a scene: Earth–Sun distance, zenith angle, observation factor.

The formulas are radiomax's (`radiomax.geometry`): the Astronomical Almanac's
solar position and distance, within 0.0121° and 8.7e-5 AU of SPA over
2000–2050. This module adds what a carrier knows: the date as a
``datetime``, and where its pixels are.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from geoarray import GeoArray
from jaxtyping import Float
from pyproj import CRS, Transformer
from radiomax.geometry import earth_sun_distance, solar_zenith, utc_day_seconds
from scipy.interpolate import RegularGridInterpolator

from geoops._src.geo import affine_of, pixel_xy


#: Spacing of the tie points the solar zenith is computed on, m. Between them
#: its cosine is bilinear, which is off by at most h²/(8R²) = 8e-8.
_TIE_SPACING_M = 5_000.0
_METRES_PER_DEGREE = 111_000.0


def earth_sun_distance_correction_factor(acquisition_date: datetime) -> float:
    """Earth–Sun distance ``d`` in astronomical units at a date and time.

    From ~0.983 AU at perihelion (~3 January) to ~1.017 AU at aphelion
    (~4 July); the irradiance at the Earth scales as ``1/d²``.

    Args:
        acquisition_date: Date and time of acquisition, UTC unless it carries
            a time zone.

    Returns:
        Earth–Sun distance in AU.
    """
    return float(earth_sun_distance(*utc_day_seconds(acquisition_date)))


def compute_sza(
    center_coords: tuple[float, float],
    acquisition_date: datetime,
    *,
    crs_coords: str | None = None,
) -> float:
    """Solar zenith angle (degrees) at a location and time.

    Args:
        center_coords: ``(x, y)`` location. If ``crs_coords`` is
            ``None`` this is interpreted as ``(lon, lat)`` in
            EPSG:4326.
        acquisition_date: Date and time of acquisition, UTC unless it carries
            a time zone.
        crs_coords: CRS of ``center_coords``. ``None`` → EPSG:4326.

    Returns:
        Solar zenith angle in degrees, in ``[0, 180]``, without refraction.
    """
    lon, lat = _to_lonlat(*center_coords, crs_coords)
    return float(solar_zenith(lon, lat, *utc_day_seconds(acquisition_date)))


def observation_date_correction_factor(
    acquisition_date: datetime,
    *,
    sza_deg: float | None = None,
    center_coords: tuple[float, float] | None = None,
    crs_coords: str | None = None,
) -> float | None:
    r"""Compute the ``π·d²/cos(θ_z)`` factor used by the TOA-reflectance equation.

    .. math::

        \text{obfactor} \;=\; \frac{\pi \, d^{2}}{\cos\!\bigl(\theta_{z}\bigr)}

    Args:
        acquisition_date: Date and time of acquisition.
        sza_deg: Solar zenith angle in degrees. If provided, takes
            precedence over ``center_coords``.
        center_coords: ``(lon, lat)`` at which to compute the angle.
        crs_coords: CRS of ``center_coords``. ``None`` → EPSG:4326.

    Returns:
        The π·d²/cos(θ_z) factor as a float, or ``None`` if neither
        ``sza_deg`` nor ``center_coords`` was given.
    """
    if sza_deg is None and center_coords is None:
        return None
    if sza_deg is None:
        assert center_coords is not None
        sza_deg = compute_sza(center_coords, acquisition_date, crs_coords=crs_coords)
    d = earth_sun_distance_correction_factor(acquisition_date)
    return float(np.pi * d**2 / np.cos(np.deg2rad(sza_deg)))


def footprint_zenith(
    gt: GeoArray,
    acquisition_date: datetime,
    *,
    tie_spacing_m: float = _TIE_SPACING_M,
) -> Float[np.ndarray, "h w"]:
    """Solar zenith angle (degrees) at every pixel of a georeferenced carrier.

    Computed on tie points at most 5 km apart (every pixel, when the pixels
    are coarser) and interpolated in its cosine.

    Args:
        gt: The carrier; its transform and CRS say where the pixels are.
        acquisition_date: Date and time of acquisition.
        tie_spacing_m: Spacing of the tie points, m; 0 computes every pixel.

    Returns:
        ``(H, W)`` zenith angles.
    """
    height, width = gt.shape[-2:]
    affine = affine_of(gt)
    unit = _METRES_PER_DEGREE if CRS.from_user_input(gt.crs).is_geographic else 1.0
    rows = _ties(height, unit * float(np.hypot(affine.b, affine.e)), tie_spacing_m)
    cols = _ties(width, unit * float(np.hypot(affine.a, affine.d)), tie_spacing_m)
    x, y = pixel_xy(gt, rows[:, None], cols[None, :])
    lon, lat = _to_lonlat(x, y, gt.crs)
    zenith = solar_zenith(lon, lat, *utc_day_seconds(acquisition_date))
    if rows.size == height and cols.size == width:
        return np.asarray(zenith)
    cosine = np.cos(np.deg2rad(zenith))
    mesh = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    smooth = RegularGridInterpolator((rows, cols), cosine)(tuple(mesh))
    return np.rad2deg(np.arccos(np.clip(smooth, -1.0, 1.0)))


def _ties(size: int, pixel_m: float, spacing_m: float) -> Float[np.ndarray, " n"]:
    """Pixel indices at most ``spacing_m`` apart, the first and last included."""
    step = max(1, int(spacing_m / pixel_m)) if pixel_m > 0 else 1
    count = min(size, max(2, (size - 1) // step + 2)) if step > 1 else size
    return np.linspace(0.0, size - 1.0, count)


def _to_lonlat(x: Any, y: Any, crs: Any) -> tuple[Any, Any]:
    """Longitude and latitude of coordinates in ``crs`` (``None``: already those)."""
    if crs is None or CRS.from_user_input(crs) == CRS.from_epsg(4326):
        return x, y
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)
