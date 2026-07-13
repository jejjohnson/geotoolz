"""Solar-geometry helpers — Earth–Sun distance, SZA, observation factor.

Thin layer over `georeader.reflectance` that exposes the three solar
quantities the TOA / BOA pipeline needs, plus a small local helper that
turns a user-supplied solar zenith angle into the ``π·d²/cos(θ_z)``
observation-date correction factor (so the user can skip the
``pysolar`` call entirely and pass a metadata-derived SZA instead).

The georeader functions are re-exported here so radiometry operators
import a single canonical location, and so future overrides (Spice-based
SZA, sub-pixel solar geometry, ...) only need to change one file.

References:
    Spencer, J. W. (1971). Fourier series representation of the position
    of the sun. Search 2(5), 172.

    NASA Goddard Space Flight Center. Earth-Sun distance correction.
    https://oceancolor.gsfc.nasa.gov/docs/rsr/f0.txt

    Sentinel-2 L1C Algorithm Theoretical Basis Document — TOA reflectance.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from georeader.reflectance import (
    compute_sza as _compute_sza,
    earth_sun_distance_correction_factor as _earth_sun_distance_correction_factor,
    observation_date_correction_factor as _observation_date_correction_factor,
)


def earth_sun_distance_correction_factor(acquisition_date: datetime) -> float:
    r"""Earth–Sun distance ``d`` in astronomical units for a date.

    .. math::

        d \;=\; 1 - 0.01673 \cdot \cos\!\bigl(0.0172 \cdot (t - 4)\bigr)

    where ``t`` is the day of year (1–366), ``0.0172 ≈ 2π/365.256`` is
    Earth's mean angular velocity in rad/day, ``0.01673`` is the
    orbital eccentricity, and the ``-4`` offset puts perihelion at
    ~Jan 3-4. ``d`` ranges from ~0.983 AU (perihelion) to ~1.017 AU
    (aphelion); irradiance at the sensor scales as ``1/d²`` (inverse
    square law).

    Args:
        acquisition_date: UTC date of acquisition. Only the day of year
            is used.

    Returns:
        Earth–Sun distance in AU (~0.983 to ~1.017).
    """
    return float(_earth_sun_distance_correction_factor(acquisition_date))


def compute_sza(
    center_coords: tuple[float, float],
    acquisition_date: datetime,
    *,
    crs_coords: str | None = None,
) -> float:
    """Solar zenith angle (degrees) at a location and UTC time.

    Thin wrapper over :func:`georeader.reflectance.compute_sza`, which
    uses ``pysolar`` under the hood. The angle is the complement of the
    solar altitude — ``SZA = 90° - altitude`` — and is the quantity that
    enters the cosine term in the TOA-reflectance equation.

    Args:
        center_coords: ``(x, y)`` location. If ``crs_coords`` is
            ``None`` this is interpreted as ``(lon, lat)`` in
            EPSG:4326.
        acquisition_date: UTC datetime of acquisition.
        crs_coords: CRS of ``center_coords``. ``None`` → EPSG:4326.

    Returns:
        Solar zenith angle in degrees, in ``[0, 180]``.
    """
    return float(_compute_sza(center_coords, acquisition_date, crs_coords=crs_coords))


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

    Two paths:

    - If the user already knows the solar zenith angle (e.g. from the
      product metadata), pass ``sza_deg`` and skip the ``pysolar``
      computation entirely. ``d`` still comes from the acquisition
      date.
    - Otherwise pass ``center_coords`` and let georeader compute the
      angle from ``pysolar``.
    - If neither is provided, returns ``None`` so the caller can defer
      to georeader's own auto-resolution (e.g.
      ``radiance_to_reflectance`` derives ``center_coords`` from the
      GeoTensor transform when called without one).

    Args:
        acquisition_date: UTC datetime; only the day of year matters for
            ``d``.
        sza_deg: Solar zenith angle in degrees. If provided, takes
            precedence over ``center_coords``.
        center_coords: ``(lon, lat)`` for ``pysolar`` SZA.
        crs_coords: CRS of ``center_coords``. ``None`` → EPSG:4326.

    Returns:
        The π·d²/cos(θ_z) factor as a float, or ``None`` if neither
        ``sza_deg`` nor ``center_coords`` was given.
    """
    if sza_deg is not None:
        d = earth_sun_distance_correction_factor(acquisition_date)
        return float(np.pi * (d**2) / np.cos(np.deg2rad(sza_deg)))
    if center_coords is not None:
        return float(
            _observation_date_correction_factor(
                center_coords, acquisition_date, crs_coords=crs_coords
            )
        )
    return None
