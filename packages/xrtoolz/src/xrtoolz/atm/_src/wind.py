"""Horizontal wind diagnostics (speed, direction, components).

All three functions use the meteorological convention: direction is the
compass bearing the wind blows **from**, in degrees clockwise from north,
so a westerly (``u > 0, v = 0``) has ``wind_from_direction = 270°``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from xrreader.types import Variable, apply_cf_attrs


_WIND_SPEED = Variable(
    name="wind_speed",
    standard_name="wind_speed",
    long_name="Wind speed",
    units="m s-1",
)
_WIND_FROM = Variable(
    name="wind_from_direction",
    standard_name="wind_from_direction",
    long_name="Wind from direction",
    units="degree",
)
_WIND_TO = Variable(
    name="wind_to_direction",
    standard_name="wind_to_direction",
    long_name="Wind to direction",
    units="degree",
)


def wind_speed(
    ds: xr.Dataset,
    *,
    u: str = "u",
    v: str = "v",
    name: str = "wind_speed",
) -> xr.Dataset:
    """Horizontal wind speed ``|V| = sqrt(u² + v²)``.

    Args:
        ds: Dataset holding the eastward and northward wind components.
        u: Name of the eastward component.
        v: Name of the northward component.
        name: Name of the output variable.

    Returns:
        ``ds`` with ``name`` added (CF ``wind_speed``, same units as the
        components).
    """
    speed = (ds[u] ** 2 + ds[v] ** 2) ** 0.5
    out = ds.copy()
    out[name] = apply_cf_attrs(speed, _WIND_SPEED, overwrite=True)
    return out


def wind_direction(
    ds: xr.Dataset,
    *,
    u: str = "u",
    v: str = "v",
    name: str = "wind_direction",
    convention: Literal["from", "to"] = "from",
) -> xr.Dataset:
    """Wind direction in degrees clockwise from north.

    ``θ_from = (270° − atan2(v, u)·180/π) mod 360°`` is the bearing the wind
    blows *from* (meteorological convention); ``convention="to"`` returns
    ``θ_to = (θ_from + 180°) mod 360°``. Calm air (``u = v = 0``) maps to
    ``270°`` under ``"from"`` (``atan2(0, 0) = 0``).

    Args:
        ds: Dataset holding the wind components.
        u: Name of the eastward component.
        v: Name of the northward component.
        name: Name of the output variable.
        convention: ``"from"`` (CF ``wind_from_direction``) or ``"to"``
            (CF ``wind_to_direction``).

    Returns:
        ``ds`` with ``name`` added, in degrees on ``[0, 360)``.

    Raises:
        ValueError: If ``convention`` is not ``"from"`` or ``"to"``.
    """
    if convention not in ("from", "to"):
        raise ValueError(f"convention must be 'from' or 'to', got {convention!r}")
    theta = np.mod(270.0 - np.degrees(np.arctan2(ds[v], ds[u])), 360.0)
    if convention == "to":
        theta = np.mod(theta + 180.0, 360.0)
    out = ds.copy()
    variable = _WIND_FROM if convention == "from" else _WIND_TO
    out[name] = apply_cf_attrs(theta, variable, overwrite=True)
    return out


def wind_components(
    ds: xr.Dataset,
    *,
    speed: str = "wind_speed",
    direction: str = "wind_direction",
    convention: Literal["from", "to"] = "from",
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Eastward / northward components from speed and direction.

    Inverse of :func:`wind_speed` + :func:`wind_direction`. With the
    meteorological ``"from"`` bearing ``θ``:
    ``u = −|V| sin θ``, ``v = −|V| cos θ``; under ``"to"`` the signs flip.

    Args:
        ds: Dataset holding the speed and direction.
        speed: Name of the wind-speed variable.
        direction: Name of the direction variable (degrees from north).
        convention: Whether ``direction`` is a ``"from"`` or ``"to"`` bearing.
        u: Name of the eastward output component.
        v: Name of the northward output component.

    Returns:
        ``ds`` with ``u`` and ``v`` added (CF ``eastward_wind`` /
        ``northward_wind``).

    Raises:
        ValueError: If ``convention`` is not ``"from"`` or ``"to"``.
    """
    if convention not in ("from", "to"):
        raise ValueError(f"convention must be 'from' or 'to', got {convention!r}")
    theta = np.radians(ds[direction])
    sign = -1.0 if convention == "from" else 1.0
    out = ds.copy()
    out[u] = apply_cf_attrs(sign * ds[speed] * np.sin(theta), "u", overwrite=True)
    out[v] = apply_cf_attrs(sign * ds[speed] * np.cos(theta), "v", overwrite=True)
    return out
