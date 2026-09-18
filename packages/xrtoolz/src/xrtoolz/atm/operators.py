"""Layer-1 ``Operator`` wrappers around :mod:`xrtoolz.atm._src`.

Each operator is a thin, configurable adapter over a pure-function
primitive: ``__init__`` captures the variable / coordinate names, and
``_apply`` forwards the dataset to the primitive. Operators take an
:class:`xarray.Dataset` and return it augmented with the computed
diagnostic. See :mod:`xrtoolz.atm._src.wind` and
:mod:`xrtoolz.atm._src.vertical` for the underlying implementations.
"""

from __future__ import annotations

from typing import Any, Literal

import xarray as xr

from xrcore import Operator
from xrtoolz.atm._src import vertical as _vertical, wind as _wind


# ---------- wind -----------------------------------------------------------


class WindSpeed(Operator):
    """Horizontal wind speed ``sqrt(u² + v²)``.

    Args:
        u: Name of the eastward wind component.
        v: Name of the northward wind component.
        name: Name of the output variable.

    Returns:
        The input dataset with ``name`` added.
    """

    def __init__(self, u: str = "u", v: str = "v", name: str = "wind_speed"):
        self.u = u
        self.v = v
        self.name = name

    def _apply(self, ds):
        return _wind.wind_speed(ds, u=self.u, v=self.v, name=self.name)

    def get_config(self) -> dict[str, Any]:
        return {"u": self.u, "v": self.v, "name": self.name}


class WindDirection(Operator):
    """Wind direction in degrees clockwise from north.

    Meteorological ``"from"`` bearing by default,
    ``θ = (270° − atan2(v, u)) mod 360°``; ``convention="to"`` adds 180°.

    Args:
        u: Name of the eastward wind component.
        v: Name of the northward wind component.
        name: Name of the output variable.
        convention: ``"from"`` or ``"to"``.

    Returns:
        The input dataset with ``name`` added.
    """

    def __init__(
        self,
        u: str = "u",
        v: str = "v",
        name: str = "wind_direction",
        convention: Literal["from", "to"] = "from",
    ):
        self.u = u
        self.v = v
        self.name = name
        self.convention = convention

    def _apply(self, ds):
        return _wind.wind_direction(
            ds, u=self.u, v=self.v, name=self.name, convention=self.convention
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "u": self.u,
            "v": self.v,
            "name": self.name,
            "convention": self.convention,
        }


class WindComponents(Operator):
    """Eastward / northward components from speed and direction.

    Inverse of :class:`WindSpeed` + :class:`WindDirection`:
    ``u = −|V| sin θ``, ``v = −|V| cos θ`` for a ``"from"`` bearing.

    Args:
        speed: Name of the wind-speed variable.
        direction: Name of the direction variable (degrees from north).
        convention: Whether ``direction`` is a ``"from"`` or ``"to"`` bearing.
        u: Name of the eastward output component.
        v: Name of the northward output component.

    Returns:
        The input dataset with ``u`` and ``v`` added.
    """

    def __init__(
        self,
        speed: str = "wind_speed",
        direction: str = "wind_direction",
        convention: Literal["from", "to"] = "from",
        u: str = "u",
        v: str = "v",
    ):
        self.speed = speed
        self.direction = direction
        self.convention = convention
        self.u = u
        self.v = v

    def _apply(self, ds):
        return _wind.wind_components(
            ds,
            speed=self.speed,
            direction=self.direction,
            convention=self.convention,
            u=self.u,
            v=self.v,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "speed": self.speed,
            "direction": self.direction,
            "convention": self.convention,
            "u": self.u,
            "v": self.v,
        }


# ---------- vertical -------------------------------------------------------


class ColumnIntegral(Operator):
    """Integrate a variable along its vertical coordinate.

    ``"trapezoid"`` applies the trapezoid rule on the coordinate values;
    ``"sum"`` is ``Σ f_k Δz_k`` with cell widths from the coordinate. See
    :func:`xrtoolz.atm.column_integral`.

    Args:
        variable: Name of the variable to integrate.
        dim: Vertical dimension.
        coord: Name of the coordinate locating the levels, or ``None`` for
            the index coordinate on ``dim``.
        method: ``"trapezoid"`` or ``"sum"``.
        name: Name of the output variable; defaults to
            ``f"{variable}_column"``.

    Returns:
        The input dataset with ``name`` added (``dim`` reduced away).
    """

    def __init__(
        self,
        variable: str,
        *,
        dim: str = "z",
        coord: str | None = None,
        method: Literal["trapezoid", "sum"] = "trapezoid",
        name: str | None = None,
    ):
        self.variable = variable
        self.dim = dim
        self.coord = coord
        self.method = method
        self.name = name

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        name = self.name if self.name is not None else f"{self.variable}_column"
        out = ds.copy()
        out[name] = _vertical.column_integral(
            ds[self.variable], dim=self.dim, coord=self.coord, method=self.method
        )
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "dim": self.dim,
            "coord": self.coord,
            "method": self.method,
            "name": self.name,
        }


class HypsometricHeight(Operator):
    """Geopotential height of pressure levels (hypsometric equation).

    ``z_{k+1} = z_k + (R_d T̄_v / g) ln(p_k / p_{k+1})`` integrated upward
    from the surface. See :func:`xrtoolz.atm.hypsometric_height`.

    Args:
        pressure: Pressure per level.
        temperature: Air temperature per level [K].
        specific_humidity: Optional specific humidity for ``T_v``.
        surface_pressure: Surface pressure.
        surface_height: Optional terrain height added to every level.
        level: Vertical dimension.
        name: Name of the output variable.

    Returns:
        The input dataset with ``name`` added.
    """

    def __init__(
        self,
        pressure: str = "pressure",
        temperature: str = "temperature",
        specific_humidity: str | None = None,
        surface_pressure: str = "sp",
        surface_height: str | None = None,
        level: str = "level",
        name: str = "height",
    ):
        self.pressure = pressure
        self.temperature = temperature
        self.specific_humidity = specific_humidity
        self.surface_pressure = surface_pressure
        self.surface_height = surface_height
        self.level = level
        self.name = name

    def _apply(self, ds):
        return _vertical.hypsometric_height(
            ds,
            pressure=self.pressure,
            temperature=self.temperature,
            specific_humidity=self.specific_humidity,
            surface_pressure=self.surface_pressure,
            surface_height=self.surface_height,
            level=self.level,
            name=self.name,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "pressure": self.pressure,
            "temperature": self.temperature,
            "specific_humidity": self.specific_humidity,
            "surface_pressure": self.surface_pressure,
            "surface_height": self.surface_height,
            "level": self.level,
            "name": self.name,
        }


class PBLHeightBulkRichardson(Operator):
    """Boundary-layer height from the bulk Richardson number.

    First height where ``Ri_b = g (θ_v − θ_{v,s}) (z − z_s) / (θ_{v,s}
    (u² + v²))`` reaches ``ri_crit``, interpolated between levels; NaN
    when no level crosses. See
    :func:`xrtoolz.atm.pbl_height_bulk_richardson`.

    Args:
        theta_v: Virtual potential temperature per level [K].
        u: Eastward wind per level.
        v: Northward wind per level.
        height: Height above ground per level [m].
        ri_crit: Critical bulk Richardson number.
        level: Vertical dimension.
        name: Name of the output variable.

    Returns:
        The input dataset with ``name`` added (``level`` reduced away).
    """

    def __init__(
        self,
        theta_v: str,
        u: str = "u",
        v: str = "v",
        height: str = "z_agl",
        ri_crit: float = 0.25,
        level: str = "level",
        name: str = "pbl_height",
    ):
        self.theta_v = theta_v
        self.u = u
        self.v = v
        self.height = height
        self.ri_crit = ri_crit
        self.level = level
        self.name = name

    def _apply(self, ds):
        return _vertical.pbl_height_bulk_richardson(
            ds,
            theta_v=self.theta_v,
            u=self.u,
            v=self.v,
            height=self.height,
            ri_crit=self.ri_crit,
            level=self.level,
            name=self.name,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "theta_v": self.theta_v,
            "u": self.u,
            "v": self.v,
            "height": self.height,
            "ri_crit": self.ri_crit,
            "level": self.level,
            "name": self.name,
        }
