"""Finite-difference operators on spherical (lon/lat) coordinates.

Given longitude ``λ`` and latitude ``φ`` in **degrees** on a uniform grid
(in degrees, hence uniform in radians too), these operators return the
metric horizontal derivatives:

::

    ∂F/∂x = (1 / (R cos φ)) · ∂F/∂λ
    ∂F/∂y = (1 / R) · ∂F/∂φ

with ``R`` the Earth radius (default :data:`xr_toolz.calc.EARTH_RADIUS`).

By convention, ``dim=<lon-name>`` returns ``∂F/∂x`` (metric, m⁻¹) and
``dim=<lat-name>`` returns ``∂F/∂y``. The returned DataArray name is
``f"d{da.name}_dx"`` or ``f"d{da.name}_dy"`` to make the convention
visible at the call site.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from xr_toolz.calc._src import cartesian
from xr_toolz.calc._src.constants import EARTH_RADIUS


_RAD_PER_DEG = float(np.pi / 180.0)


def _radian_step(coord: xr.DataArray, *, rtol: float) -> float:
    """Uniform step of a degree-valued lon/lat coord, returned in radians."""
    deg_step = cartesian._uniform_step(coord, rtol=rtol)
    return deg_step * _RAD_PER_DEG


def _broadcast_along(values: np.ndarray, *, ndim: int, axis: int) -> np.ndarray:
    """Reshape a 1-D array so it broadcasts along ``axis`` of an ndim array."""
    shape = [1] * ndim
    shape[axis] = values.size
    return values.reshape(shape)


def spherical_partial(
    da: xr.DataArray,
    dim: str,
    *,
    accuracy: int = 1,
    method: str = "central",
    lon: str = "lon",
    lat: str = "lat",
    radius: float = EARTH_RADIUS,
    uniform_rtol: float = 1e-6,
) -> xr.DataArray:
    """Metric partial derivative on a lon/lat grid.

    Args:
        da: Input field with both ``lon`` and ``lat`` coordinates.
        dim: Either ``lon`` (returns ``∂F/∂x``) or ``lat`` (returns
            ``∂F/∂y``).
        accuracy: ``finitediffx`` accuracy order.
        method: ``"central"`` | ``"forward"`` | ``"backward"``.
        lon: Name of the longitude coordinate (degrees east).
        lat: Name of the latitude coordinate (degrees north).
        radius: Earth radius in metres.
        uniform_rtol: Tolerance for the uniform-spacing check on each
            coordinate (in degrees).

    Returns:
        DataArray with the same dims/coords as ``da`` and a name of
        ``f"d{da.name}_dx"`` or ``f"d{da.name}_dy"``.
    """
    if lon not in da.coords:
        raise ValueError(f"Coordinate {lon!r} not present on DataArray.")
    if lat not in da.coords:
        raise ValueError(f"Coordinate {lat!r} not present on DataArray.")
    if dim not in (lon, lat):
        raise ValueError(
            f"dim={dim!r} must be the lon coord ({lon!r}) or the lat "
            f"coord ({lat!r}) for geometry='spherical'."
        )
    if dim not in da.dims:
        raise ValueError(
            f"Dimension {dim!r} not present on DataArray with dims={da.dims}."
        )

    axis = da.get_axis_num(dim)
    step_rad = _radian_step(da[dim], rtol=uniform_rtol)
    raw = cartesian._difference(
        da.values,
        axis=axis,
        step_size=step_rad,
        accuracy=accuracy,
        method=method,
    )

    name_suffix: str
    if dim == lon:
        lat_values_rad = np.deg2rad(np.asarray(da[lat].values))
        cos_phi = _broadcast_along(
            np.cos(lat_values_rad), ndim=da.ndim, axis=da.get_axis_num(lat)
        )
        out = raw / (radius * cos_phi)
        name_suffix = "dx"
    else:  # dim == lat
        out = raw / radius
        name_suffix = "dy"

    base = da.name
    out_name = f"d{base}_{name_suffix}" if base is not None else None

    return xr.DataArray(
        out,
        dims=da.dims,
        coords={k: da.coords[k] for k in da.coords},
        name=out_name,
        attrs=dict(da.attrs),
    )


def spherical_gradient(
    da: xr.DataArray,
    *,
    dims: tuple[str, ...] | None = None,
    accuracy: int | tuple[int, ...] = 1,
    method: str = "central",
    lon: str = "lon",
    lat: str = "lat",
    radius: float = EARTH_RADIUS,
    uniform_rtol: float = 1e-6,
) -> xr.Dataset:
    """Horizontal gradient ``(∂F/∂x, ∂F/∂y)`` on a lon/lat grid.

    Args:
        da: Input scalar field.
        dims: Coordinates to differentiate. Defaults to ``(lon, lat)``;
            must be a subset of those two for ``geometry="spherical"``.
        accuracy: Scalar or per-dim tuple.
        method: Forwarded to :mod:`finitediffx`.
        lon, lat, radius, uniform_rtol: Forwarded to
            :func:`spherical_partial`.

    Returns:
        Dataset with one DataArray per requested dim, named
        ``f"d{da.name or 'f'}_dx"`` and/or ``f"d{da.name or 'f'}_dy"``.
    """
    target_dims = (lon, lat) if dims is None else tuple(dims)
    for d in target_dims:
        if d not in (lon, lat):
            raise ValueError(
                f"dims={target_dims!r} contains {d!r}; expected entries "
                f"from ({lon!r}, {lat!r}) for geometry='spherical'."
            )
    if isinstance(accuracy, int):
        per_dim = (accuracy,) * len(target_dims)
    else:
        per_dim = tuple(accuracy)
        if len(per_dim) != len(target_dims):
            raise ValueError(
                f"accuracy tuple length ({len(per_dim)}) does not match "
                f"number of dims ({len(target_dims)})."
            )

    base = da.name or "f"
    out: dict[str, xr.DataArray] = {}
    for d, acc in zip(target_dims, per_dim, strict=True):
        component = spherical_partial(
            da,
            d,
            accuracy=acc,
            method=method,
            lon=lon,
            lat=lat,
            radius=radius,
            uniform_rtol=uniform_rtol,
        )
        suffix = "dx" if d == lon else "dy"
        out[f"d{base}_{suffix}"] = component
    return xr.Dataset(out)
