"""Finite-difference operators on spherical (lon/lat) coordinates.

Given longitude ``λ`` and latitude ``φ`` in **degrees** on a uniform grid
(in degrees, hence uniform in radians too), these operators return the
metric horizontal derivatives:

::

    ∂F/∂x = (1 / (R cos φ)) · ∂F/∂λ
    ∂F/∂y = (1 / R) · ∂F/∂φ

with ``R`` the Earth radius (default :data:`xrgrad.EARTH_RADIUS`).

By convention, ``dim=<lon-name>`` returns ``∂F/∂x`` (metric, m⁻¹) and
``dim=<lat-name>`` returns ``∂F/∂y``. The returned DataArray name is
``f"d{da.name}_dx"`` or ``f"d{da.name}_dy"`` to make the convention
visible at the call site.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import xarray as xr
from jaxtyping import Float

from xrgrad._src import cartesian
from xrgrad._src.constants import EARTH_RADIUS


_RAD_PER_DEG = float(np.pi / 180.0)


def _radian_step(coord: xr.DataArray, *, rtol: float) -> float:
    """Uniform step of a degree-valued lon/lat coord, returned in radians."""
    deg_step = cartesian._uniform_step(coord, rtol=rtol)
    return deg_step * _RAD_PER_DEG


def _broadcast_along(
    values: Float[np.ndarray, "n"], *, ndim: int, axis: int
) -> Float[np.ndarray, "..."]:
    """Reshape a 1-D array so it broadcasts along ``axis`` of an ndim array."""
    shape = [1] * ndim
    shape[axis] = values.size
    return values.reshape(shape)


def _validate_periodic_lon(
    da: xr.DataArray, dim: str, *, lon: str, lat: str, rtol: float
) -> None:
    """Guard the periodic option on a lon/lat grid.

    Raises:
        NotImplementedError: if ``dim`` is the latitude axis, which does
            not wrap (the poles are a different problem entirely).
        ValueError: if the longitude axis does not span a full 360°, so
            wrapping it would join two points that are not neighbours.
    """
    if dim == lat:
        raise NotImplementedError(
            f"periodic=True is not supported for the latitude axis {lat!r}: "
            "latitude does not wrap. Only the longitude axis is periodic on "
            "a lon/lat grid."
        )
    step_deg = cartesian._uniform_step(da[lon], rtol=rtol)
    span_deg = abs(step_deg) * da.sizes[lon]
    if not np.isclose(span_deg, 360.0, rtol=max(rtol, 1e-6), atol=0.0):
        raise ValueError(
            f"periodic longitude requires a full 360° grid; coordinate "
            f"{lon!r} spans {span_deg:g}° ({da.sizes[lon]} points of "
            f"{step_deg:g}°). A regional grid has no wrap-around neighbour."
        )


def spherical_partial(
    da: xr.DataArray,
    dim: str,
    *,
    order: int = 1,
    accuracy: int = 1,
    method: str = "central",
    periodic: bool = False,
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
        order: Derivative order. Only ``1`` is supported — see Raises.
        accuracy: ``finitediffx`` accuracy order.
        method: ``"central"`` | ``"forward"`` | ``"backward"``.
        periodic: Wrap the longitude axis, removing the one-sided seam at
            the dateline. Valid only when ``dim`` is the longitude
            coordinate and the grid spans the full 360° — see Raises.
        lon: Name of the longitude coordinate (degrees east).
        lat: Name of the latitude coordinate (degrees north).
        radius: Earth radius in metres.
        uniform_rtol: Tolerance for the uniform-spacing check on each
            coordinate (in degrees).

    Returns:
        DataArray with the same dims/coords as ``da`` and a name of
        ``f"d{da.name}_dx"`` or ``f"d{da.name}_dy"``.

    Raises:
        ValueError: if ``lon``/``lat`` are missing, if ``dim`` is neither
            of them, or if a coordinate that this derivative actually
            needs is not 1-D. Differentiating along ``lat`` needs only the
            latitude axis, so a scalar ``lon`` coordinate (as left behind
            by ``da.sel(lon=...)``) is accepted.
        NotImplementedError: if ``order > 1``. Repeating the metric
            derivative is not the geometric second derivative — the
            metric factors do not commute with ``∂`` — so use
            :func:`xrgrad.laplacian`, which carries the curvature
            corrections. Also if ``periodic`` is requested for the
            latitude axis, which does not wrap.
    """
    cartesian._validate_order(order)
    if lon not in da.coords:
        raise ValueError(f"Coordinate {lon!r} not present on DataArray.")
    if lat not in da.coords:
        raise ValueError(f"Coordinate {lat!r} not present on DataArray.")
    if order > 1:
        raise NotImplementedError(
            f"order={order} is not supported for geometry='spherical': "
            "repeating the metric derivative 1/(R cos φ) ∂/∂λ is not the "
            "geometric second derivative. Use xrgrad.laplacian, which "
            "includes the curvature corrections."
        )
    if dim not in (lon, lat):
        raise ValueError(
            f"dim={dim!r} must be the lon coord ({lon!r}) or the lat "
            f"coord ({lat!r}) for geometry='spherical'."
        )
    # Only the differentiated axis has to be 1-D here. The other coordinate
    # is constrained further down, and only when its metric factor is
    # actually used — a lat derivative is well defined even when lon has
    # been collapsed to a scalar by e.g. ``da.sel(lon=...)``.
    cartesian._require_1d_coord(da[dim])
    if dim not in da.dims:
        raise ValueError(
            f"Dimension {dim!r} not present on DataArray with dims={da.dims}."
        )

    if periodic:
        _validate_periodic_lon(da, dim, lon=lon, lat=lat, rtol=uniform_rtol)

    axis = da.get_axis_num(dim)
    step_rad = _radian_step(da[dim], rtol=uniform_rtol)
    raw = cartesian._difference_maybe_periodic(
        da.values,
        axis=axis,
        step_size=step_rad,
        accuracy=accuracy,
        method=method,
        periodic=periodic,
    )

    name_suffix: str
    if dim == lon:
        # cos φ has to broadcast along the latitude axis, so the latitude
        # coordinate must itself be 1-D for this branch only.
        cartesian._require_1d_coord(da[lat])
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
    periodic: frozenset[Hashable] = frozenset(),
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
        periodic: Set of coordinate names to wrap (longitude only).
        lon: Longitude coordinate name, forwarded to
            :func:`spherical_partial`.
        lat: Latitude coordinate name, forwarded to
            :func:`spherical_partial`.
        radius: Sphere radius in metres, forwarded to
            :func:`spherical_partial`.
        uniform_rtol: Relative tolerance for the uniform-spacing check,
            forwarded to :func:`spherical_partial`.

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
    per_dim = cartesian._per_dim_accuracy(accuracy, target_dims)

    base = da.name or "f"
    out: dict[str, xr.DataArray] = {}
    for d, acc in zip(target_dims, per_dim, strict=True):
        component = spherical_partial(
            da,
            d,
            accuracy=acc,
            method=method,
            periodic=d in periodic,
            lon=lon,
            lat=lat,
            radius=radius,
            uniform_rtol=uniform_rtol,
        )
        suffix = "dx" if d == lon else "dy"
        out[f"d{base}_{suffix}"] = component
    return xr.Dataset(out)
