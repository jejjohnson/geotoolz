"""Geometry-dispatched finite-difference operators.

Public entrypoints route to the geometry-specific implementations in
:mod:`xr_toolz.calc._src.cartesian`,
:mod:`xr_toolz.calc._src.rectilinear`, and
:mod:`xr_toolz.calc._src.spherical`. Vector-calculus operators
(``divergence``, ``curl``, ``laplacian``) are defined in terms of the
per-axis :func:`partial`, with curvature corrections added for the
spherical case so the result matches the equivalent ``metpy.calc``
operators on lon/lat fields.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr

from xr_toolz.calc._src import cartesian, rectilinear, spherical
from xr_toolz.calc._src.constants import EARTH_RADIUS


Geometry = Literal["cartesian", "rectilinear", "spherical"]


def partial(
    da: xr.DataArray,
    dim: str,
    *,
    geometry: Geometry = "cartesian",
    accuracy: int = 1,
    method: str = "central",
    **geom_kw: Any,
) -> xr.DataArray:
    """Partial derivative ``∂da/∂<dim>`` under the given geometry.

    Args:
        da: Input field.
        dim: Dimension along which to differentiate.
        geometry: One of ``"cartesian"`` (uniform grid), ``"rectilinear"``
            (non-uniform 1-D coords), ``"spherical"`` (lon/lat in degrees).
        accuracy: Accuracy order forwarded to :mod:`finitediffx`.
        method: ``"central"`` (default), ``"forward"``, or ``"backward"``.
        **geom_kw: Geometry-specific keyword arguments (e.g. ``radius``
            and coord names for spherical, ``uniform_rtol`` for cartesian).

    Returns:
        DataArray with the same dims/coords as ``da``.
    """
    if geometry == "cartesian":
        return cartesian.cartesian_partial(
            da, dim, accuracy=accuracy, method=method, **geom_kw
        )
    if geometry == "rectilinear":
        return rectilinear.rectilinear_partial(
            da, dim, accuracy=accuracy, method=method, **geom_kw
        )
    if geometry == "spherical":
        return spherical.spherical_partial(
            da, dim, accuracy=accuracy, method=method, **geom_kw
        )
    raise ValueError(
        f"Unknown geometry {geometry!r}; expected one of "
        "'cartesian', 'rectilinear', 'spherical'."
    )


def gradient(
    da: xr.DataArray,
    *,
    dims: tuple[str, ...] | None = None,
    geometry: Geometry = "cartesian",
    accuracy: int | tuple[int, ...] = 1,
    method: str = "central",
    **geom_kw: Any,
) -> xr.Dataset:
    """Gradient ``∇da`` under the given geometry.

    Args:
        da: Input scalar field.
        dims: Dimensions to differentiate against (defaults to ``da.dims``).
        geometry: ``"cartesian"`` | ``"rectilinear"`` | ``"spherical"``.
        accuracy: Scalar or per-dim tuple.
        method: Forwarded to :mod:`finitediffx`.
        **geom_kw: Geometry-specific keyword arguments.

    Returns:
        Dataset with one DataArray per dim in ``dims``.
    """
    if geometry == "cartesian":
        return cartesian.cartesian_gradient(
            da, dims=dims, accuracy=accuracy, method=method, **geom_kw
        )
    if geometry == "rectilinear":
        return rectilinear.rectilinear_gradient(
            da, dims=dims, accuracy=accuracy, method=method, **geom_kw
        )
    if geometry == "spherical":
        return spherical.spherical_gradient(
            da, dims=dims, accuracy=accuracy, method=method, **geom_kw
        )
    raise ValueError(
        f"Unknown geometry {geometry!r}; expected one of "
        "'cartesian', 'rectilinear', 'spherical'."
    )


def _broadcast_lat_factor(
    factor_1d: np.ndarray, *, ref: xr.DataArray, lat: str
) -> np.ndarray:
    """Reshape a 1-D latitude-dependent factor to broadcast on ``ref``."""
    shape = [1] * ref.ndim
    shape[ref.get_axis_num(lat)] = factor_1d.size
    return factor_1d.reshape(shape)


def divergence(
    ds: xr.Dataset,
    components: tuple[str, str] | tuple[str, str, str],
    *,
    dims: tuple[str, ...],
    geometry: Geometry = "cartesian",
    accuracy: int = 1,
    method: str = "central",
    **geom_kw: Any,
) -> xr.DataArray:
    """Divergence ``∇·F`` of a vector field stored as Dataset variables.

    Args:
        ds: Dataset containing the vector components.
        components: Variable names for ``(F_x, F_y, …)``. Must match the
            length of ``dims``.
        dims: Spatial dimensions that pair with each component (e.g.
            ``("x", "y")`` for cartesian, ``("lon", "lat")`` for spherical).
        geometry: ``"cartesian"`` | ``"rectilinear"`` | ``"spherical"``.
        accuracy: Stencil accuracy order.
        method: ``"central"`` | ``"forward"`` | ``"backward"``.
        **geom_kw: Geometry-specific kwargs (``radius``, ``lon``, ``lat``,
            ``uniform_rtol``).

    Returns:
        Scalar DataArray with the same dims/coords as the components.

    Notes:
        For spherical geometry the curvature correction
        ``-(v tan φ) / R`` is added so the result matches
        :func:`metpy.calc.divergence` on lon/lat fields.
    """
    if len(components) != len(dims):
        raise ValueError(
            f"components ({len(components)}) and dims ({len(dims)}) "
            "must have the same length."
        )
    total: xr.DataArray | None = None
    for comp_name, dim in zip(components, dims, strict=True):
        d = partial(
            ds[comp_name],
            dim,
            geometry=geometry,
            accuracy=accuracy,
            method=method,
            **geom_kw,
        )
        total = d if total is None else total + d
    assert total is not None  # len(components) >= 2 enforced by the API

    if geometry == "spherical":
        if len(components) != 2:
            raise ValueError(
                "spherical divergence is defined for 2-D (lon, lat) "
                f"fields; got {len(components)} components."
            )
        lat = geom_kw.get("lat", "lat")
        radius = geom_kw.get("radius", EARTH_RADIUS)
        v = ds[components[1]]
        phi = np.deg2rad(np.asarray(v[lat].values))
        tan_phi = _broadcast_lat_factor(np.tan(phi), ref=v, lat=lat)
        curvature = -(v.values * tan_phi) / radius
        total = total + xr.DataArray(
            curvature,
            dims=v.dims,
            coords={k: v.coords[k] for k in v.coords},
        )

    return total.rename(None)


def curl(
    ds: xr.Dataset,
    components: tuple[str, str],
    *,
    dims: tuple[str, str],
    geometry: Geometry = "cartesian",
    accuracy: int = 1,
    method: str = "central",
    **geom_kw: Any,
) -> xr.DataArray:
    """2-D scalar curl ``∂v/∂x − ∂u/∂y`` (vertical component of ``∇×F``).

    Args:
        ds: Dataset with the two horizontal components.
        components: ``(u_name, v_name)`` — eastward then northward.
        dims: ``(x_dim, y_dim)`` paired with the components.
        geometry: ``"cartesian"`` | ``"rectilinear"`` | ``"spherical"``.
        accuracy, method: Forwarded to :mod:`finitediffx`.
        **geom_kw: Forwarded to :func:`partial`.

    Returns:
        Scalar DataArray with the same dims/coords as the components.

    Notes:
        For spherical geometry the curvature correction
        ``+(u tan φ) / R`` is added so the result matches
        :func:`metpy.calc.vorticity` on lon/lat fields.
    """
    if len(components) != 2 or len(dims) != 2:
        raise ValueError("2-D curl needs exactly two components and two dims.")
    u_name, v_name = components
    x_dim, y_dim = dims
    dvdx = partial(
        ds[v_name],
        x_dim,
        geometry=geometry,
        accuracy=accuracy,
        method=method,
        **geom_kw,
    )
    dudy = partial(
        ds[u_name],
        y_dim,
        geometry=geometry,
        accuracy=accuracy,
        method=method,
        **geom_kw,
    )
    out = dvdx - dudy

    if geometry == "spherical":
        lat = geom_kw.get("lat", "lat")
        radius = geom_kw.get("radius", EARTH_RADIUS)
        u = ds[u_name]
        phi = np.deg2rad(np.asarray(u[lat].values))
        tan_phi = _broadcast_lat_factor(np.tan(phi), ref=u, lat=lat)
        curvature = (u.values * tan_phi) / radius
        out = out + xr.DataArray(
            curvature,
            dims=u.dims,
            coords={k: u.coords[k] for k in u.coords},
        )

    return out.rename(None)


def laplacian(
    da: xr.DataArray,
    *,
    dims: tuple[str, ...] | None = None,
    geometry: Geometry = "cartesian",
    accuracy: int = 1,
    method: str = "central",
    **geom_kw: Any,
) -> xr.DataArray:
    """Laplacian ``Δf = ∇·∇f``.

    Implemented as gradient followed by divergence so the spherical
    curvature correction is inherited automatically.
    """
    if dims is None:
        if geometry == "spherical":
            target_dims: tuple[str, ...] = (
                geom_kw.get("lon", "lon"),
                geom_kw.get("lat", "lat"),
            )
        else:
            target_dims = tuple(da.dims)
    else:
        target_dims = tuple(dims)

    grad = gradient(
        da,
        dims=target_dims,
        geometry=geometry,
        accuracy=accuracy,
        method=method,
        **geom_kw,
    )
    component_names = tuple(grad.data_vars)
    return divergence(
        grad,
        components=component_names,  # type: ignore[arg-type]
        dims=target_dims,
        geometry=geometry,
        accuracy=accuracy,
        method=method,
        **geom_kw,
    )
