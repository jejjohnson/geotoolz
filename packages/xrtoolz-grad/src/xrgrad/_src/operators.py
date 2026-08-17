"""Geometry-dispatched finite-difference operators.

Public entrypoints route to the geometry-specific implementations in
:mod:`xrgrad._src.cartesian`,
:mod:`xrgrad._src.rectilinear`, and
:mod:`xrgrad._src.spherical`. Vector-calculus operators
(``divergence``, ``curl``, ``laplacian``) are defined in terms of the
per-axis :func:`partial`, with curvature corrections added for the
spherical case so the result matches the equivalent ``metpy.calc``
operators on lon/lat fields.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Literal, cast

import numpy as np
import xarray as xr
from jaxtyping import Float

from xrgrad._src import cartesian, rectilinear, spherical
from xrgrad._src.constants import EARTH_RADIUS


Geometry = Literal["cartesian", "rectilinear", "spherical"]

# Which ``**geom_kw`` keys each geometry backend understands. Used to route
# kwargs when ``divergence`` runs a different geometry per dimension: a
# spherical-only key like ``radius`` must not reach the cartesian backend.
_GEOM_KWARGS: dict[str, frozenset[str]] = {
    "cartesian": frozenset({"uniform_rtol"}),
    "rectilinear": frozenset({"uniform_rtol"}),
    "spherical": frozenset({"lon", "lat", "radius", "uniform_rtol"}),
}

Periodic = str | Sequence[str] | None


def _periodic_set(
    periodic: Periodic, dims: tuple[Hashable, ...]
) -> frozenset[Hashable]:
    """Normalise the ``periodic`` argument to a validated set of dim names.

    Raises:
        ValueError: if a named dimension is not among ``dims``.
    """
    if periodic is None:
        return frozenset()
    names: tuple[str, ...] = (
        (periodic,) if isinstance(periodic, str) else tuple(periodic)
    )
    unknown = [n for n in names if n not in dims]
    if unknown:
        raise ValueError(
            f"periodic={periodic!r} names {unknown!r}, which are not among "
            f"the differentiated dims {tuple(dims)!r}."
        )
    return frozenset(names)


def _resolve_geometries(
    geometry: Geometry | Sequence[Geometry], dims: tuple[Hashable, ...]
) -> tuple[Geometry, ...]:
    """Broadcast a scalar geometry over ``dims``, or validate a per-dim tuple.

    Raises:
        ValueError: for an unknown geometry name, or a sequence whose
            length does not match ``dims``.
    """
    # ``str`` is itself a ``Sequence[str]``, so the isinstance check is what
    # separates the scalar form from the per-dim one; the cast records that
    # the else-branch really does yield whole geometry names, not characters.
    per_dim: tuple[Geometry, ...]
    if isinstance(geometry, str):
        per_dim = (cast("Geometry", geometry),) * len(dims)
    else:
        per_dim = cast("tuple[Geometry, ...]", tuple(geometry))
        if len(per_dim) != len(dims):
            raise ValueError(
                f"geometry sequence length ({len(per_dim)}) does not match "
                f"number of dims ({len(dims)})."
            )
    unknown = [g for g in per_dim if g not in _GEOM_KWARGS]
    if unknown:
        raise ValueError(
            f"Unknown geometry {unknown[0]!r}; expected one of {tuple(_GEOM_KWARGS)!r}."
        )
    return per_dim


def _geom_kw_for(geometry: Geometry, geom_kw: dict[str, Any]) -> dict[str, Any]:
    """Select the ``geom_kw`` entries that ``geometry``'s backend accepts."""
    allowed = _GEOM_KWARGS[geometry]
    return {k: v for k, v in geom_kw.items() if k in allowed}


def _validate_geom_kw(
    geometries: tuple[Geometry, ...], geom_kw: dict[str, Any]
) -> None:
    """Reject kwargs no geometry in play understands.

    Filtering per geometry would otherwise swallow a typo silently.

    Raises:
        ValueError: if a keyword is not accepted by any active geometry.
    """
    accepted = frozenset().union(*(_GEOM_KWARGS[g] for g in geometries))
    unknown = sorted(set(geom_kw) - accepted)
    if unknown:
        raise ValueError(
            f"Unexpected keyword(s) {unknown!r} for geometry "
            f"{tuple(dict.fromkeys(geometries))!r}; accepted: "
            f"{sorted(accepted)!r}."
        )


def partial(
    da: xr.DataArray,
    dim: str,
    *,
    order: int = 1,
    geometry: Geometry = "cartesian",
    accuracy: int = 1,
    method: str = "central",
    periodic: bool = False,
    **geom_kw: Any,
) -> xr.DataArray:
    """Partial derivative ``∂ᵒʳᵈᵉʳda/∂<dim>ᵒʳᵈᵉʳ`` under the given geometry.

    Args:
        da: Input field.
        dim: Dimension along which to differentiate.
        order: Derivative order. ``1`` (default) is the first derivative.
            ``order > 1`` is supported for ``geometry="cartesian"`` (and
            for ``"rectilinear"`` when the coordinate turns out to be
            uniformly spaced, which delegates to the cartesian backend);
            see Raises for the other geometries.
        geometry: One of ``"cartesian"`` (uniform grid), ``"rectilinear"``
            (non-uniform 1-D coords), ``"spherical"`` (lon/lat in degrees).
        accuracy: Accuracy order forwarded to :mod:`finitediffx`.
        method: ``"central"`` (default), ``"forward"``, or ``"backward"``.
        periodic: Treat ``dim`` as wrapping, so the first and last points
            are differentiated against each other rather than falling
            back to one-sided stencils. For ``geometry="spherical"`` this
            is valid on the longitude axis of a full 360° grid only.
        **geom_kw: Geometry-specific keyword arguments (e.g. ``radius``
            and coord names for spherical, ``uniform_rtol`` for cartesian).

    Returns:
        DataArray with the same dims/coords as ``da``.

    Raises:
        ValueError: for an unknown ``geometry``, a non-positive ``order``,
            or ``periodic`` on a spherical grid that is not global.
        NotImplementedError: for ``order > 1`` under
            ``geometry="spherical"``, or under ``geometry="rectilinear"``
            with a genuinely non-uniform coordinate; likewise for
            ``periodic`` on a latitude axis or a non-uniform coordinate.

    Example:
        ``d(x²)/dx = 2x`` — exact on interior points with the central
        stencil; the boundary points fall back to one-sided differences
        of the requested accuracy:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> x = np.linspace(0.0, 3.0, 4)
        >>> da = xr.DataArray(x**2, dims="x", coords={"x": x})
        >>> partial(da, "x").values
        array([1., 2., 4., 5.])

        ```

        The second derivative ``d²(x²)/dx² = 2`` comes from a single
        order-2 stencil rather than two composed passes:

        ```pycon
        >>> partial(da, "x", order=2).values
        array([2., 2., 2., 2.])

        ```
    """
    kw: dict[str, Any] = dict(
        order=order,
        accuracy=accuracy,
        method=method,
        periodic=periodic,
        **geom_kw,
    )
    if geometry == "cartesian":
        return cartesian.cartesian_partial(da, dim, **kw)
    if geometry == "rectilinear":
        return rectilinear.rectilinear_partial(da, dim, **kw)
    if geometry == "spherical":
        return spherical.spherical_partial(da, dim, **kw)
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
    periodic: Periodic = None,
    **geom_kw: Any,
) -> xr.Dataset:
    """Gradient ``∇da`` under the given geometry.

    Args:
        da: Input scalar field.
        dims: Dimensions to differentiate against (defaults to ``da.dims``).
        geometry: ``"cartesian"`` | ``"rectilinear"`` | ``"spherical"``.
        accuracy: Scalar or per-dim tuple.
        method: Forwarded to :mod:`finitediffx`.
        periodic: Dimension name, or sequence of names, to treat as
            wrapping. Must be among the differentiated dims.
        **geom_kw: Geometry-specific keyword arguments.

    Returns:
        Dataset with one DataArray per dim in ``dims``, named
        ``d<name>_d<axis>`` (``f`` when the input is unnamed).

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> x = np.linspace(0.0, 3.0, 4)
        >>> X, Y = np.meshgrid(x, x, indexing="xy")
        >>> f = xr.DataArray(
        ...     X**2 + Y**2, dims=("y", "x"), coords={"y": x, "x": x},
        ... )
        >>> grads = gradient(f, dims=("x", "y"))
        >>> sorted(grads.data_vars)
        ['df_dx', 'df_dy']

        ```
    """
    if geometry == "spherical":
        target = (
            (geom_kw.get("lon", "lon"), geom_kw.get("lat", "lat"))
            if dims is None
            else tuple(dims)
        )
    else:
        target = tuple(da.dims) if dims is None else tuple(dims)
    wrap = _periodic_set(periodic, target)

    if geometry == "cartesian":
        return cartesian.cartesian_gradient(
            da, dims=dims, accuracy=accuracy, method=method, periodic=wrap, **geom_kw
        )
    if geometry == "rectilinear":
        return rectilinear.rectilinear_gradient(
            da, dims=dims, accuracy=accuracy, method=method, periodic=wrap, **geom_kw
        )
    if geometry == "spherical":
        return spherical.spherical_gradient(
            da, dims=dims, accuracy=accuracy, method=method, periodic=wrap, **geom_kw
        )
    raise ValueError(
        f"Unknown geometry {geometry!r}; expected one of "
        "'cartesian', 'rectilinear', 'spherical'."
    )


def _broadcast_lat_factor(
    factor_1d: Float[np.ndarray, "lat"], *, ref: xr.DataArray, lat: str
) -> Float[np.ndarray, "..."]:
    """Reshape a 1-D latitude-dependent factor to broadcast on ``ref``."""
    shape = [1] * ref.ndim
    shape[ref.get_axis_num(lat)] = factor_1d.size
    return factor_1d.reshape(shape)


def divergence(
    ds: xr.Dataset,
    components: tuple[str, str] | tuple[str, str, str],
    *,
    dims: tuple[str, ...],
    geometry: Geometry | Sequence[Geometry] = "cartesian",
    accuracy: int | tuple[int, ...] = 1,
    method: str = "central",
    periodic: Periodic = None,
    **geom_kw: Any,
) -> xr.DataArray:
    r"""Divergence ``∇·F`` of a vector field stored as Dataset variables.

    Args:
        ds: Dataset containing the vector components.
        components: Variable names for ``(F_x, F_y, …)``. Must match the
            length of ``dims``.
        dims: Spatial dimensions that pair with each component (e.g.
            ``("x", "y")`` for cartesian, ``("lon", "lat")`` for spherical).
        geometry: A single geometry applied to every dim, or a per-dim
            sequence pairing with ``dims``. The per-dim form supports
            mixed grids — the common case being a spherical horizontal
            pair plus a rectilinear vertical axis, e.g.
            ``("spherical", "spherical", "rectilinear")`` with
            ``dims=("lon", "lat", "depth")``.
        accuracy: Scalar (applied to every dim) or per-dim tuple matching
            the length of ``dims``.
        method: ``"central"`` | ``"forward"`` | ``"backward"``.
        periodic: Dimension name, or sequence of names, to treat as
            wrapping. Must be among ``dims``.
        **geom_kw: Geometry-specific kwargs (``radius``, ``lon``, ``lat``,
            ``uniform_rtol``). Each is routed only to the backends that
            accept it, so ``radius`` may accompany a mixed geometry
            without reaching the cartesian or rectilinear axes.

    Returns:
        Scalar DataArray with the same dims/coords as the components.

    Raises:
        ValueError: if ``components`` and ``dims`` lengths differ, if a
            geometry sequence does not match ``dims``, or if exactly one
            axis (or more than two) is marked spherical — the metric
            needs the lon/lat pair together.

    Notes:
        The curvature correction ``-(v tan φ) / R`` is added whenever both
        the longitude and latitude axes are spherical, so the result
        matches :func:`metpy.calc.divergence` on lon/lat fields:

        .. math::

            \nabla \cdot \mathbf{F} =
            \frac{1}{R\cos\varphi} \frac{\partial F_x}{\partial \lambda}
            + \frac{1}{R} \frac{\partial F_y}{\partial \varphi}
            - \frac{F_y \tan\varphi}{R}

    Example:
        The radial field ``F = (x, y)`` has constant divergence 2:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> x = np.linspace(0.0, 3.0, 4)
        >>> X, Y = np.meshgrid(x, x, indexing="xy")
        >>> coords = {"y": x, "x": x}
        >>> ds = xr.Dataset(
        ...     {
        ...         "u": xr.DataArray(X, dims=("y", "x"), coords=coords),
        ...         "v": xr.DataArray(Y, dims=("y", "x"), coords=coords),
        ...     }
        ... )
        >>> np.unique(divergence(ds, ("u", "v"), dims=("x", "y")).values)
        array([2.])

        ```

        A 3-D ocean budget mixes a spherical horizontal pair with a
        rectilinear (stretched) vertical axis in one call:

        ```pycon
        >>> depth = np.array([0.0, 10.0, 30.0, 70.0])
        >>> lon, lat = np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0, 12.0])
        >>> shape = (depth.size, lat.size, lon.size)
        >>> dims3 = ("depth", "lat", "lon")
        >>> coords3 = {"depth": depth, "lat": lat, "lon": lon}
        >>> flow = xr.Dataset(
        ...     {
        ...         name: xr.DataArray(np.zeros(shape), dims=dims3, coords=coords3)
        ...         for name in ("u", "v", "w")
        ...     }
        ... )
        >>> div = divergence(
        ...     flow,
        ...     ("u", "v", "w"),
        ...     dims=("lon", "lat", "depth"),
        ...     geometry=("spherical", "spherical", "rectilinear"),
        ... )
        >>> div.dims
        ('depth', 'lat', 'lon')

        ```
    """
    if len(components) != len(dims):
        raise ValueError(
            f"components ({len(components)}) and dims ({len(dims)}) "
            "must have the same length."
        )
    geometries = _resolve_geometries(geometry, dims)
    _validate_geom_kw(geometries, geom_kw)
    per_dim = cartesian._per_dim_accuracy(accuracy, dims)
    wrap = _periodic_set(periodic, dims)

    lon = geom_kw.get("lon", "lon")
    lat = geom_kw.get("lat", "lat")
    spherical_dims = [
        d for d, g in zip(dims, geometries, strict=True) if g == "spherical"
    ]
    # Validated up front so the geometry-level message wins over whichever
    # per-axis partial would otherwise fail first.
    if spherical_dims and sorted(spherical_dims) != sorted({lon, lat}):
        if all(g == "spherical" for g in geometries):
            raise ValueError(
                "spherical divergence is defined for 2-D (lon, lat) "
                f"fields; got dims={tuple(dims)!r}. For a 3-D field pass "
                "a per-dim geometry sequence marking the vertical axis "
                "'rectilinear' or 'cartesian', e.g. geometry="
                "('spherical', 'spherical', 'rectilinear')."
            )
        raise ValueError(
            f"spherical axes {spherical_dims!r} must be exactly the "
            f"longitude/latitude pair ({lon!r}, {lat!r}): the metric "
            "term -(v tan φ)/R couples the two, so neither can be "
            "spherical on its own. Mark the remaining axes "
            "'rectilinear' or 'cartesian'."
        )

    total: xr.DataArray | None = None
    for comp_name, dim, geom, acc in zip(
        components, dims, geometries, per_dim, strict=True
    ):
        d = partial(
            ds[comp_name],
            dim,
            geometry=geom,
            accuracy=acc,
            method=method,
            periodic=dim in wrap,
            **_geom_kw_for(geom, geom_kw),
        )
        total = d if total is None else total + d
    assert total is not None  # narrows Optional; only empty components trips this

    if spherical_dims:
        radius = geom_kw.get("radius", EARTH_RADIUS)
        v = ds[components[dims.index(lat)]]
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
    accuracy: int | tuple[int, ...] = 1,
    method: str = "central",
    periodic: Periodic = None,
    **geom_kw: Any,
) -> xr.DataArray:
    r"""2-D scalar curl ``∂v/∂x − ∂u/∂y`` (vertical component of ``∇×F``).

    Args:
        ds: Dataset with the two horizontal components.
        components: ``(u_name, v_name)`` — eastward then northward.
        dims: ``(x_dim, y_dim)`` paired with the components.
        geometry: ``"cartesian"`` | ``"rectilinear"`` | ``"spherical"``.
        accuracy: Scalar (applied to both dims) or a 2-tuple pairing with
            ``dims``, forwarded to :mod:`finitediffx`.
        periodic: Dimension name, or sequence of names, to treat as
            wrapping. Must be among ``dims``.
        method: Stencil family (e.g. ``"central"``), forwarded to
            :mod:`finitediffx`.
        **geom_kw: Forwarded to :func:`partial`.

    Returns:
        Scalar DataArray with the same dims/coords as the components.

    Notes:
        For spherical geometry the curvature correction
        ``+(u tan φ) / R`` is added so the result matches
        :func:`metpy.calc.vorticity` on lon/lat fields:

        .. math::

            \zeta =
            \frac{1}{R\cos\varphi} \frac{\partial v}{\partial \lambda}
            - \frac{1}{R} \frac{\partial u}{\partial \varphi}
            + \frac{u \tan\varphi}{R}

    Example:
        Rigid rotation ``F = (-y, x)`` has constant vorticity 2:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> x = np.linspace(0.0, 3.0, 4)
        >>> X, Y = np.meshgrid(x, x, indexing="xy")
        >>> coords = {"y": x, "x": x}
        >>> ds = xr.Dataset(
        ...     {
        ...         "u": xr.DataArray(-Y, dims=("y", "x"), coords=coords),
        ...         "v": xr.DataArray(X, dims=("y", "x"), coords=coords),
        ...     }
        ... )
        >>> np.unique(curl(ds, ("u", "v"), dims=("x", "y")).values)
        array([2.])

        ```
    """
    if len(components) != 2 or len(dims) != 2:
        raise ValueError("2-D curl needs exactly two components and two dims.")
    u_name, v_name = components
    x_dim, y_dim = dims
    acc_x, acc_y = cartesian._per_dim_accuracy(accuracy, dims)
    wrap = _periodic_set(periodic, dims)
    dvdx = partial(
        ds[v_name],
        x_dim,
        geometry=geometry,
        accuracy=acc_x,
        method=method,
        periodic=x_dim in wrap,
        **geom_kw,
    )
    dudy = partial(
        ds[u_name],
        y_dim,
        geometry=geometry,
        accuracy=acc_y,
        method=method,
        periodic=y_dim in wrap,
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
    accuracy: int | tuple[int, ...] = 1,
    method: str = "central",
    periodic: Periodic = None,
    **geom_kw: Any,
) -> xr.DataArray:
    """Laplacian ``Δf = ∇·∇f``.

    For ``geometry="cartesian"`` this sums direct second-derivative
    stencils (``Σ ∂²f/∂dim²``), which is exact for quadratic fields at
    the default ``accuracy=1`` and contaminates only a single boundary
    ring. An axis too short to host a second-derivative stencil falls
    back to the composed form below, so short axes keep working.

    The other geometries compose gradient followed by divergence so the
    spherical curvature correction is inherited automatically. That
    applies to *all* rectilinear grids, including those whose coordinate
    is uniform enough to delegate each partial to the cartesian backend.
    On those paths the second derivative is two stacked first-derivative
    stencils, so pass ``accuracy=2`` (or higher) when you need the
    interior to reproduce polynomial fields exactly.

    Args:
        da: Input scalar field.
        dims: Dimensions to differentiate against. Defaults to ``da.dims``
            (``(lon, lat)`` for spherical geometry).
        geometry: ``"cartesian"`` | ``"rectilinear"`` | ``"spherical"``.
        accuracy: Scalar (applied to every dim) or per-dim tuple matching
            the length of ``dims``.
        method: Forwarded to :mod:`finitediffx`.
        periodic: Dimension name, or sequence of names, to treat as
            wrapping. Must be among the differentiated dims.
        **geom_kw: Geometry-specific keyword arguments.

    Returns:
        Unnamed scalar DataArray with the same dims/coords as ``da``.

    Raises:
        ValueError: if ``dims`` resolves to an empty tuple, or for an
            unknown ``geometry``.

    Example:
        ``Δ(x² + y²) = 4``, exact on interior points:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> x = np.linspace(0.0, 3.0, 4)
        >>> X, Y = np.meshgrid(x, x, indexing="xy")
        >>> f = xr.DataArray(
        ...     X**2 + Y**2, dims=("y", "x"), coords={"y": x, "x": x},
        ... )
        >>> lap = laplacian(f, dims=("x", "y"))
        >>> np.unique(lap.values[1:-1, 1:-1])
        array([4.])

        ```
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

    if not target_dims:
        raise ValueError("laplacian needs at least one dimension to sum over.")

    per_dim = cartesian._per_dim_accuracy(accuracy, target_dims)
    wrap = _periodic_set(periodic, target_dims)

    if geometry == "cartesian":
        total: xr.DataArray | None = None
        for dim, acc in zip(target_dims, per_dim, strict=True):
            wrapped = dim in wrap
            try:
                second = cartesian.cartesian_partial(
                    da,
                    dim,
                    order=2,
                    accuracy=acc,
                    method=method,
                    periodic=wrapped,
                    **geom_kw,
                )
            except ValueError:
                # finitediffx rejects a direct second-derivative stencil when
                # the axis is too short to host one (e.g. two samples). Fall
                # back to the composed two-pass form, which is what this
                # function used before the order=2 fast path existed, so short
                # axes keep working. A ValueError raised for any other reason
                # — missing, non-1-D, or non-uniform coordinate — is raised
                # again by these same calls.
                first = cartesian.cartesian_partial(
                    da, dim, accuracy=acc, method=method, periodic=wrapped, **geom_kw
                )
                second = cartesian.cartesian_partial(
                    first, dim, accuracy=acc, method=method, periodic=wrapped, **geom_kw
                )
            total = second if total is None else total + second
        assert total is not None  # narrowed by the empty-dims guard above
        return total.rename(None)

    grad = gradient(
        da,
        dims=target_dims,
        geometry=geometry,
        accuracy=per_dim,
        method=method,
        periodic=sorted(wrap),  # type: ignore[arg-type]
        **geom_kw,
    )
    component_names = tuple(grad.data_vars)
    return divergence(
        grad,
        components=component_names,  # type: ignore[arg-type]
        dims=target_dims,
        geometry=geometry,
        accuracy=per_dim,
        method=method,
        periodic=sorted(wrap),  # type: ignore[arg-type]
        **geom_kw,
    )
