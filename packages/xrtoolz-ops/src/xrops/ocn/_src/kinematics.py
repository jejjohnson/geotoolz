"""Kinematic and geostrophic quantities for ocean fields.

All operators take an :class:`xr.Dataset` and return a new Dataset with
the computed variable. The horizontal diagnostics are gridax's and
diagnostix's kernels, on the collocated grid that the coordinates describe.
By default that is a sphere (``dims=("lon", "lat")``,
``geometry="spherical"``), in degrees, with the metric and curvature
terms of each diagnostic, and the Coriolis parameter is derived from
latitude.

Idealised model output on an f/β-plane passes ``dims=("x", "y")`` with
``geometry="cartesian"`` (uniform spacing in metres) or
``"rectilinear"`` (non-uniform), and an explicit Coriolis ``f=`` to the
functions that need one — a scalar ``f₀``, a ``(f₀, β)`` or
``(f₀, β, y₀)`` β-plane tuple, or a DataArray used as-is.

Provided quantities:

- Coriolis parameter, stream function, geostrophic + ageostrophic
  velocities.
- Vorticity (relative, absolute, shear, curvature), divergence,
  enstrophy, kinetic energy, eddy kinetic energy, velocity magnitudes.
- Deformation (shear, stretching, total) and Okubo–Weiss parameter.
- Tracer advection, Petterssen frontogenesis, single-layer barotropic
  potential vorticity.
- Vertical-column diagnostics: Brunt–Väisälä frequency, mixed layer
  depth.

Conventions:

- ``ssh`` is sea-surface height in metres.
- ``u`` and ``v`` are zonal and meridional velocities in m/s.
- ``psi`` (stream function) satisfies ``u = -dpsi/dy``, ``v = dpsi/dx``.
"""

from __future__ import annotations

from typing import cast

import gridax
import numpy as np
import xarray as xr
from diagnostix import columns, kinematics
from geophysix import rotation
from geophysix.seawater import LINEAR_EOS

from xrops import grad
from xrops._src.collocated import Geometry, on_grid
from xrops._src.columns import levels_of, on_columns


type Coriolis = (
    float | tuple[float, float] | tuple[float, float, float] | xr.DataArray | None
)


def coriolis_parameter(
    lat: xr.DataArray | float | np.ndarray,
) -> xr.DataArray | float | np.ndarray:
    """Coriolis parameter ``f = 2 Ω sin(φ)`` in s⁻¹.

    Args:
        lat: Latitude(s) in **degrees**.

    Returns:
        Coriolis parameter with the same shape as ``lat``.
    """
    f = rotation.coriolis(lat)
    if isinstance(f, xr.DataArray):
        return f
    return float(f) if np.ndim(f) == 0 else f


def _coriolis_field(
    da: xr.DataArray,
    dims: tuple[str, str],
    geometry: Geometry,
    f: Coriolis,
) -> xr.DataArray | float:
    """Resolve the Coriolis parameter for ``da``'s grid.

    Args:
        da: Field whose grid ``f`` must broadcast against.
        dims: ``(x_dim, y_dim)`` horizontal dims; ``dims[1]`` is the
            latitude (spherical) or the β-plane ``y`` axis.
        geometry: Horizontal geometry of ``dims``.
        f: ``None`` derives ``2 Ω sin φ`` from the ``dims[1]`` latitude
            coordinate (spherical only). A scalar is a constant f-plane;
            ``(f₀, β)`` is the β-plane ``f₀ + β (y − y₀)`` with ``y₀`` the
            centre of the ``dims[1]`` coordinate, and ``(f₀, β, y₀)``
            pins ``y₀``. A DataArray is used as-is.

    Returns:
        Coriolis parameter (s⁻¹), a float or a DataArray on ``dims[1]``.

    Raises:
        ValueError: ``f`` is ``None`` on a non-spherical grid, or the
            tuple has the wrong length.
    """
    if f is None:
        if geometry != "spherical":
            raise ValueError(
                f"geometry={geometry!r} has no latitude to derive the Coriolis "
                "parameter from; pass f= as a scalar f0, an (f0, beta) "
                "β-plane tuple, or a DataArray."
            )
        return cast(xr.DataArray, coriolis_parameter(da[dims[1]]))
    if isinstance(f, xr.DataArray):
        return f
    if isinstance(f, tuple):
        if len(f) not in (2, 3):
            raise ValueError(
                f"a β-plane f= is (f0, beta) or (f0, beta, y0); got {f!r}."
            )
        f0, beta, *pinned = f
        y = da[dims[1]]
        y0 = float(pinned[0]) if pinned else 0.5 * float(y.min() + y.max())
        return float(f0) + float(beta) * (y - y0)
    return float(f)


def _coriolis_on(
    da: xr.DataArray, dims: tuple[str, str], geometry: Geometry, f: Coriolis
) -> xr.DataArray:
    """The Coriolis parameter on both horizontal dims of ``da``'s grid."""
    f_field = _coriolis_field(da, dims, geometry, f)
    plane = da.isel({dim: 0 for dim in da.dims if dim not in dims}, drop=True)
    # times 1, so that the plane's dims and coordinates come with it
    return xr.ones_like(plane) * f_field


def streamfunction(
    ds: xr.Dataset,
    variable: str = "ssh",
    g: float | None = None,
    f0: float | None = None,
) -> xr.Dataset:
    """Stream function from sea-surface height.

    Uses the linear geostrophic approximation ``ψ = (g / f₀) η``, i.e.
    ``η = (f₀ / g) ψ``.

    Args:
        ds: Dataset containing ``variable`` (SSH) and a ``lat`` coord.
        variable: Name of the SSH variable.
        g: Gravity in m/s². Defaults to :data:`xrops.grad.GRAVITY`.
        f0: Coriolis parameter in s⁻¹. Defaults to the mean over the
            latitude coordinate of ``ds``.

    Returns:
        Dataset with a single variable ``"psi"`` (stream function).
    """
    ssh = ds[variable]
    g_value = grad.GRAVITY if g is None else g
    f0_value = float(coriolis_parameter(ssh["lat"]).mean()) if f0 is None else float(f0)
    psi = (g_value / f0_value) * ssh
    psi.attrs.update(
        long_name="Stream Function",
        standard_name="stream_function",
    )
    return psi.to_dataset(name="psi")


def geostrophic_velocities(
    ds: xr.Dataset,
    variable: str = "ssh",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
    f: Coriolis = None,
) -> xr.Dataset:
    """Geostrophic ``u`` and ``v`` from a height field (SSH).

    Applies the standard linear geostrophic balance::

        u = -(g / f) ∂η/∂y
        v =  (g / f) ∂η/∂x

    with ``∂/∂x`` and ``∂/∂y`` taken by :func:`xrops.grad.partial` under
    ``geometry`` (the lon/lat sphere by default).

    Do **not** pass a stream-function field here — applying the same
    formula to ``ψ = (g/f) η`` would double the scaling and give
    unit-inconsistent velocities. :func:`streamfunction` is provided as
    a separate diagnostic.

    Args:
        ds: Dataset containing the height ``variable``.
        variable: Name of the SSH (height) variable, in metres.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.
        f: Coriolis parameter. ``None`` (default) derives it from the
            latitude ``dims[1]`` (spherical only); otherwise a scalar
            ``f₀``, an ``(f₀, β)`` / ``(f₀, β, y₀)`` β-plane tuple, or a
            DataArray.

    Returns:
        Dataset with ``u`` and ``v`` (m/s).

    Example:
        A Gaussian SSH bump on a β-plane in metres::

            x = y = np.linspace(-5e5, 5e5, 101)
            eta = 0.5 * np.exp(-(x[None, :] ** 2 + y[:, None] ** 2) / 1e10)
            ds = xr.Dataset({"eta": (("y", "x"), eta)}, coords={"x": x, "y": y})
            geostrophic_velocities(
                ds, "eta", dims=("x", "y"), geometry="cartesian", f=(1e-4, 1.6e-11)
            )
    """
    ssh = ds[variable]
    f_field = _coriolis_on(ssh, dims, geometry, f)

    def component(index: int) -> xr.DataArray:
        return on_grid(
            lambda height, coriolis, grid: gridax.geostrophic_velocity(
                height, grid, f=coriolis, gravity=grad.GRAVITY
            )[index],
            ssh,
            f_field,
            dims=dims,
            geometry=geometry,
        )

    u, v = component(0), component(1)
    u.attrs.update(long_name="Zonal Velocity", standard_name="zonal_velocity")
    v.attrs.update(long_name="Meridional Velocity", standard_name="meridional_velocity")
    return xr.Dataset({"u": u, "v": v})


def kinetic_energy(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Kinetic energy ``0.5 (u² + v²)``."""
    ke = (0.5 * (ds[u] ** 2 + ds[v] ** 2)).drop_attrs(deep=False)
    ke.attrs.update(long_name="Kinetic Energy", standard_name="kinetic_energy")
    return ke.to_dataset(name="ke")


def relative_vorticity(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
) -> xr.Dataset:
    """Relative vorticity ``ζ = ∂v/∂x - ∂u/∂y``.

    On the default spherical geometry this is the flux form,
    ``1/(R cos φ) (∂v/∂λ − ∂(u cos φ)/∂φ)`` (``gridax.curl``).

    Args:
        ds: Dataset with the velocity components.
        u, v: Velocity component names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.

    Returns:
        Dataset with a single variable ``"vort_r"`` (s⁻¹).

    Example:
        Solid-body rotation ``u = −y, v = x`` on a Cartesian grid has
        ``ζ = 2`` everywhere::

            x = y = np.linspace(0.0, 1e4, 11)
            X, Y = np.meshgrid(x, y)
            ds = xr.Dataset(
                {"u": (("y", "x"), -Y), "v": (("y", "x"), X)},
                coords={"x": x, "y": y},
            )
            relative_vorticity(ds, dims=("x", "y"), geometry="cartesian")
    """
    zeta = on_grid(gridax.curl, ds[u], ds[v], dims=dims, geometry=geometry)
    zeta.attrs.update(
        long_name="Relative Vorticity", standard_name="relative_vorticity"
    )
    return zeta.to_dataset(name="vort_r")


def absolute_vorticity(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
    f: Coriolis = None,
) -> xr.Dataset:
    """Absolute vorticity ``η = ζ + f``.

    Args:
        ds: Dataset with the velocity components.
        u, v: Velocity component names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.
        f: Coriolis parameter. ``None`` (default) derives it from the
            latitude ``dims[1]`` (spherical only); otherwise a scalar
            ``f₀``, an ``(f₀, β)`` / ``(f₀, β, y₀)`` β-plane tuple, or a
            DataArray.

    Returns:
        Dataset with a single variable ``"vort_a"`` (s⁻¹).
    """
    zeta = relative_vorticity(ds, u=u, v=v, dims=dims, geometry=geometry)["vort_r"]
    eta = zeta + _coriolis_field(zeta, dims, geometry, f)
    eta.attrs.update(long_name="Absolute Vorticity", standard_name="absolute_vorticity")
    return eta.to_dataset(name="vort_a")


def divergence(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
) -> xr.Dataset:
    """Horizontal divergence ``∂u/∂x + ∂v/∂y``.

    On the default spherical geometry this is the flux form,
    ``1/(R cos φ) (∂u/∂λ + ∂(v cos φ)/∂φ)`` (``gridax.divergence``).

    Args:
        ds: Dataset with the velocity components.
        u, v: Velocity component names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.

    Returns:
        Dataset with a single variable ``"div"`` (s⁻¹).
    """
    div = on_grid(gridax.divergence, ds[u], ds[v], dims=dims, geometry=geometry)
    div.attrs.update(long_name="Divergence", standard_name="divergence")
    return div.to_dataset(name="div")


def enstrophy(
    ds: xr.Dataset,
    variable: str = "vort_r",
) -> xr.Dataset:
    """Enstrophy ``0.5 ζ²``."""
    ens = (0.5 * (ds[variable] ** 2)).drop_attrs(deep=False)
    ens.attrs.update(long_name="Enstrophy", standard_name="enstrophy")
    return ens.to_dataset(name="ens")


def shear_strain(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
) -> xr.Dataset:
    """Shear strain ``Sₛ = ∂v/∂x + ∂u/∂y``.

    On the default spherical geometry, with derivatives along arc length,
    ``u tan φ / R`` is added: a rotation of the sphere has no strain.

    Args:
        ds: Dataset with the velocity components.
        u, v: Velocity component names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.

    Returns:
        Dataset with a single variable ``"shear_strain"`` (s⁻¹).
    """
    sh = on_grid(gridax.shear_strain, ds[u], ds[v], dims=dims, geometry=geometry)
    sh.attrs.update(long_name="Shear Strain", standard_name="shear_strain")
    return sh.to_dataset(name="shear_strain")


def tensor_strain(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
) -> xr.Dataset:
    """Normal / tensor strain ``Sₙ = ∂u/∂x - ∂v/∂y``.

    On the default spherical geometry, with derivatives along arc length,
    ``v tan φ / R`` is subtracted: a rotation of the sphere has no strain.

    Args:
        ds: Dataset with the velocity components.
        u, v: Velocity component names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.

    Returns:
        Dataset with a single variable ``"tensor_strain"`` (s⁻¹).
    """
    st = on_grid(gridax.tensor_strain, ds[u], ds[v], dims=dims, geometry=geometry)
    st.attrs.update(long_name="Tensor Strain", standard_name="tensor_strain")
    return st.to_dataset(name="tensor_strain")


def strain_magnitude(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
) -> xr.Dataset:
    """Total strain magnitude ``sqrt(Sₙ² + Sₛ²)``.

    Args:
        ds: Dataset with the velocity components.
        u, v: Velocity component names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.

    Returns:
        Dataset with a single variable ``"strain"`` (s⁻¹).
    """
    squared = on_grid(
        gridax.strain_rate_squared, ds[u], ds[v], dims=dims, geometry=geometry
    )
    total = np.sqrt(squared)
    total.attrs.update(long_name="Strain Magnitude", standard_name="strain")
    return total.to_dataset(name="strain")


def okubo_weiss(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
) -> xr.Dataset:
    """Okubo–Weiss parameter ``Sₙ² + Sₛ² − ζ²``.

    Positive in strain-dominated regions, negative in vortical regions.

    Args:
        ds: Dataset with the velocity components.
        u, v: Velocity component names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.

    Returns:
        Dataset with a single variable ``"ow"`` (s⁻²).
    """
    ow = on_grid(
        gridax.okubo_weiss_parameter, ds[u], ds[v], dims=dims, geometry=geometry
    )
    ow.attrs.update(long_name="Okubo-Weiss Parameter", standard_name="okubo_weiss")
    return ow.to_dataset(name="ow")


def coriolis_normalized(
    ds: xr.Dataset,
    variable: str,
    f0: float | None = None,
) -> xr.Dataset:
    """Normalize a variable by the Coriolis parameter ``f₀``.

    Common for plotting Rossby numbers (``ζ / f``, ``σ / f``).

    Args:
        ds: Dataset.
        variable: Variable to normalize.
        f0: Coriolis parameter. Defaults to the mean over ``ds.lat``.

    Returns:
        Dataset with ``variable`` replaced by ``variable / f0``.
    """
    f0_value = float(coriolis_parameter(ds["lat"]).mean()) if f0 is None else float(f0)
    out = ds.copy()
    out[variable] = out[variable] / f0_value
    return out


def ageostrophic_velocities(
    ds: xr.Dataset,
    variable: str = "ssh",
    u: str = "u",
    v: str = "v",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
    f: Coriolis = None,
) -> xr.Dataset:
    """Ageostrophic velocity ``(u_a, v_a) = (u, v) − u_g(η)``.

    Args:
        ds: Dataset containing the SSH ``variable`` and the total
            velocities ``u``/``v``.
        variable: Name of the SSH (height) variable from which the
            geostrophic velocities are derived.
        u, v: Names of the total horizontal velocity components.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.
        f: Coriolis parameter. ``None`` (default) derives it from the
            latitude ``dims[1]`` (spherical only); otherwise a scalar
            ``f₀``, an ``(f₀, β)`` / ``(f₀, β, y₀)`` β-plane tuple, or a
            DataArray.

    Returns:
        Dataset with ``u_a`` and ``v_a`` (ageostrophic components).
    """
    ssh = ds[variable]
    f_field = _coriolis_on(ssh, dims, geometry, f)

    def component(index: int) -> xr.DataArray:
        return on_grid(
            lambda u_, v_, height, coriolis, grid: kinematics.ageostrophic_velocity(
                u_, v_, height, grid, f=coriolis, gravity=grad.GRAVITY
            )[index],
            ds[u],
            ds[v],
            ssh,
            f_field,
            dims=dims,
            geometry=geometry,
        )

    u_a, v_a = component(0), component(1)
    u_a.attrs.update(
        long_name="Ageostrophic Zonal Velocity",
        standard_name="ageostrophic_zonal_velocity",
    )
    v_a.attrs.update(
        long_name="Ageostrophic Meridional Velocity",
        standard_name="ageostrophic_meridional_velocity",
    )
    return xr.Dataset({"u_a": u_a, "v_a": v_a})


def advection(
    ds: xr.Dataset,
    scalar: str,
    components: tuple[str, ...] = ("u", "v"),
    dims: tuple[str, ...] = ("lon", "lat"),
    *,
    geometry: Geometry = "spherical",
) -> xr.Dataset:
    """Tracer advection ``−u·∇c``, on the lon/lat sphere by default.

    Sign convention follows :func:`metpy.calc.advection`: a positive
    value means the tracer is being increased at that point by the flow.

    Args:
        ds: Dataset containing ``scalar`` and the wind/current
            components.
        scalar: Name of the advected scalar field (e.g. ``"sst"``,
            ``"ssh"``).
        components: Wind/current component variable names paired with
            ``dims`` — ``(u, v)`` for the horizontal form, or
            ``(u, v, w)`` to include vertical advection.
        dims: Two horizontal coordinate names, optionally followed by a
            vertical coordinate name. Under the default spherical
            geometry the horizontal pair must be ``lon``/``lat``. The
            vertical axis is differentiated with
            ``geometry="rectilinear"``, so a stretched depth axis needs no
            special handling. ``w`` must live on the same grid as
            ``scalar`` — staggered vertical grids are not supported.
        geometry: Geometry of the horizontal pair — ``"spherical"``
            (default), ``"cartesian"`` or ``"rectilinear"``.

    Returns:
        Dataset with a single variable ``f"{scalar}_advection"``.

    Raises:
        ValueError: if ``components`` and ``dims`` lengths differ, if
            there are not 2 or 3 of each, if the first two dims are not
            the lon/lat pair under spherical geometry, or if the vertical
            dim repeats one of them.
    """
    if len(components) != len(dims):
        raise ValueError(
            f"components ({len(components)}) and dims ({len(dims)}) must "
            f"have the same length; got components={components!r} "
            f"dims={dims!r}."
        )
    if len(dims) not in (2, 3):
        raise ValueError(
            "advection takes two horizontal dims, optionally plus one "
            f"vertical dim; got dims={dims!r}."
        )
    if geometry == "spherical" and sorted(dims[:2]) != ["lat", "lon"]:
        raise ValueError(
            f"the first two dims must be the horizontal lon/lat pair so the "
            f"spherical metric applies; got {tuple(dims[:2])!r}."
        )
    if len(dims) == 3 and dims[2] in dims[:2]:
        raise ValueError(
            f"the vertical dim must differ from the horizontal pair; got dims={dims!r}."
        )
    missing = [d for d in dims if d not in ds[scalar].dims]
    if missing:
        raise ValueError(
            f"scalar {scalar!r} has dims {tuple(ds[scalar].dims)!r} and does "
            f"not span {missing!r}; every advected dim must be on the tracer."
        )
    # Each component must live on the axis it is paired with. Without this a
    # staggered velocity (w on 'depth_w' while the tracer is on 'depth')
    # broadcasts instead of aligning, silently returning a field carrying
    # both vertical dims rather than the tracer's grid.
    for comp_name, dim in zip(components, dims, strict=True):
        if dim not in ds[comp_name].dims:
            raise ValueError(
                f"component {comp_name!r} has dims {tuple(ds[comp_name].dims)!r} "
                f"and does not span its paired dim {dim!r}. Staggered grids are "
                f"not supported — interpolate {comp_name!r} onto the {scalar!r} "
                "grid first."
            )
    # The pair may arrive as (lat, lon): the grid takes x first.
    named = dict(zip(dims, components, strict=True))
    horizontal = ("lon", "lat") if geometry == "spherical" else (dims[0], dims[1])
    along_x, along_y = (ds[named[dim]] for dim in horizontal)
    out = on_grid(
        kinematics.advection,
        ds[scalar],
        along_x,
        along_y,
        dims=horizontal,
        geometry=geometry,
    )
    if len(dims) == 3:
        # the vertical axis is a plain 1-D coordinate, typically stretched
        vertical = grad.partial(ds[scalar], dims[2], geometry="rectilinear")
        out = out - ds[components[2]] * vertical
    out.attrs.update(
        long_name=f"Advection of {scalar}",
        standard_name=f"{scalar}_advection",
    )
    return out.to_dataset(name=f"{scalar}_advection")


def shear_vorticity(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
) -> xr.Dataset:
    """Vertical shear vorticity component (along-flow).

    From [Majumdar2024]_::

        ζ_s = (v u ∂u/∂x + v² ∂v/∂x − u² ∂u/∂y − u v ∂v/∂y) / (u² + v²)

    with the curvature terms of the velocity gradient on a sphere. The
    relative vorticity decomposes as
    ``ζ = shear_vorticity + curvature_vorticity``, on a sphere to the
    truncation of the flux form. The decomposition is undefined where the
    wind vanishes — calm points (``u = v = 0``) are returned as ``0``
    rather than NaN/Inf.

    Args:
        ds: Dataset with the velocity components.
        u, v: Velocity component names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.

    Returns:
        Dataset with a single variable ``vort_shear``.
    """
    zeta_s = on_grid(
        kinematics.shear_vorticity, ds[u], ds[v], dims=dims, geometry=geometry
    )
    zeta_s.attrs.update(long_name="Shear Vorticity", standard_name="shear_vorticity")
    return zeta_s.rename(None).to_dataset(name="vort_shear")


def curvature_vorticity(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
) -> xr.Dataset:
    """Vertical curvature vorticity component (cross-flow).

    From [Majumdar2024]_::

        ζ_c = (u² ∂v/∂x − v² ∂u/∂y − v u ∂u/∂x + u v ∂v/∂y) / (u² + v²)

    Calm points (``u = v = 0``) are returned as ``0``, matching the
    convention in :func:`shear_vorticity`.

    Args:
        ds: Dataset with the velocity components.
        u, v: Velocity component names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.

    Returns:
        Dataset with a single variable ``vort_curv``.
    """
    zeta_c = on_grid(
        kinematics.curvature_vorticity, ds[u], ds[v], dims=dims, geometry=geometry
    )
    zeta_c.attrs.update(
        long_name="Curvature Vorticity", standard_name="curvature_vorticity"
    )
    return zeta_c.rename(None).to_dataset(name="vort_curv")


def frontogenesis(
    ds: xr.Dataset,
    scalar: str,
    u: str = "u",
    v: str = "v",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
) -> xr.Dataset:
    """Petterssen 2-D kinematic frontogenesis, on the lon/lat sphere by default.

    Implements the form from [Bluestein1993]_ pp. 248–253::

        F = ½ |∇c| [D cos(2β) − δ]

    where ``c`` is the scalar (potential temperature in atmospheric use,
    or any other density-correlated tracer such as SST in oceanography),
    ``D`` is the total deformation, ``β`` is the angle between the axis
    of dilatation and the isentropes, and ``δ`` is the divergence.

    Args:
        ds: Dataset with the scalar and the velocity components.
        scalar: Name of the scalar field (e.g. ``"sst"``, ``"theta"``).
        u, v: Velocity component names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.

    Returns:
        Dataset with a single variable ``f"{scalar}_frontogenesis"``.
    """
    front = on_grid(
        kinematics.frontogenesis,
        ds[scalar],
        ds[u],
        ds[v],
        dims=dims,
        geometry=geometry,
    )
    front.attrs.update(
        long_name=f"Frontogenesis of {scalar}",
        standard_name=f"{scalar}_frontogenesis",
    )
    return front.rename(None).to_dataset(name=f"{scalar}_frontogenesis")


def velocity_magnitude(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
    w: str | None = None,
) -> xr.Dataset:
    """Velocity magnitude ``|U|``.

    2-D ``sqrt(u² + v²)`` if ``w`` is ``None``; otherwise the full
    3-D ``sqrt(u² + v² + w²)``.

    Returns:
        Dataset with a single variable ``"speed"``.
    """
    sq = ds[u] ** 2 + ds[v] ** 2
    if w is not None:
        sq = sq + ds[w] ** 2
    speed = np.sqrt(sq)
    speed.attrs.update(
        long_name="Velocity Magnitude", standard_name="velocity_magnitude"
    )
    return speed.rename(None).to_dataset(name="speed")


def horizontal_velocity_magnitude(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Horizontal velocity magnitude ``sqrt(u² + v²)``.

    Equivalent to :func:`velocity_magnitude` with ``w=None``; provided
    under the oceanspy-style name for callers that prefer the explicit
    horizontal-only form.
    """
    return velocity_magnitude(ds, u=u, v=v)


def eddy_kinetic_energy(
    ds: xr.Dataset,
    u_anom: str = "u_anom",
    v_anom: str = "v_anom",
) -> xr.Dataset:
    """Eddy kinetic energy ``EKE = ½ (u'² + v'²)``.

    The caller supplies anomaly fields (typically a Reynolds
    decomposition: ``u' = u − ⟨u⟩``). Choosing the temporal/spatial mean
    is a science decision left to the caller; we just compute the
    quadratic combination.

    Args:
        ds: Dataset with the velocity anomaly fields.
        u_anom, v_anom: Variable names of the anomalies.

    Returns:
        Dataset with a single variable ``"eke"``.
    """
    eke = (0.5 * (ds[u_anom] ** 2 + ds[v_anom] ** 2)).drop_attrs(deep=False)
    eke.attrs.update(
        long_name="Eddy Kinetic Energy", standard_name="eddy_kinetic_energy"
    )
    return eke.rename(None).to_dataset(name="eke")


def _height(da: xr.DataArray, depth: str, positive: str) -> xr.DataArray:
    """The height of each level, positive upward."""
    levels = levels_of(da, depth)
    return -levels if positive == "down" else levels


def brunt_vaisala_frequency(
    ds: xr.Dataset,
    density: str = "rho",
    depth: str = "depth",
    rho0: float = LINEAR_EOS.reference_density,
    g: float | None = None,
    positive: str = "down",
) -> xr.Dataset:
    """Squared Brunt–Väisälä (buoyancy) frequency.

    Defined as ``N² = −(g/ρ₀) ∂ρ/∂z`` with ``z`` positive **upward**.
    In a stable column ``N² > 0``. We return the same sign convention
    independent of how the input vertical coordinate is oriented;
    set ``positive`` to match the dataset.

    The derivative is exact for a quadratic at an inner level, however
    uneven the levels, and one-sided at the two ends
    (``diagnostix.columns.buoyancy_frequency_squared``).

    Args:
        ds: Dataset with a ``density`` variable on a vertical
            coordinate.
        density: In-situ or potential density variable (kg m⁻³).
        depth: Name of the vertical coordinate (metres).
        rho0: Reference density (kg m⁻³). Defaults to 1025.
        g: Gravity (m s⁻²). Defaults to :data:`xrops.grad.GRAVITY`.
        positive: Either ``"down"`` (default — oceanographic depth
            coordinate, increasing into the column) or ``"up"`` (height
            above some reference, decreasing into the column).

    Returns:
        Dataset with a single variable ``"n_squared"``.
    """
    if positive not in ("down", "up"):
        raise ValueError(f"positive={positive!r} must be 'down' or 'up'.")
    g_value = grad.GRAVITY if g is None else g
    n_sq = on_columns(
        lambda rho, z: columns.buoyancy_frequency_squared(
            rho, z, reference_density=rho0, gravity=g_value
        ),
        ds[density],
        _height(ds[density], depth, positive),
        dim=depth,
        reduces=False,
    )
    n_sq.attrs.update(
        long_name="Squared Brunt-Väisälä Frequency",
        standard_name="square_of_brunt_vaisala_frequency_in_sea_water",
        units="s-2",
    )
    return n_sq.rename(None).to_dataset(name="n_squared")


def lapse_rate(
    ds: xr.Dataset,
    temperature: str = "T",
    depth: str = "depth",
    positive: str = "down",
) -> xr.Dataset:
    """Vertical lapse rate ``Γ = −∂T/∂z`` (z positive upward).

    The sign convention matches atmospheric usage: a column with
    temperature decreasing upward has a positive lapse rate. We flip
    the sign automatically when ``positive="down"`` (oceanographic
    depth convention) so the returned values still follow the
    upward-z definition.

    Args:
        ds: Dataset with a temperature (or potential-temperature)
            variable on a vertical coordinate.
        temperature: Variable name (e.g. ``"T"``, ``"theta"``).
        depth: Vertical coordinate name.
        positive: ``"down"`` (oceanographic depth, default) or ``"up"``
            (height-above-surface).

    Returns:
        Dataset with a single variable ``"lapse_rate"``.
    """
    if positive not in ("down", "up"):
        raise ValueError(f"positive={positive!r} must be 'down' or 'up'.")
    gamma = on_columns(
        columns.lapse_rate,
        ds[temperature],
        _height(ds[temperature], depth, positive),
        dim=depth,
        reduces=False,
    )
    gamma.attrs.update(
        long_name="Lapse Rate",
        standard_name="air_temperature_lapse_rate",
    )
    return gamma.rename(None).to_dataset(name="lapse_rate")


def mixed_layer_depth(
    ds: xr.Dataset,
    density: str = "rho",
    depth: str = "depth",
    reference_depth: float = 10.0,
    threshold: float = 0.03,
) -> xr.Dataset:
    """Mixed-layer depth via the de Boyer Montégut density threshold.

    Implements the criterion of [deBoyerMontegut2004]_::

        MLD = first depth z below z_ref where ρ(z) − ρ(z_ref) = Δρ

    with ``z_ref = 10 m`` and ``Δρ = 0.03 kg/m³`` as the standard
    defaults, and ``ρ`` linear between the levels, at ``z_ref`` too
    (``diagnostix.columns.mixed_layer_depth``).

    A column where the criterion is never met (mixed to its bottom) gives
    the deepest depth with data, and one without data gives NaN. A level
    without data is left out of its column.

    Args:
        ds: Dataset with a ``density`` variable on a vertical coordinate
            named ``depth`` (positive downward, metres).
        density: Density variable name.
        depth: Vertical coordinate name.
        reference_depth: Depth at which ``ρ_ref`` is read (m).
        threshold: Density jump used to mark the base of the mixed layer
            (kg m⁻³).

    Returns:
        Dataset with a single variable ``"mld"`` collapsed over the
        vertical coordinate.
    """
    rho = ds[density]
    if depth not in rho.dims:
        raise ValueError(
            f"Density {density!r} is not defined on the {depth!r} dimension."
        )
    levels = levels_of(rho, depth)
    if levels[0] > levels[-1]:
        # the kernel reads a column downwards
        rho, levels = (da.isel({depth: slice(None, None, -1)}) for da in (rho, levels))
    mld = on_columns(
        lambda rho_, z: columns.mixed_layer_depth(
            rho_,
            z,
            reference_depth=reference_depth,
            threshold=threshold,
            valid=~np.isnan(rho_),
        ),
        rho,
        levels,
        dim=depth,
        reduces=True,
    )
    mld.attrs.update(
        long_name="Mixed Layer Depth",
        standard_name="ocean_mixed_layer_thickness_defined_by_sigma_t",
        units="m",
    )
    return mld.rename(None).to_dataset(name="mld")


def potential_vorticity_barotropic(
    ds: xr.Dataset,
    height: str = "h",
    u: str = "u",
    v: str = "v",
    *,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
    f: Coriolis = None,
) -> xr.Dataset:
    """Single-layer barotropic potential vorticity ``(ζ + f) / h``.

    For a shallow-water / single-layer model ``h`` is the layer
    thickness; for atmospheric height-coordinate use it is the
    geopotential height of the layer.

    Args:
        ds: Dataset with ``u``, ``v``, and the layer thickness/height.
        height: Variable name of the layer thickness / height.
        u, v: Velocity component names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` (uniform metres) or ``"rectilinear"``.
        f: Coriolis parameter. ``None`` (default) derives it from the
            latitude ``dims[1]`` (spherical only); otherwise a scalar
            ``f₀``, an ``(f₀, β)`` / ``(f₀, β, y₀)`` β-plane tuple, or a
            DataArray.

    Returns:
        Dataset with a single variable ``"pv_barotropic"``.
    """
    pv = on_grid(
        gridax.sw_potential_vorticity,
        ds[u],
        ds[v],
        ds[height],
        _coriolis_on(ds[u], dims, geometry, f),
        dims=dims,
        geometry=geometry,
    )
    pv.attrs.update(
        long_name="Barotropic Potential Vorticity",
        standard_name="barotropic_potential_vorticity",
    )
    return pv.rename(None).to_dataset(name="pv_barotropic")


def density_from_ts(
    ds: xr.Dataset,
    *,
    salinity: str = "so",
    temperature: str = "thetao",
    pressure: str | float | xr.DataArray = 0.0,
    lon: str = "lon",
    lat: str = "lat",
    eos: str = "teos10",
) -> xr.Dataset:
    """In-situ seawater density via TEOS-10 (lazy ``gsw`` import).

    Args:
        ds: Dataset with practical salinity ``salinity`` (PSU) and
            potential temperature ``temperature`` (°C).
        salinity, temperature: Variable names.
        pressure: Sea pressure in dbar — either a variable name in
            ``ds``, a scalar (default ``0.0`` for surface density), or
            an :class:`xr.DataArray` that broadcasts against the
            salinity field.
        lon, lat: Coordinate names used by ``gsw.SA_from_SP``. The
            coords may be 1-D; they are broadcast to the salinity
            shape before calling :mod:`gsw`.
        eos: ``"teos10"`` (default, via :mod:`gsw`). Reserved for future
            ``"linear"`` / ``"jmd"`` alternatives.

    Returns:
        Dataset with a single variable ``"rho"`` (kg/m³).

    Raises:
        ImportError: If :mod:`gsw` is not installed (it is an optional
            extra; install via ``pip install xrtoolz-ops[oceanography]``
            or ``pip install gsw``).
    """
    if eos != "teos10":
        raise NotImplementedError(
            f"eos={eos!r} is not implemented; only 'teos10' is supported. "
            f"Open an issue if you need a linear / JMD EOS."
        )
    import importlib

    try:
        gsw = importlib.import_module("gsw")
    except ImportError as exc:
        raise ImportError(
            "density_from_ts requires the optional 'gsw' dependency. "
            "Install with: pip install xrtoolz-ops[oceanography] (or pip install gsw)."
        ) from exc

    sp = ds[salinity]
    pt = ds[temperature]
    if isinstance(pressure, str):
        p_input: float | xr.DataArray = ds[pressure]
    else:
        p_input = pressure
    lon_da = ds[lon]
    lat_da = ds[lat]
    sp_b, lon_b, lat_b = xr.broadcast(sp, lon_da, lat_da)
    if isinstance(p_input, xr.DataArray):
        sp_b, p_b = xr.broadcast(sp_b, p_input)
    else:
        p_b = p_input
    sa = gsw.SA_from_SP(sp_b, p_b, lon_b, lat_b)
    ct = gsw.CT_from_pt(sa, pt)
    rho = gsw.rho(sa, ct, p_b)
    rho_da = xr.DataArray(rho, coords=sp_b.coords, dims=sp_b.dims, name="rho")
    rho_da.attrs.update(
        long_name="In-situ Seawater Density (TEOS-10)",
        standard_name="sea_water_density",
        units="kg m-3",
    )
    return rho_da.to_dataset()
