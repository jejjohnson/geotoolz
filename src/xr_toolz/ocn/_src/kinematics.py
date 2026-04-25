"""Kinematic and geostrophic quantities for ocean fields.

All operators take an :class:`xr.Dataset` with geographic coordinates
(``lon``, ``lat``) and return a new Dataset with the computed variable.
Spherical-metric finite differencing is delegated to :mod:`xr_toolz.calc`,
which converts lon/lat (degrees) to metric ``∂/∂x`` and ``∂/∂y`` via the
``1/(R cos φ)`` and ``1/R`` factors and applies the curvature corrections
that ``∇·`` and ``∇×`` require on a sphere.

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

import numpy as np
import xarray as xr

from xr_toolz import calc


def coriolis_parameter(
    lat: xr.DataArray | float | np.ndarray,
) -> xr.DataArray | float | np.ndarray:
    """Coriolis parameter ``f = 2 Ω sin(φ)`` in s⁻¹.

    Args:
        lat: Latitude(s) in **degrees**.

    Returns:
        Coriolis parameter with the same shape as ``lat``.
    """
    return 2.0 * calc.OMEGA * np.sin(np.deg2rad(lat))


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
        g: Gravity in m/s². Defaults to :data:`xr_toolz.calc.GRAVITY`.
        f0: Coriolis parameter in s⁻¹. Defaults to the mean over the
            latitude coordinate of ``ds``.

    Returns:
        Dataset with a single variable ``"psi"`` (stream function).
    """
    ssh = ds[variable]
    g_value = calc.GRAVITY if g is None else g
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
) -> xr.Dataset:
    """Geostrophic ``u`` and ``v`` from a height field (SSH).

    Applies the standard linear geostrophic balance::

        u = -(g / f) ∂η/∂y
        v =  (g / f) ∂η/∂x

    with ``∂/∂x`` and ``∂/∂y`` taken on the lon/lat sphere via
    :func:`xr_toolz.calc.partial`.

    Do **not** pass a stream-function field here — applying the same
    formula to ``ψ = (g/f) η`` would double the scaling and give
    unit-inconsistent velocities. :func:`streamfunction` is provided as
    a separate diagnostic.
    """
    ssh = ds[variable]
    f = coriolis_parameter(ssh["lat"])
    deta_dx = calc.partial(ssh, "lon", geometry="spherical")
    deta_dy = calc.partial(ssh, "lat", geometry="spherical")
    u = -(calc.GRAVITY / f) * deta_dy
    v = (calc.GRAVITY / f) * deta_dx
    u.attrs.update(long_name="Zonal Velocity", standard_name="zonal_velocity")
    v.attrs.update(long_name="Meridional Velocity", standard_name="meridional_velocity")
    return xr.Dataset({"u": u, "v": v})


def kinetic_energy(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Kinetic energy ``0.5 (u² + v²)``."""
    ke = 0.5 * (ds[u] ** 2 + ds[v] ** 2)
    ke.attrs.update(long_name="Kinetic Energy", standard_name="kinetic_energy")
    return ke.to_dataset(name="ke")


def relative_vorticity(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Relative vorticity ``ζ = ∂v/∂x - ∂u/∂y`` with spherical curvature."""
    zeta = calc.curl(ds, (u, v), dims=("lon", "lat"), geometry="spherical")
    zeta.attrs.update(
        long_name="Relative Vorticity", standard_name="relative_vorticity"
    )
    return zeta.to_dataset(name="vort_r")


def absolute_vorticity(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Absolute vorticity ``η = ζ + f``."""
    zeta = relative_vorticity(ds, u=u, v=v)["vort_r"]
    eta = zeta + coriolis_parameter(zeta["lat"])
    eta.attrs.update(long_name="Absolute Vorticity", standard_name="absolute_vorticity")
    return eta.to_dataset(name="vort_a")


def divergence(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Horizontal divergence ``∂u/∂x + ∂v/∂y`` with spherical curvature."""
    div = calc.divergence(ds, (u, v), dims=("lon", "lat"), geometry="spherical")
    div.attrs.update(long_name="Divergence", standard_name="divergence")
    return div.to_dataset(name="div")


def enstrophy(
    ds: xr.Dataset,
    variable: str = "vort_r",
) -> xr.Dataset:
    """Enstrophy ``0.5 ζ²``."""
    ens = 0.5 * (ds[variable] ** 2)
    ens.attrs.update(long_name="Enstrophy", standard_name="enstrophy")
    return ens.to_dataset(name="ens")


def shear_strain(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Shear strain ``Sₛ = ∂v/∂x + ∂u/∂y``."""
    dvdx = calc.partial(ds[v], "lon", geometry="spherical")
    dudy = calc.partial(ds[u], "lat", geometry="spherical")
    sh = dvdx + dudy
    sh.attrs.update(long_name="Shear Strain", standard_name="shear_strain")
    return sh.to_dataset(name="shear_strain")


def tensor_strain(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Normal / tensor strain ``Sₙ = ∂u/∂x - ∂v/∂y``."""
    dudx = calc.partial(ds[u], "lon", geometry="spherical")
    dvdy = calc.partial(ds[v], "lat", geometry="spherical")
    st = dudx - dvdy
    st.attrs.update(long_name="Tensor Strain", standard_name="tensor_strain")
    return st.to_dataset(name="tensor_strain")


def strain_magnitude(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Total strain magnitude ``sqrt(Sₙ² + Sₛ²)``."""
    sn = tensor_strain(ds, u=u, v=v)["tensor_strain"]
    ss = shear_strain(ds, u=u, v=v)["shear_strain"]
    total = np.sqrt(sn**2 + ss**2)
    total.attrs.update(long_name="Strain Magnitude", standard_name="strain")
    return total.to_dataset(name="strain")


def okubo_weiss(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Okubo–Weiss parameter ``Sₙ² + Sₛ² − ζ²``.

    Positive in strain-dominated regions, negative in vortical regions.
    """
    sn = tensor_strain(ds, u=u, v=v)["tensor_strain"]
    ss = shear_strain(ds, u=u, v=v)["shear_strain"]
    zeta = relative_vorticity(ds, u=u, v=v)["vort_r"]
    ow = sn**2 + ss**2 - zeta**2
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
) -> xr.Dataset:
    """Ageostrophic velocity ``(u_a, v_a) = (u, v) − u_g(η)``.

    Args:
        ds: Dataset containing the SSH ``variable`` and the total
            velocities ``u``/``v``.
        variable: Name of the SSH (height) variable from which the
            geostrophic velocities are derived.
        u, v: Names of the total horizontal velocity components.

    Returns:
        Dataset with ``u_a`` and ``v_a`` (ageostrophic components).
    """
    geo = geostrophic_velocities(ds, variable=variable)
    u_a = ds[u] - geo["u"]
    v_a = ds[v] - geo["v"]
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
    components: tuple[str, str] = ("u", "v"),
    dims: tuple[str, str] = ("lon", "lat"),
) -> xr.Dataset:
    """Horizontal tracer advection ``−u·∇c`` on the lon/lat sphere.

    Sign convention follows :func:`metpy.calc.advection`: a positive
    value means the tracer is being increased at that point by the flow.

    Args:
        ds: Dataset containing ``scalar`` and the wind/current
            components.
        scalar: Name of the advected scalar field (e.g. ``"sst"``,
            ``"ssh"``).
        components: Two horizontal wind/current component variable names
            paired with ``dims``.
        dims: Two horizontal coordinate names — must be the lon and lat
            dims so the spherical metric applies. Vertical advection
            (``w ∂c/∂z``) belongs in a future ``atm`` thermodynamics
            module that adds a vertical-coord geometry; for now this
            function is strictly horizontal.

    Returns:
        Dataset with a single variable ``f"{scalar}_advection"``.

    Raises:
        ValueError: if ``components`` / ``dims`` are not 2-D, or if
            ``dims`` includes anything outside ``("lon", "lat")``.
    """
    if len(components) != 2 or len(dims) != 2:
        raise ValueError(
            "advection is 2-D (lon/lat); pass exactly two components and "
            f"two dims, got components={components!r} dims={dims!r}."
        )
    invalid = [d for d in dims if d not in ("lon", "lat")]
    if invalid:
        raise ValueError(
            f"advection currently supports only horizontal lon/lat "
            f"dimensions; got invalid dim(s) {invalid!r}. Vertical "
            "advection requires a vertical-coord geometry not yet "
            "wired into xr_toolz.calc."
        )
    flux: xr.DataArray | None = None
    for comp_name, dim in zip(components, dims, strict=True):
        partial_c = calc.partial(ds[scalar], dim, geometry="spherical")
        term = ds[comp_name] * partial_c
        flux = term if flux is None else flux + term
    assert flux is not None
    out = -flux
    out.attrs.update(
        long_name=f"Advection of {scalar}",
        standard_name=f"{scalar}_advection",
    )
    return out.to_dataset(name=f"{scalar}_advection")


def _vector_derivatives(
    ds: xr.Dataset, u: str, v: str
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return ``(∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y)`` on the lon/lat sphere."""
    dudx = calc.partial(ds[u], "lon", geometry="spherical")
    dudy = calc.partial(ds[u], "lat", geometry="spherical")
    dvdx = calc.partial(ds[v], "lon", geometry="spherical")
    dvdy = calc.partial(ds[v], "lat", geometry="spherical")
    return dudx, dudy, dvdx, dvdy


def shear_vorticity(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Vertical shear vorticity component (along-flow).

    From [Majumdar2024]_::

        ζ_s = (v u ∂u/∂x + v² ∂v/∂x − u² ∂u/∂y − u v ∂v/∂y) / (u² + v²)

    The total relative vorticity decomposes as
    ``ζ = shear_vorticity + curvature_vorticity``. The decomposition
    is undefined where the wind vanishes — calm points
    (``u = v = 0``) are returned as ``0`` rather than NaN/Inf.

    Returns:
        Dataset with a single variable ``vort_shear``.
    """
    uu = ds[u]
    vv = ds[v]
    dudx, dudy, dvdx, dvdy = _vector_derivatives(ds, u, v)
    speed_sq = uu**2 + vv**2
    numerator = vv * uu * dudx + vv * vv * dvdx - uu * uu * dudy - uu * vv * dvdy
    safe_speed_sq = speed_sq.where(speed_sq != 0.0, 1.0)
    zeta_s = xr.where(speed_sq != 0.0, numerator / safe_speed_sq, 0.0)
    zeta_s.attrs.update(long_name="Shear Vorticity", standard_name="shear_vorticity")
    return zeta_s.rename(None).to_dataset(name="vort_shear")


def curvature_vorticity(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Vertical curvature vorticity component (cross-flow).

    From [Majumdar2024]_::

        ζ_c = (u² ∂v/∂x − v² ∂u/∂y − v u ∂u/∂x + u v ∂v/∂y) / (u² + v²)

    Calm points (``u = v = 0``) are returned as ``0``, matching the
    convention in :func:`shear_vorticity`.

    Returns:
        Dataset with a single variable ``vort_curv``.
    """
    uu = ds[u]
    vv = ds[v]
    dudx, dudy, dvdx, dvdy = _vector_derivatives(ds, u, v)
    speed_sq = uu**2 + vv**2
    numerator = uu * uu * dvdx - vv * vv * dudy - vv * uu * dudx + uu * vv * dvdy
    safe_speed_sq = speed_sq.where(speed_sq != 0.0, 1.0)
    zeta_c = xr.where(speed_sq != 0.0, numerator / safe_speed_sq, 0.0)
    zeta_c.attrs.update(
        long_name="Curvature Vorticity", standard_name="curvature_vorticity"
    )
    return zeta_c.rename(None).to_dataset(name="vort_curv")


def frontogenesis(
    ds: xr.Dataset,
    scalar: str,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Petterssen 2-D kinematic frontogenesis on the lon/lat sphere.

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

    Returns:
        Dataset with a single variable ``f"{scalar}_frontogenesis"``.
    """
    dcdx = calc.partial(ds[scalar], "lon", geometry="spherical")
    dcdy = calc.partial(ds[scalar], "lat", geometry="spherical")
    mag = np.sqrt(dcdx**2 + dcdy**2)
    sh = shear_strain(ds, u=u, v=v)["shear_strain"]
    st = tensor_strain(ds, u=u, v=v)["tensor_strain"]
    total = strain_magnitude(ds, u=u, v=v)["strain"]
    div = divergence(ds, u=u, v=v)["div"]
    # ψ is the angle of the axis of dilatation; sin β projects the
    # scalar gradient onto that axis.
    psi = 0.5 * np.arctan2(sh, st)
    # ``cos(2β) = 1 − 2 sin²β``; keep the form metpy uses.
    sin_beta = xr.where(
        mag != 0.0,
        (dcdx * np.cos(psi) + dcdy * np.sin(psi)) / mag.where(mag != 0.0, 1.0),
        0.0,
    )
    front = 0.5 * mag * (total * (1.0 - 2.0 * sin_beta**2) - div)
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
    eke = 0.5 * (ds[u_anom] ** 2 + ds[v_anom] ** 2)
    eke.attrs.update(
        long_name="Eddy Kinetic Energy", standard_name="eddy_kinetic_energy"
    )
    return eke.rename(None).to_dataset(name="eke")


def brunt_vaisala_frequency(
    ds: xr.Dataset,
    density: str = "rho",
    depth: str = "depth",
    rho0: float = 1025.0,
    g: float | None = None,
    positive: str = "down",
) -> xr.Dataset:
    """Squared Brunt–Väisälä (buoyancy) frequency.

    Defined as ``N² = −(g/ρ₀) ∂ρ/∂z`` with ``z`` positive **upward**.
    In a stable column ``N² > 0``. We return the same sign convention
    independent of how the input vertical coordinate is oriented;
    set ``positive`` to match the dataset.

    Vertical FD uses the ``rectilinear`` geometry so non-uniform
    pressure / depth grids are supported.

    Args:
        ds: Dataset with a ``density`` variable on a vertical
            coordinate.
        density: In-situ or potential density variable (kg m⁻³).
        depth: Name of the vertical coordinate (metres).
        rho0: Reference density (kg m⁻³). Defaults to 1025.
        g: Gravity (m s⁻²). Defaults to :data:`xr_toolz.calc.GRAVITY`.
        positive: Either ``"down"`` (default — oceanographic depth
            coordinate, increasing into the column) or ``"up"`` (height
            above some reference, decreasing into the column).

    Returns:
        Dataset with a single variable ``"n_squared"``.
    """
    if positive not in ("down", "up"):
        raise ValueError(f"positive={positive!r} must be 'down' or 'up'.")
    g_value = calc.GRAVITY if g is None else g
    drho_dz = calc.partial(ds[density], depth, geometry="rectilinear")
    sign = 1.0 if positive == "down" else -1.0
    n_sq = sign * (g_value / rho0) * drho_dz
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
    dT_dz = calc.partial(ds[temperature], depth, geometry="rectilinear")
    sign = 1.0 if positive == "down" else -1.0
    gamma = sign * dT_dz
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

        MLD = first depth z below z_ref where ρ(z) − ρ(z_ref) > Δρ

    with ``z_ref = 10 m`` and ``Δρ = 0.03 kg/m³`` as the standard
    defaults. The reference density is taken as the value of ``density``
    at the model level closest to ``reference_depth``.

    For columns where the criterion is never met (fully mixed water
    column), the returned value is the deepest depth in the profile.

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
    z = rho[depth]
    rho_ref = rho.sel({depth: reference_depth}, method="nearest")
    excess = rho - rho_ref - threshold
    # First index along ``depth`` (and below ``reference_depth``) where
    # ``excess > 0``. ``argmax`` over a boolean field returns the first
    # ``True`` index; if no level satisfies the criterion we fall back
    # to the deepest level.
    below_ref = (z >= reference_depth).astype(bool)
    crossing = (excess > 0.0) & below_ref
    found_any = crossing.any(dim=depth)
    first_idx = crossing.argmax(dim=depth)
    last_idx = xr.full_like(first_idx, fill_value=z.size - 1)
    pick_idx = xr.where(found_any, first_idx, last_idx)
    mld = z.isel({depth: pick_idx})
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
) -> xr.Dataset:
    """Single-layer barotropic potential vorticity ``(ζ + f) / h``.

    For a shallow-water / single-layer model ``h`` is the layer
    thickness; for atmospheric height-coordinate use it is the
    geopotential height of the layer.

    Args:
        ds: Dataset with ``u``, ``v``, and the layer thickness/height.
        height: Variable name of the layer thickness / height.
        u, v: Velocity component names.

    Returns:
        Dataset with a single variable ``"pv_barotropic"``.
    """
    eta = absolute_vorticity(ds, u=u, v=v)["vort_a"]
    pv = eta / ds[height]
    pv.attrs.update(
        long_name="Barotropic Potential Vorticity",
        standard_name="barotropic_potential_vorticity",
    )
    return pv.rename(None).to_dataset(name="pv_barotropic")
