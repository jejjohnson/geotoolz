"""Geostrophic and strain-related quantities for ocean fields.

All operators take an :class:`xr.Dataset` with geographic coordinates
(``lon``, ``lat``) and return a new Dataset with the computed variable.
Physical finite-differencing is delegated to :mod:`metpy.calc`, which
handles the lon/lat → metric-distance conversion using pint quantities
under the hood.

Conventions:

- ``ssh`` is sea-surface height in meters.
- ``u`` and ``v`` are zonal and meridional velocities in m/s.
- ``psi`` (stream function) satisfies ``u = -dpsi/dy``, ``v = dpsi/dx``.
"""

from __future__ import annotations

import metpy.calc as mpcalc
import numpy as np
import pint_xarray  # noqa: F401 — registers the .pint accessor on xarray
import xarray as xr
from metpy.constants import earth_gravity as GRAVITY
from pint import Quantity


_DEFAULT_VAR_UNITS = {
    "u": "m s-1",
    "v": "m s-1",
    "ssh": "m",
    "psi": "m^2 / s",
    "vort_r": "1 / s",
    "vort_a": "1 / s",
    "lon": "degrees_east",
    "lat": "degrees_north",
}


def _assume_units(ds: xr.Dataset, variables: tuple[str, ...]) -> xr.Dataset:
    """Assign default ``units`` attrs for ``variables`` if they are missing.

    MetPy requires every variable involved in a kinematic computation
    to carry a ``units`` attribute. We default to the physical unit
    implied by the variable name so users can call the operator on raw
    data without having to pre-annotate.
    """
    ds = ds.copy()
    for name in variables:
        if name not in ds.variables:
            continue
        attrs = dict(ds[name].attrs)
        if "units" not in attrs:
            default = _DEFAULT_VAR_UNITS.get(name, "dimensionless")
            attrs["units"] = default
            ds[name] = ds[name].assign_attrs(attrs)
    return ds


def coriolis_parameter(
    lat: xr.DataArray | float,
) -> xr.DataArray | float:
    """Coriolis parameter ``f = 2 Ω sin(φ)`` in s⁻¹.

    Args:
        lat: Latitude(s) in **degrees**.

    Returns:
        Coriolis parameter with the same shape as ``lat``.
    """
    radians = np.deg2rad(lat)
    return mpcalc.coriolis_parameter(radians)


def streamfunction(
    ds: xr.Dataset,
    variable: str = "ssh",
    g: float | None = None,
    f0: float | None = None,
) -> xr.Dataset:
    """Stream function from sea-surface height.

    Uses the linear geostrophic approximation ``η = (g / f₀) ψ``, i.e.
    ``ψ = (g / f₀) η``.

    Args:
        ds: Dataset containing ``variable`` (SSH) and a ``lat`` coord.
        variable: Name of the SSH variable.
        g: Gravity in m/s². Defaults to MetPy's ``earth_gravity``.
        f0: Coriolis parameter in s⁻¹. Defaults to the mean over the
            latitude coordinate of ``ds``.

    Returns:
        Dataset with a single variable ``"psi"`` (stream function).
    """
    ssh = _assume_units(ds, (variable,))[variable]

    f0_q = (
        mpcalc.coriolis_parameter(np.deg2rad(ssh.lat.pint.dequantify())).mean()
        if f0 is None
        else Quantity(f0, "1/s")
    )
    g_q = GRAVITY if g is None else g * GRAVITY.units

    psi = (g_q / f0_q) * ssh.pint.quantify()
    psi.attrs.update(
        long_name="Stream Function",
        standard_name="stream_function",
    )
    return psi.pint.dequantify().to_dataset(name="psi")


def geostrophic_velocities(
    ds: xr.Dataset,
    variable: str = "ssh",
) -> xr.Dataset:
    """Geostrophic ``u`` and ``v`` from a height field (SSH).

    Delegates to :func:`metpy.calc.geostrophic_wind`, which expects a
    **height** input (units of length) and applies the ``g / f`` scaling
    internally::

        u = -(g / f) ∂h/∂y
        v =  (g / f) ∂h/∂x

    Do **not** pass a stream-function field here — applying
    ``geostrophic_wind`` to ``ψ = (g/f) η`` would double the scaling and
    give unit-inconsistent velocities. :func:`streamfunction` is
    provided as a separate diagnostic.

    Args:
        ds: Dataset containing ``variable``.
        variable: Name of the SSH (height) variable.

    Returns:
        Dataset with ``u`` and ``v`` variables in m/s.
    """
    qds = _assume_units(ds, (variable,)).metpy.quantify()
    u, v = mpcalc.geostrophic_wind(height=qds[variable])
    out = xr.Dataset(dict(u=u, v=v)).metpy.dequantify()
    out = out.assign_coords(lat=ds.lat, lon=ds.lon)
    out["u"].attrs.update(long_name="Zonal Velocity", standard_name="zonal_velocity")
    out["v"].attrs.update(
        long_name="Meridional Velocity", standard_name="meridional_velocity"
    )
    return out


def kinetic_energy(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Kinetic energy ``0.5 (u² + v²)``."""
    qds = _assume_units(ds, (u, v)).metpy.quantify()
    ke = 0.5 * (qds[u] ** 2 + qds[v] ** 2)
    ke.attrs.update(long_name="Kinetic Energy", standard_name="kinetic_energy")
    return ke.to_dataset(name="ke").metpy.dequantify()


def relative_vorticity(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Relative vorticity ``ζ = ∂v/∂x - ∂u/∂y``."""
    qds = _assume_units(ds, (u, v)).metpy.quantify()
    zeta = mpcalc.vorticity(
        u=qds[u], v=qds[v], latitude=ds.lat, longitude=ds.lon
    ).assign_coords(lat=ds.lat, lon=ds.lon)
    zeta.attrs.update(
        long_name="Relative Vorticity", standard_name="relative_vorticity"
    )
    return zeta.to_dataset(name="vort_r").metpy.dequantify()


def absolute_vorticity(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Absolute vorticity ``η = ζ + f``."""
    qds = _assume_units(ds, (u, v)).metpy.quantify()
    eta = mpcalc.absolute_vorticity(
        u=qds[u], v=qds[v], latitude=ds.lat, longitude=ds.lon
    )
    eta.attrs.update(long_name="Absolute Vorticity", standard_name="absolute_vorticity")
    return eta.to_dataset(name="vort_a").metpy.dequantify()


def divergence(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Horizontal divergence ``∂u/∂x + ∂v/∂y``."""
    qds = _assume_units(ds, (u, v)).metpy.quantify()
    div = mpcalc.divergence(u=qds[u], v=qds[v], latitude=ds.lat, longitude=ds.lon)
    div.attrs.update(long_name="Divergence", standard_name="divergence")
    return div.to_dataset(name="div").metpy.dequantify()


def enstrophy(
    ds: xr.Dataset,
    variable: str = "vort_r",
) -> xr.Dataset:
    """Enstrophy ``0.5 ζ²``."""
    qds = _assume_units(ds, (variable,)).metpy.quantify()
    ens = 0.5 * (qds[variable] ** 2)
    ens.attrs.update(long_name="Enstrophy", standard_name="enstrophy")
    return ens.to_dataset(name="ens").metpy.dequantify()


def shear_strain(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Shear strain ``Sₛ = ∂v/∂x + ∂u/∂y`` via metpy."""
    qds = _assume_units(ds, (u, v)).metpy.quantify()
    sh = mpcalc.shearing_deformation(
        u=qds[u], v=qds[v], latitude=ds.lat, longitude=ds.lon
    )
    sh.attrs.update(long_name="Shear Strain", standard_name="shear_strain")
    return sh.to_dataset(name="shear_strain").metpy.dequantify()


def tensor_strain(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Normal / tensor strain ``Sₙ = ∂u/∂x - ∂v/∂y`` via metpy."""
    qds = _assume_units(ds, (u, v)).metpy.quantify()
    st = mpcalc.stretching_deformation(
        u=qds[u], v=qds[v], latitude=ds.lat, longitude=ds.lon
    )
    st.attrs.update(long_name="Tensor Strain", standard_name="tensor_strain")
    return st.to_dataset(name="tensor_strain").metpy.dequantify()


def strain_magnitude(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Total strain magnitude ``sqrt(Sₙ² + Sₛ²)``."""
    qds = _assume_units(ds, (u, v)).metpy.quantify()
    total = mpcalc.total_deformation(
        u=qds[u],
        v=qds[v],
        latitude=ds.lat.metpy.dequantify(),
        longitude=ds.lon.metpy.dequantify(),
    )
    total.attrs.update(long_name="Strain Magnitude", standard_name="strain")
    return total.to_dataset(name="strain").metpy.dequantify()


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
    qds = _assume_units(ds, (variable,)).metpy.quantify()
    f = (
        mpcalc.coriolis_parameter(np.deg2rad(ds.lat)).mean()
        if f0 is None
        else Quantity(f0, "1/s")
    )
    # qds[variable] is already quantified via ds.metpy.quantify();
    # don't double-wrap.
    qds[variable] = qds[variable] / f
    return qds.metpy.dequantify()
