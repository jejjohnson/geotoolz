"""Vertical-column diagnostics: column integrals, hypsometric height, PBL.

Every function here reduces or accumulates along one vertical dimension
and works on both eager and dask-backed arrays: the level index is
dropped internally so slices along the vertical align by position, and no
data are materialised.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from diagnostix import columns
from geophysix.constants import STANDARD
from geophysix.thermo import virtual_temperature
from xrproducts.types import Variable, apply_cf_attrs

from xrops._src.columns import on_columns, positional


#: ``R_d / g₀`` [m K⁻¹], the hypsometric scale per kelvin.
_R_D_OVER_G = STANDARD.dry_air_gas_constant / STANDARD.standard_gravity

_GEOPOTENTIAL_HEIGHT = Variable(
    name="geopotential_height",
    standard_name="geopotential_height",
    long_name="Geopotential height",
    units="m",
)


def _lo_hi(da: xr.DataArray, dim: str) -> tuple[xr.DataArray, xr.DataArray]:
    """``(da[k], da[k+1])`` pairs along ``dim`` (each of length ``n − 1``)."""
    return da.isel({dim: slice(None, -1)}), da.isel({dim: slice(1, None)})


def column_integral(
    da: xr.DataArray,
    *,
    dim: str = "z",
    coord: str | xr.DataArray | None = None,
) -> xr.DataArray:
    """Integrate ``da`` along the vertical coordinate, by the trapezoid rule.

    ``∫ f dz ≈ Σ_k ½ (f_k + f_{k+1}) (z_{k+1} − z_k)``

    (``diagnostix.columns.column_integral``). It is exact for a field linear
    in ``z``, and equals the sum ``Σ_k f_k Δz_k`` over cells that end half-way
    to each neighbour and at the first and last levels — unlike ``sum · dz``
    with a constant ``dz``, which over-counts an endpoint-inclusive grid by
    ``n_z / (n_z − 1)``. The sign follows the coordinate's orientation.

    Args:
        da: Field to integrate (any dims, must include ``dim``).
        dim: Vertical dimension.
        coord: Coordinate giving the position of each level: ``None`` uses
            the index coordinate on ``dim``; a ``str`` names a coordinate
            of ``da`` (may be multi-dimensional, e.g. a terrain-following
            ``z_agl``); a ``DataArray`` is used directly and must carry
            ``dim``.

    Returns:
        ``da`` reduced over ``dim``; units are ``da`` units × coordinate
        units. A NaN at any level makes the whole column NaN.

    Raises:
        ValueError: If ``dim`` is missing, has fewer than two levels, no
            coordinate is available.
    """
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not in DataArray dims {tuple(da.dims)}")
    if da.sizes[dim] < 2:
        raise ValueError(f"column_integral needs at least two levels along {dim!r}")
    if coord is None:
        if dim not in da.coords:
            raise ValueError(
                f"no coordinate on {dim!r}; pass `coord=` to locate the levels"
            )
        z = da[dim]
    elif isinstance(coord, str):
        z = da[coord]
    else:
        z = coord
    if dim not in z.dims:
        raise ValueError(f"coordinate {z.name!r} does not carry dim {dim!r}")

    out = on_columns(columns.column_integral, da, z, dim=dim, reduces=True)
    out = out.transpose(*(d for d in da.dims if d != dim), ...)
    return out.rename(da.name)


def hypsometric_height(
    ds: xr.Dataset,
    *,
    pressure: str = "pressure",
    temperature: str = "temperature",
    specific_humidity: str | None = None,
    surface_pressure: str = "sp",
    surface_height: str | None = None,
    level: str = "level",
    name: str = "height",
) -> xr.Dataset:
    """Geopotential height of pressure levels from the hypsometric equation.

    Integrates layer by layer upward from the surface with the virtual
    temperature ``T_v = T (1 + (1/ε − 1) q)``, ``1/ε − 1 ≈ 0.608``
    (`geophysix.thermo.virtual_temperature`):

    ``z_{k+1} = z_k + (R_d T̄_{v,k} / g) ln(p_k / p_{k+1})``

    where ``T̄_{v,k}`` is the mean of the two bounding levels. The first
    level is anchored to the surface, ``z_0 = z_s + (R_d T_{v,0} / g)
    ln(p_s / p_0)``, using that level's virtual temperature for the
    surface layer. For an isothermal column this reduces to
    ``z = (R_d T / g) ln(p_s / p)`` exactly. Levels with ``p > p_s`` get
    negative heights (extrapolated below ground). A NaN temperature at a
    level invalidates that level and every level above it.

    ``pressure`` and ``surface_pressure`` must share units (only ratios
    enter). A 1-D pressure coordinate may be ordered either way; a
    multi-dimensional pressure field must be ordered surface first.

    Args:
        ds: Dataset with the pressure, temperature and surface-pressure
            fields (``pressure`` may be the ``level`` coordinate itself).
        pressure: Pressure per level.
        temperature: Air temperature per level [K].
        specific_humidity: Optional specific humidity [kg kg⁻¹] for the
            virtual-temperature correction; ``None`` treats the air as dry.
        surface_pressure: Surface pressure (no ``level`` dim).
        surface_height: Optional terrain height [m] added to every level;
            ``None`` measures height above the surface.
        level: Vertical dimension.
        name: Name of the output variable.

    Returns:
        ``ds`` with ``name`` added (CF ``geopotential_height``, m).

    Raises:
        ValueError: If ``pressure`` or ``temperature`` lack ``level``.
    """
    p = ds[pressure]
    t = ds[temperature]
    for label, da in ((pressure, p), (temperature, t)):
        if level not in da.dims:
            raise ValueError(f"{label!r} does not carry the level dim {level!r}")
    tv = virtual_temperature(t, ds[specific_humidity]) if specific_humidity else t
    ps = ds[surface_pressure]
    # 1-D pressure axes may run top-down (CDS/ERA5 style); integrate
    # surface-first and flip the result back into the dataset's order.
    flip = p.ndim == 1 and bool(p.values[0] < p.values[-1])
    if flip:
        p = p.isel({level: slice(None, None, -1)})
        tv = tv.isel({level: slice(None, None, -1)})

    p_pos = positional(p, level)
    tv_pos = positional(tv, level)
    p_lo, p_hi = _lo_hi(p_pos, level)
    tv_lo, tv_hi = _lo_hi(tv_pos, level)
    dz_layers = _R_D_OVER_G * 0.5 * (tv_lo + tv_hi) * np.log(p_lo / p_hi)
    first = {level: slice(0, 1)}
    dz_first = _R_D_OVER_G * tv_pos.isel(first) * np.log(ps / p_pos.isel(first))
    z = xr.concat([dz_first, dz_layers], dim=level).cumsum(level, skipna=False)
    if surface_height is not None:
        z = z + ds[surface_height]
    if flip:
        z = z.isel({level: slice(None, None, -1)})
    z = z.transpose(*[d for d in t.dims if d in z.dims], ...)
    if level in ds.coords:
        z = z.assign_coords({level: ds[level]})
    out = ds.copy()
    out[name] = apply_cf_attrs(z, _GEOPOTENTIAL_HEIGHT, overwrite=True)
    return out


def pbl_height_bulk_richardson(
    ds: xr.Dataset,
    *,
    theta_v: str,
    u: str = "u",
    v: str = "v",
    height: str = "z_agl",
    ri_crit: float = 0.25,
    level: str = "level",
    name: str = "pbl_height",
) -> xr.Dataset:
    """Boundary-layer height from the bulk Richardson number.

    Vogelezang & Holtslag (1996) form without the surface-friction term,
    with the surface values taken from the lowest level ``s``:

    ``Ri_b(z) = g (θ_v(z) − θ_{v,s}) (z − z_s)
    / (θ_{v,s} ((u(z) − u_s)² + (v(z) − v_s)²))``

    ``h`` is the first height (scanning upward from the level above the
    surface) where ``Ri_b ≥ Ri_crit``, linearly interpolated between the
    bracketing levels. Columns that never reach ``Ri_crit`` get NaN rather
    than the top level. The surface level itself is assigned ``Ri_b = 0``;
    a level aloft with no shear relative to the surface (``u = u_s``,
    ``v = v_s``) takes the sign of the numerator: ``+∞`` when stably
    stratified (so ``h`` stops at the level below it), ``−∞`` when
    unstable (never a crossing) and ``0`` when ``θ_v = θ_{v,s}``. A NaN at
    or below ``h`` gives NaN (``diagnostix.columns.boundary_layer_height``).

    The vertical ordering is read from ``height``: profiles may be surface
    first or top first (they are flipped internally), but every column must
    share one orientation.

    Args:
        ds: Dataset with virtual potential temperature, wind components and
            height per level.
        theta_v: Virtual potential temperature [K].
        u: Eastward wind [m s⁻¹].
        v: Northward wind [m s⁻¹].
        height: Height above ground per level [m] (may be a coordinate,
            1-D or per column).
        ri_crit: Critical bulk Richardson number.
        level: Vertical dimension.
        name: Name of the output variable.

    Returns:
        ``ds`` with ``name`` added (CF
        ``atmosphere_boundary_layer_thickness``, m); the ``level`` dim is
        reduced away.

    Raises:
        ValueError: If ``theta_v`` or ``height`` lack ``level``, or if the
            vertical orientation of ``height`` differs between columns.
    """
    thv = ds[theta_v]
    z = ds[height]
    for label, da in ((theta_v, thv), (height, z)):
        if level not in da.dims:
            raise ValueError(f"{label!r} does not carry the level dim {level!r}")
    # Orientation from the height field: flip top-first profiles so the
    # surface is level 0. Columns whose end levels are NaN are ignored.
    z_first, z_last = z.isel({level: 0}), z.isel({level: -1})
    n_top_first = int((z_first > z_last).sum())
    n_surface_first = int((z_first < z_last).sum())
    if n_top_first and n_surface_first:
        raise ValueError(
            f"{height!r} is surface-first in some columns and top-first in "
            "others; the vertical orientation must be consistent"
        )
    if n_top_first:
        ds = ds.isel({level: slice(None, None, -1)})
        thv, z = ds[theta_v], ds[height]
    h = on_columns(
        lambda theta, u_, v_, height_: columns.boundary_layer_height(
            theta, u_, v_, height_, critical=ri_crit
        ),
        thv,
        ds[u],
        ds[v],
        z,
        dim=level,
        reduces=True,
    )
    out = ds.copy()
    out[name] = apply_cf_attrs(h, "blh", overwrite=True)
    return out
