"""Vertical-column diagnostics: column integrals, hypsometric height, PBL.

Every function here reduces or accumulates along one vertical dimension
and works on both eager and dask-backed arrays: the level index is
dropped internally so slices along the vertical align by position, and no
data are materialised.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from xrreader.types import Variable, apply_cf_attrs

from xrtoolz.atm._src.constants import EPSILON_TV, R_D, G


_GEOPOTENTIAL_HEIGHT = Variable(
    name="geopotential_height",
    standard_name="geopotential_height",
    long_name="Geopotential height",
    units="m",
)


def _positional(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Drop every coordinate on ``dim`` so slices along it align by position."""
    return da.drop_vars([name for name, c in da.coords.items() if dim in c.dims])


def _lo_hi(da: xr.DataArray, dim: str) -> tuple[xr.DataArray, xr.DataArray]:
    """``(da[k], da[k+1])`` pairs along ``dim`` (each of length ``n − 1``)."""
    return da.isel({dim: slice(None, -1)}), da.isel({dim: slice(1, None)})


def _cell_widths(z: xr.DataArray, dim: str) -> xr.DataArray:
    """Δz per level from the coordinate: half-way to each neighbour.

    Interior levels get ``(z[k+1] − z[k−1]) / 2``; the end levels get half
    of their single adjacent spacing, so the widths sum to ``z[-1] − z[0]``.
    """
    z_lo, z_hi = _lo_hi(z, dim)
    faces = z_hi - z_lo
    interior = 0.5 * (
        faces.isel({dim: slice(1, None)}) + faces.isel({dim: slice(None, -1)})
    )
    first = 0.5 * faces.isel({dim: slice(0, 1)})
    last = 0.5 * faces.isel({dim: slice(-1, None)})
    return xr.concat([first, interior, last], dim=dim)


def column_integral(
    da: xr.DataArray,
    *,
    dim: str = "z",
    coord: str | xr.DataArray | None = None,
    method: Literal["trapezoid", "sum"] = "trapezoid",
) -> xr.DataArray:
    """Integrate ``da`` along the vertical coordinate.

    ``method="trapezoid"`` applies the trapezoid rule on the coordinate
    values (exact for fields linear in ``z``, the right choice for
    endpoint-inclusive level grids):

    ``∫ f dz ≈ Σ_k ½ (f_k + f_{k+1}) (z_{k+1} − z_k)``

    ``method="sum"`` is the cell-centred Riemann sum ``Σ_k f_k Δz_k`` with
    ``Δz_k`` the cell width around level ``k`` (half-way to each neighbour,
    half a spacing at the ends). Both methods give ``c (z_1 − z_0)`` for a
    uniform field ``c`` on ``[z_0, z_1]`` — unlike ``sum · dz`` with a
    constant ``dz``, which over-counts an endpoint-inclusive grid by
    ``n_z / (n_z − 1)``. The sign follows the coordinate's orientation.

    Args:
        da: Field to integrate (any dims, must include ``dim``).
        dim: Vertical dimension.
        coord: Coordinate giving the position of each level: ``None`` uses
            the index coordinate on ``dim``; a ``str`` names a coordinate
            of ``da`` (may be multi-dimensional, e.g. a terrain-following
            ``z_agl``); a ``DataArray`` is used directly and must carry
            ``dim``.
        method: ``"trapezoid"`` or ``"sum"``.

    Returns:
        ``da`` reduced over ``dim``; units are ``da`` units × coordinate
        units. A NaN at any level makes the whole column NaN.

    Raises:
        ValueError: If ``dim`` is missing, has fewer than two levels, no
            coordinate is available, or ``method`` is unknown.
    """
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not in DataArray dims {tuple(da.dims)}")
    if da.sizes[dim] < 2:
        raise ValueError(f"column_integral needs at least two levels along {dim!r}")
    if method not in ("trapezoid", "sum"):
        raise ValueError(f"method must be 'trapezoid' or 'sum', got {method!r}")
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

    f = _positional(da, dim)
    z = _positional(z, dim)
    if method == "trapezoid":
        f_lo, f_hi = _lo_hi(f, dim)
        z_lo, z_hi = _lo_hi(z, dim)
        out = (0.5 * (f_lo + f_hi) * (z_hi - z_lo)).sum(dim, skipna=False)
    else:
        out = (f * _cell_widths(z, dim)).sum(dim, skipna=False)
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
    temperature ``T_v = T (1 + 0.61 q)``:

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
    tv = t * (1.0 + EPSILON_TV * ds[specific_humidity]) if specific_humidity else t
    ps = ds[surface_pressure]
    # 1-D pressure axes may run top-down (CDS/ERA5 style); integrate
    # surface-first and flip the result back into the dataset's order.
    flip = p.ndim == 1 and bool(p.values[0] < p.values[-1])
    if flip:
        p = p.isel({level: slice(None, None, -1)})
        tv = tv.isel({level: slice(None, None, -1)})

    p_pos = _positional(p, level)
    tv_pos = _positional(tv, level)
    p_lo, p_hi = _lo_hi(p_pos, level)
    tv_lo, tv_hi = _lo_hi(tv_pos, level)
    dz_layers = (R_D / G) * 0.5 * (tv_lo + tv_hi) * np.log(p_lo / p_hi)
    first = {level: slice(0, 1)}
    dz_first = (R_D / G) * tv_pos.isel(first) * np.log(ps / p_pos.isel(first))
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


def _first_crossing(ri: np.ndarray, z: np.ndarray, ri_crit: float) -> np.ndarray:
    """Height where ``ri`` first reaches ``ri_crit`` (levels on the last axis).

    Linear interpolation between the bracketing levels; NaN where no level
    above the surface reaches the threshold.
    """
    ri = ri.copy()
    ri[..., 0] = 0.0  # 0/0 at the surface level by construction
    above = ri >= ri_crit
    above[..., 0] = False
    found = above.any(axis=-1)
    k = np.argmax(above, axis=-1)
    k = np.where(found, k, 1)  # keep the gather in bounds where not found
    idx = np.indices(k.shape, sparse=True)
    ri_k, ri_km1 = ri[(*idx, k)], ri[(*idx, k - 1)]
    z_k, z_km1 = z[(*idx, k)], z[(*idx, k - 1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (ri_crit - ri_km1) / (ri_k - ri_km1)
    h = z_km1 + frac * (z_k - z_km1)
    return np.where(found, h, np.nan)


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
    unstable (never a crossing) and ``0`` when ``θ_v = θ_{v,s}``.

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
    surface = {level: 0}
    thv_s = thv.isel(surface)
    z_s = z.isel(surface)
    shear2 = (ds[u] - ds[u].isel(surface)) ** 2 + (ds[v] - ds[v].isel(surface)) ** 2
    numerator = G * (thv - thv_s) * (z - z_s)
    # Calm levels (zero shear relative to the surface) have infinite Ri_b
    # with the sign of the numerator (0 where the numerator is zero too):
    # mask the denominator so the division never warns, then restore the
    # limit where the shear was zero (NaN inputs stay NaN). The surface
    # level is zeroed in the kernel.
    calm = shear2 == 0.0
    calm_ri = xr.where(numerator > 0.0, np.inf, 0.0).where(numerator >= 0.0, -np.inf)
    calm_ri = calm_ri.where(numerator.notnull())
    ri = (numerator / (thv_s * shear2.where(~calm))).where(~calm, calm_ri)
    ri, z_b = xr.broadcast(_positional(ri, level), _positional(z, level))
    if ri.chunks is not None or z_b.chunks is not None:
        # The kernel needs the whole column: one chunk along ``level``.
        ri, z_b = ri.chunk({level: -1}), z_b.chunk({level: -1})
    h = xr.apply_ufunc(
        _first_crossing,
        ri,
        z_b,
        input_core_dims=[[level], [level]],
        kwargs={"ri_crit": ri_crit},
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    out = ds.copy()
    out[name] = apply_cf_attrs(h, "blh", overwrite=True)
    return out
