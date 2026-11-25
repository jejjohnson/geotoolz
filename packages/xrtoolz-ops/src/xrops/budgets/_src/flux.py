"""The flux of a quantity out of a region."""

from __future__ import annotations

import numpy as np
import xarray as xr
from diagnostix import budgets

from xrops._src.collocated import Geometry, over_grid


def boundary_flux(
    ds: xr.Dataset,
    *,
    region: xr.DataArray,
    variable: str | None = None,
    u: str = "u",
    v: str = "v",
    thickness: str | None = None,
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
) -> xr.DataArray:
    """The net flux of a quantity out of a region, through its own faces.

    ``∮ q H u·n dl`` over the faces between the cells of the region and the
    others (``diagnostix.budgets.boundary_flux``): what passes from one cell of
    the region to another cancels. Its sum over a partition of a region is the
    flux out of the whole.

    Args:
        ds: Dataset with the velocities and, if named, the quantity and the
            thickness.
        region: Boolean mask on the two horizontal dims, ``True`` inside. It may
            not touch an edge of the domain; a whole circle of longitudes has
            none along ``lon``.
        variable: Name of the quantity ``q``; ``None`` for 1, the volume flux.
        u, v: Velocity component names (m/s).
        thickness: Name of the thickness ``H`` (m) of a layer, such as the
            heights of the levels; ``None`` for 1, the flux per metre of depth.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default; lon/lat in degrees),
            ``"cartesian"`` or ``"rectilinear"``.

    Returns:
        DataArray ``"boundary_flux"``, positive outward, on the dims of the
        fields but the two horizontal ones. The units are those of ``q`` m²/s,
        or m³/s with a thickness. A NaN beside a face of the boundary gives NaN.

    Raises:
        ValueError: If ``region`` is not boolean on the two dims, has no cell or
            every cell, or touches an edge of the domain.
    """
    xdim, ydim = dims
    if set(region.dims) != set(dims) or region.dtype != bool:
        raise ValueError(
            f"region must be boolean on the dims {dims!r}; got {region.dtype} on "
            f"{region.dims!r}."
        )
    inside = np.asarray(region.transpose(ydim, xdim).values)
    q = xr.ones_like(ds[u]) if variable is None else ds[variable]
    fields = [q, ds[u], ds[v]]
    if thickness is not None:
        # the kernel multiplies q by H, whatever dims H has
        fields[0] = q * ds[thickness]

    def kernel(q_, u_, v_, grid):
        volume = budgets.ControlVolume.from_region(inside, grid)
        return budgets.boundary_flux(q_, u_, v_, grid, volume)

    out = over_grid(kernel, *fields, dims=dims, geometry=geometry)
    return out.rename("boundary_flux")


__all__ = ["boundary_flux"]
