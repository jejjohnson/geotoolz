"""Volume-budget residual — V4.3.

For an incompressible ocean, ``∇·u = 0``. The volume budget is the
divergence of the velocity field; deviations from zero indicate either
a compressible flow representation or numerical noise.
"""

from __future__ import annotations

import xarray as xr

import xrgrad


def volume_budget_residual(
    ds: xr.Dataset,
    *,
    u_var: str = "u",
    v_var: str = "v",
    w_var: str | None = None,
    lat: str = "lat",
    lon: str = "lon",
    depth: str | None = "depth",
) -> xr.DataArray:
    """``∇·u`` with spherical curvature ``+ ∂w/∂z`` if ``w_var`` is given.

    Closes to ≈ 0 for an incompressible flow. ``w_var=None`` returns
    the surface 2-D divergence only. Uses :func:`xrgrad.divergence`
    so the spherical curvature term is included; when the vertical
    component is present it goes in the same call as a third
    ``rectilinear`` axis rather than a separately added partial.
    """
    if w_var is not None and depth is not None and depth in ds[w_var].dims:
        return xrgrad.divergence(
            ds[[u_var, v_var, w_var]],
            (u_var, v_var, w_var),
            dims=(lon, lat, depth),
            geometry=("spherical", "spherical", "rectilinear"),
            lon=lon,
            lat=lat,
        ).rename("volume_budget_residual")
    return xrgrad.divergence(
        ds[[u_var, v_var]],
        (u_var, v_var),
        dims=(lon, lat),
        geometry="spherical",
        lon=lon,
        lat=lat,
    ).rename("volume_budget_residual")


__all__ = ["volume_budget_residual"]
