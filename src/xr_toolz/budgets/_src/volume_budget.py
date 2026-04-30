"""Volume-budget residual — V4.3.

For an incompressible ocean, ``∇·u = 0``. The volume budget is the
divergence of the velocity field; deviations from zero indicate either
a compressible flow representation or numerical noise.
"""

from __future__ import annotations

import xarray as xr

from xr_toolz import calc


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
    the surface 2-D divergence only. Uses :func:`xr_toolz.calc.divergence`
    so the spherical curvature term is included.
    """
    flow = ds[[u_var, v_var]]
    div = calc.divergence(
        flow,
        (u_var, v_var),
        dims=(lon, lat),
        geometry="spherical",
        lon=lon,
        lat=lat,
    )
    if w_var is not None and depth is not None and depth in ds[w_var].dims:
        div = div + calc.partial(ds[w_var], depth, geometry="rectilinear")
    return div.rename("volume_budget_residual")


__all__ = ["volume_budget_residual"]
