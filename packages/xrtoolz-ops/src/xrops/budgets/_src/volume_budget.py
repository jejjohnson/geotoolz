"""Volume-budget residual — V4.3.

For an incompressible ocean, ``∇·u = 0``. The volume budget is the
divergence of the velocity field; deviations from zero indicate either
a compressible flow representation or numerical noise.
"""

from __future__ import annotations

import xarray as xr

from xrops.budgets._src._common import _flux_divergence


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
    """``∇·u`` through the faces of each cell, ``+ ∂w/∂z`` if ``w_var`` is given.

    Closes to ≈ 0 for an incompressible flow. ``w_var=None`` returns the
    surface 2-D divergence only. The edge cells of the domain are NaN, as in
    :func:`~xrops.budgets.heat_budget_residual`.
    """
    u = ds[u_var]
    return _flux_divergence(
        xr.ones_like(u),
        u=u,
        v=ds[v_var],
        w=ds[w_var] if w_var is not None else None,
        lat=lat,
        lon=lon,
        depth=depth,
    ).rename("volume_budget_residual")


__all__ = ["volume_budget_residual"]
