"""Salt-budget residual — V4.3."""

from __future__ import annotations

import xarray as xr

from xr_toolz.budgets._src._common import _tracer_budget_residual


def salt_budget_residual(
    ds: xr.Dataset,
    *,
    salt_var: str = "so",
    u_var: str = "u",
    v_var: str = "v",
    w_var: str | None = None,
    surface_flux_var: str | None = None,
    time_dim: str = "time",
    lat: str = "lat",
    lon: str = "lon",
    depth: str | None = "depth",
) -> xr.DataArray:
    """Per-cell salt-budget residual ``∂S/∂t + ∇·(u S) - F_surface``."""
    return _tracer_budget_residual(
        ds,
        tracer_var=salt_var,
        u_var=u_var,
        v_var=v_var,
        w_var=w_var,
        surface_flux_var=surface_flux_var,
        time_dim=time_dim,
        lat=lat,
        lon=lon,
        depth=depth,
    )


__all__ = ["salt_budget_residual"]
