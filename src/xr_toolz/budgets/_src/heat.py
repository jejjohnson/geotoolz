"""Heat-budget residual — V4.3."""

from __future__ import annotations

import xarray as xr

from xr_toolz.budgets._src._common import _tracer_budget_residual


def heat_budget_residual(
    ds: xr.Dataset,
    *,
    temp_var: str = "theta",
    u_var: str = "u",
    v_var: str = "v",
    w_var: str | None = None,
    surface_flux_var: str | None = None,
    time_dim: str = "time",
    lat: str = "lat",
    lon: str = "lon",
    depth: str | None = "depth",
) -> xr.DataArray:
    """Per-cell heat-budget residual.

    Computes ``∂θ/∂t + ∇·(u θ) - F_surface`` where ``F_surface`` is an
    optional surface heat-flux source field. ``ρ c_p`` factors are
    omitted: the residual is in units of ``[θ] / s``, not W/m³. For
    closure tests this is sufficient.

    Args:
        ds: Dataset with potential temperature + velocities.
        temp_var, u_var, v_var, w_var: Variable names. ``w_var=None``
            silently skips the vertical-advection term — the residual
            is then a surface-only diagnostic.
        surface_flux_var: Optional source field on the surface plane.
        time_dim: Name of the time coordinate.
        lat, lon: Names of horizontal coords (degrees).
        depth: Vertical coord name. ``None`` skips vertical advection.

    Returns:
        Per-cell residual, same dims as ``ds[temp_var]``.

    Notes:
        ``ρ c_p θ`` would be the proper heat content; we use ``θ``
        directly so the closure test does not require a density field.
        Multiply the output by ``ρ_0 c_p`` upstream if you need W/m³.
    """
    return _tracer_budget_residual(
        ds,
        tracer_var=temp_var,
        u_var=u_var,
        v_var=v_var,
        w_var=w_var,
        surface_flux_var=surface_flux_var,
        time_dim=time_dim,
        lat=lat,
        lon=lon,
        depth=depth,
    )


__all__ = ["heat_budget_residual"]
