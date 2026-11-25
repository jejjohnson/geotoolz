"""Kinetic-energy-budget residual — V4.3."""

from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from xrops.budgets._src._common import _cell_residual


def kinetic_energy_budget_residual(
    ds: xr.Dataset,
    *,
    u_var: str = "u",
    v_var: str = "v",
    forcing_vars: Sequence[str] | None = None,
    time_dim: str = "time",
    lat: str = "lat",
    lon: str = "lon",
    depth: str | None = "depth",
    w_var: str | None = None,
) -> xr.DataArray:
    """Per-cell KE-budget residual.

    KE per unit mass ``E = 0.5 (u² + v²)``. Residual::

        ∂E/∂t + ∇·(u E) - Σ forcings

    Forcings (e.g. wind-stress work) are supplied as additional source
    fields summed into the source term. The edge cells of the domain are NaN,
    as in :func:`~xrops.budgets.heat_budget_residual`.
    """
    return _cell_residual(
        ds,
        0.5 * (ds[u_var] ** 2 + ds[v_var] ** 2),
        u_var=u_var,
        v_var=v_var,
        w_var=w_var,
        sources=tuple(forcing_vars or ()),
        time_dim=time_dim,
        lat=lat,
        lon=lon,
        depth=depth,
        name="kinetic_energy_budget_residual",
    )


__all__ = ["kinetic_energy_budget_residual"]
