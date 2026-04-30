"""Kinetic-energy-budget residual — V4.3."""

from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from xr_toolz.budgets._src._common import _flux_divergence, _time_derivative


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
    fields summed into the source term.
    """
    u = ds[u_var]
    v = ds[v_var]
    e = 0.5 * (u**2 + v**2)
    tendency = _time_derivative(e, time_dim)
    w = ds[w_var] if w_var is not None else None
    flux_div = _flux_divergence(e, u=u, v=v, w=w, lat=lat, lon=lon, depth=depth)
    res = tendency + flux_div
    if forcing_vars:
        for name in forcing_vars:
            res = res - ds[name]
    return res.rename("kinetic_energy_budget_residual")


__all__ = ["kinetic_energy_budget_residual"]
