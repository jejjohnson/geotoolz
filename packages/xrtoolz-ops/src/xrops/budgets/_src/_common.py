"""Shared helpers of the budget residuals."""

from __future__ import annotations

import numpy as np
import xarray as xr
from diagnostix import budgets, columns

from xrops._src.collocated import on_grid
from xrops._src.columns import levels_of, on_columns
from xrops.budgets._src.residual import budget_residual


def _time_derivative(da: xr.DataArray, time_dim: str) -> xr.DataArray:
    """``∂da/∂t`` in per-second units, second-order on uneven times.

    A datetime coordinate is converted to seconds; any other is taken to be in
    seconds already.
    """
    if time_dim not in da.dims:
        return xr.zeros_like(da)
    coord = da[time_dim]
    if np.issubdtype(coord.dtype, np.datetime64):
        return da.differentiate(time_dim, datetime_unit="s")
    return da.differentiate(time_dim)


def _flux_divergence(
    q: xr.DataArray,
    *,
    u: xr.DataArray,
    v: xr.DataArray,
    w: xr.DataArray | None,
    lat: str,
    lon: str,
    depth: str | None,
) -> xr.DataArray:
    """``∇·(u q)``: through the four faces of each cell, and along the vertical.

    The horizontal part is ``diagnostix.budgets.flux_divergence``, NaN on the
    edge cells but for a whole circle of longitudes; the vertical part is the
    derivative of ``w q`` along ``depth``, where there is a ``w``.
    """
    out = on_grid(
        budgets.flux_divergence,
        q,
        u,
        v,
        dims=(lon, lat),
        geometry="spherical",
        wrap=True,
    )
    if w is None or depth is None or depth not in w.dims:
        return out
    flux = w * q
    along = on_columns(
        columns.vertical_derivative,
        flux,
        levels_of(flux, depth),
        dim=depth,
        reduces=False,
    )
    return out + along


def _cell_residual(
    ds: xr.Dataset,
    q: xr.DataArray,
    *,
    u_var: str,
    v_var: str,
    w_var: str | None,
    sources: tuple[str, ...],
    time_dim: str,
    lat: str,
    lon: str,
    depth: str | None,
    name: str,
) -> xr.DataArray:
    """``∂q/∂t + ∇·(u q) − Σ sources`` of each cell, 0 in a closed budget."""
    res = budget_residual(
        _time_derivative(q, time_dim),
        _flux_divergence(
            q,
            u=ds[u_var],
            v=ds[v_var],
            w=ds[w_var] if w_var is not None else None,
            lat=lat,
            lon=lon,
            depth=depth,
        ),
    )
    for source in sources:
        res = res - ds[source]
    # a residual is per second, so the units of q are not its own
    return res.transpose(*q.dims, ...).rename(name).drop_attrs(deep=False)
