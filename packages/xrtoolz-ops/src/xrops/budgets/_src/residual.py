"""Generic budget-residual combinator — V4.2."""

from __future__ import annotations

import xarray as xr


def budget_residual(
    tendency: xr.DataArray,
    flux_divergence: xr.DataArray,
    *,
    source: xr.DataArray | None = None,
    sink: xr.DataArray | None = None,
) -> xr.DataArray:
    """Combine the four canonical budget terms into a residual.

    Sign convention (matches the V4.3 tracer / KE residuals)::

        residual = tendency + flux_divergence - source + sink

    where ``flux_divergence`` is ``∇·(u φ)`` — the divergence of the
    advective flux *out* of each cell. The conservation equation
    ``∂φ/∂t + ∇·(u φ) = source - sink`` rearranges to ``residual ≈ 0``.

    A closed budget has ``residual ≈ 0`` within float tolerance.
    """
    res = tendency + flux_divergence
    if source is not None:
        res = res - source
    if sink is not None:
        res = res + sink
    return res.rename("budget_residual")


__all__ = ["budget_residual"]
