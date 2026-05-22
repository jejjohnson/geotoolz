"""Control-volume integral primitive — V4.2.

Volume-weighted integral of a tracer over a region. Used as the
foundation for tendency / source / sink terms in V4.3 budgets.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr


def control_volume_integral(
    ds: xr.Dataset,
    *,
    variable: str,
    volume_metrics: xr.Dataset,
    region: xr.DataArray | None = None,
    dims: Sequence[str] = ("z", "lat", "lon"),
    cell_volume_var: str = "cell_volume",
) -> xr.DataArray:
    """Volume-weighted integral ``∫ var dV`` over a control volume.

    Args:
        ds: Dataset containing ``variable``.
        variable: Variable to integrate.
        volume_metrics: Dataset carrying ``cell_volume_var`` (m³) on
            the same horizontal / vertical grid as ``ds[variable]``.
        region: Optional boolean mask selecting the integration volume.
            ``True`` cells are kept. ``None`` integrates over the full
            grid.
        dims: Spatial dims to integrate over. Any other dims (e.g.
            ``time``) are preserved on the result.
        cell_volume_var: Name of the volume variable in
            ``volume_metrics``. Default ``"cell_volume"``.

    Returns:
        DataArray with the integrated value, reduced over ``dims``.

    Notes:
        For 2-D budgets without a vertical coordinate, pass
        ``volume_metrics`` whose ``cell_volume`` equals ``cell_area``
        (the V4.4 helper does this automatically when ``depth=None``)
        and set ``dims=("lat", "lon")``.
    """
    if cell_volume_var not in volume_metrics:
        raise ValueError(
            f"volume_metrics is missing variable {cell_volume_var!r}; "
            f"got {tuple(volume_metrics.data_vars)}."
        )
    field = ds[variable]
    cv = volume_metrics[cell_volume_var]
    weighted = field * cv
    if region is not None:
        weighted = weighted.where(region)
    int_dims = [d for d in dims if d in weighted.dims]
    out = weighted.sum(dim=int_dims, skipna=True)
    return out.rename(f"{variable}_volume_integral").astype(np.float64)


__all__ = ["control_volume_integral"]
