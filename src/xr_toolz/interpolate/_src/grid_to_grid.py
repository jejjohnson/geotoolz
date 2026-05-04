"""Grid-to-grid value resampling.

Deterministic refinement (:func:`refine`), aggregation
(:func:`coarsen`), and target-grid resampling (:func:`regrid_like`)
along one or more dimensions. Learned counterparts
(``Downscale``/``Upscale``) live in :mod:`.downscale`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import xarray as xr


def coarsen(
    ds: xr.Dataset | xr.DataArray,
    factor: dict[str, int],
    method: str = "mean",
    boundary: str = "trim",
) -> xr.Dataset | xr.DataArray:
    """Coarsen ``ds`` along one or more dimensions by integer factors.

    Thin wrapper around ``xr.Dataset.coarsen``.
    """
    coarsened = ds.coarsen(dim=factor, boundary=boundary)
    return getattr(coarsened, method)()


def refine(
    ds: xr.Dataset | xr.DataArray,
    factor: dict[str, int],
    method: str = "linear",
) -> xr.Dataset | xr.DataArray:
    """Refine ``ds`` along one or more dimensions by integer factors.

    Produces a ``factor[dim]``-times-denser grid along each dimension
    via :meth:`xr.Dataset.interp`.
    """
    new_coords: dict[str, Sequence[float]] = {}
    for dim, f in factor.items():
        old = ds[dim].values
        if f <= 0:
            raise ValueError(f"refinement factor for {dim!r} must be positive.")
        new_coords[dim] = np.linspace(old.min(), old.max(), (len(old) - 1) * f + 1)
    return ds.interp(new_coords, method=method)


def regrid_like(
    ds: xr.Dataset | xr.DataArray,
    target: xr.Dataset | xr.DataArray,
    *,
    dims: Iterable[str] = ("lat", "lon"),
    method: str = "linear",
) -> xr.Dataset | xr.DataArray:
    """Resample ``ds`` onto ``target``'s coordinates along ``dims``.

    Thin :meth:`xr.Dataset.interp` wrapper for the common
    "regrid model output to observation grid" step. Coordinates listed
    in ``dims`` must exist on both ``ds`` and ``target``; values along
    other dims pass through.
    """
    dim_list = list(dims)
    missing_target = [d for d in dim_list if d not in target.coords]
    if missing_target:
        raise ValueError(
            f"target is missing requested dims {missing_target!r} as coords; "
            f"got coords {tuple(target.coords)}. Pass `dims=` explicitly to "
            "regrid only the dims that are actually shared."
        )
    missing_source = [d for d in dim_list if d not in ds.coords]
    if missing_source:
        raise ValueError(
            f"input is missing requested dims {missing_source!r} as coords; "
            f"got coords {tuple(ds.coords)}."
        )
    target_coords = {d: target[d] for d in dim_list}
    return ds.interp(target_coords, method=method)
