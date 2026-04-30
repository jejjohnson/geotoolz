"""Unstructured points → gridded value resampling."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d

from xr_toolz.interpolate._src.binning import Grid


def points_to_grid(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    grid: Grid,
    statistic: str = "mean",
) -> xr.DataArray:
    """Bin raw (lon, lat, value) tuples onto ``grid``.

    Thin wrapper around :func:`scipy.stats.binned_statistic_2d` that
    doesn't require constructing a scattered DataArray first.
    """
    finite = np.isfinite(values)
    lon_edges, lat_edges = grid.bin_edges()
    stat, _, _, _ = binned_statistic_2d(
        np.ravel(lons)[finite],
        np.ravel(lats)[finite],
        np.ravel(values)[finite],
        statistic=statistic,
        bins=[lon_edges, lat_edges],
    )
    return xr.DataArray(
        data=stat.T,
        dims=("lat", "lon"),
        coords={"lon": grid.lon, "lat": grid.lat},
    )
