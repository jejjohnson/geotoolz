"""Binning and grid scaffolding.

Uses :func:`scipy.stats.binned_statistic_2d` for binning rather than
``pyinterp.Binning2D``, which avoids a heavy C++ dependency. Grid and
period objects are simple ``@dataclass`` definitions rather than
``odc-geo`` wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import binned_statistic_2d


@dataclass
class Grid:
    """Regular lon/lat grid defined by 1-D coordinate arrays."""

    lon: np.ndarray
    lat: np.ndarray

    @classmethod
    def from_bounds(
        cls,
        lon_bnds: tuple[float, float],
        lat_bnds: tuple[float, float],
        resolution: float,
    ) -> Grid:
        """Build a grid whose cells have width ``resolution`` degrees."""
        lon = np.arange(lon_bnds[0], lon_bnds[1] + resolution / 2, resolution)
        lat = np.arange(lat_bnds[0], lat_bnds[1] + resolution / 2, resolution)
        return cls(lon=lon, lat=lat)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, lon: str = "lon", lat: str = "lat") -> Grid:
        return cls(lon=np.asarray(ds[lon].values), lat=np.asarray(ds[lat].values))

    def coords(self) -> dict[str, np.ndarray]:
        return {"lon": self.lon, "lat": self.lat}

    def bin_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lon_edges, lat_edges)`` for use by ``np.histogram2d``."""
        return _cell_edges(self.lon), _cell_edges(self.lat)


@dataclass
class Period:
    """A pandas-compatible time window."""

    time_min: str
    time_max: str
    freq: str = "1D"

    @property
    def date_range(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.time_min, end=self.time_max, freq=self.freq)


@dataclass
class SpaceTimeGrid:
    """Lon/lat/time grid. Use :meth:`from_bounds` for the common case."""

    lon: np.ndarray
    lat: np.ndarray
    time: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))

    @classmethod
    def from_bounds(
        cls,
        lon_bnds: tuple[float, float],
        lat_bnds: tuple[float, float],
        resolution: float,
        time_min: str,
        time_max: str,
        freq: str = "1D",
    ) -> SpaceTimeGrid:
        grid = Grid.from_bounds(lon_bnds, lat_bnds, resolution)
        period = Period(time_min=time_min, time_max=time_max, freq=freq)
        return cls(lon=grid.lon, lat=grid.lat, time=period.date_range)

    @classmethod
    def from_grid_and_period(cls, grid: Grid, period: Period) -> SpaceTimeGrid:
        return cls(lon=grid.lon, lat=grid.lat, time=period.date_range)

    def coords(self) -> dict[str, np.ndarray | pd.DatetimeIndex]:
        return {"lon": self.lon, "lat": self.lat, "time": self.time}


def bin_2d(
    da: xr.DataArray,
    grid: Grid,
    statistic: str = "mean",
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """Bin a scattered DataArray onto a regular lon/lat grid.

    Args:
        da: Input DataArray whose non-NaN samples will be binned. Must
            carry 1-D ``lon`` and ``lat`` coordinates (typically as
            sibling arrays of the same length as the data).
        grid: Target :class:`Grid`.
        statistic: ``"mean"``, ``"median"``, ``"sum"``, ``"count"``,
            ``"min"``, ``"max"``, ``"std"`` — anything accepted by
            :func:`scipy.stats.binned_statistic_2d`.
        lon: Name of the longitude coordinate on ``da``.
        lat: Name of the latitude coordinate on ``da``.

    Returns:
        DataArray on the grid with ``(lat, lon)`` dims.
    """
    values = np.ravel(da.values)
    lons = np.ravel(np.asarray(da[lon].values))
    lats = np.ravel(np.asarray(da[lat].values))
    finite = np.isfinite(values)

    lon_edges, lat_edges = grid.bin_edges()
    stat, _, _, _ = binned_statistic_2d(
        lons[finite],
        lats[finite],
        values[finite],
        statistic=statistic,
        bins=[lon_edges, lat_edges],
    )
    return xr.DataArray(
        data=stat.T,
        dims=("lat", "lon"),
        coords={"lon": grid.lon, "lat": grid.lat},
        name=da.name,
        attrs=dict(da.attrs),
    )


def histogram_2d(
    da: xr.DataArray,
    grid: Grid,
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """2-D histogram count on ``grid``."""
    return bin_2d(da, grid, statistic="count", lon=lon, lat=lat)


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Turn an array of cell-center coordinates into cell edges."""
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        raise ValueError("Need at least two coordinate values to build cell edges.")
    half = 0.5 * np.diff(centers)
    edges = np.empty(centers.size + 1)
    edges[1:-1] = centers[:-1] + half
    edges[0] = centers[0] - half[0]
    edges[-1] = centers[-1] + half[-1]
    return edges
