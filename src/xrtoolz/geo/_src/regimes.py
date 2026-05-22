"""Canonical geographic and data-driven regime masks."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import regionmask
import xarray as xr
from shapely.geometry import MultiPolygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


KM_PER_DEGREE_AT_EQUATOR = 111.0


@lru_cache
def coastal_regions(*, distance_km: float = 200.0) -> regionmask.Regions:
    """Return a coarse two-region coastal/open-ocean partition.

    Built from a hand-rolled bounding-box approximation of the major
    land masses (see :func:`_coarse_land_polygons`). The geometry is
    fixed; ``distance_km`` controls the buffer width applied to the
    landmasses to delimit the coastal band.
    """
    if distance_km < 0:
        raise ValueError("distance_km must be non-negative.")

    distance_deg = distance_km / KM_PER_DEGREE_AT_EQUATOR
    world = box(-180.0, -90.0, 180.0, 90.0)
    coastal = (
        unary_union(_coarse_land_polygons()).buffer(distance_deg).intersection(world)
    )
    open_ocean = world.difference(coastal)
    return regionmask.Regions(
        [coastal, open_ocean],
        names=["coastal", "open_ocean"],
        abbrevs=["coast", "open"],
        name=f"coastal_{distance_km:g}km",
    )


def equatorial_regions(*, lat_threshold: float = 5.0) -> regionmask.Regions:
    """Return equatorial and extra-tropical latitude bands."""
    if not 0.0 < lat_threshold < 90.0:
        raise ValueError("lat_threshold must be between 0 and 90 degrees.")

    equatorial = box(-180.0, -lat_threshold, 180.0, lat_threshold)
    extratropical = MultiPolygon(
        [
            box(-180.0, -90.0, 180.0, -lat_threshold),
            box(-180.0, lat_threshold, 180.0, 90.0),
        ]
    )
    return regionmask.Regions(
        [equatorial, extratropical],
        names=["equatorial", "extratropical"],
        abbrevs=["eq", "extra"],
        name=f"equatorial_{lat_threshold:g}deg",
    )


def eddy_regions(
    ds: xr.Dataset,
    *,
    var: str,
    threshold: float | None = None,
    window: tuple[int, int] = (5, 5),
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """Return a two-class mask from local rolling variance of ``var``."""
    da = ds[var]
    dims = (lat, lon)
    missing = [dim for dim in dims if dim not in da.dims]
    if missing:
        raise ValueError(f"eddy_regions variable {var!r} is missing dims {missing}.")
    local_var = da.rolling({lat: window[0], lon: window[1]}, center=True).var()
    cutoff = float(local_var.median(skipna=True)) if threshold is None else threshold
    mask = xr.where(local_var >= cutoff, np.int64(1), np.int64(0))
    mask = mask.where(local_var.notnull())
    mask.name = "eddy_region"
    mask.attrs.update(
        long_name="Eddy variance regime",
        threshold=cutoff,
        high_variance_label=1,
        low_variance_label=0,
    )
    return mask


def _coarse_land_polygons() -> list[BaseGeometry]:
    """Approximate major land masses for offline coastal masks."""
    return [
        # North America
        box(-168.0, 7.0, -52.0, 72.0),
        # South America
        box(-82.0, -56.0, -34.0, 13.0),
        # Africa
        box(-18.0, -35.0, 52.0, 38.0),
        # Europe and Asia
        box(-11.0, 35.0, 180.0, 72.0),
        # Southeast Asia and Indonesia
        box(95.0, -11.0, 154.0, 8.0),
        # Australia
        box(112.0, -45.0, 154.0, -10.0),
        # New Zealand
        box(166.0, -48.0, 179.0, -34.0),
        # Arctic high-latitude land/ice
        box(-180.0, 60.0, 180.0, 83.0),
        # Antarctica
        box(-180.0, -90.0, 180.0, -60.0),
    ]


__all__ = ["coastal_regions", "eddy_regions", "equatorial_regions"]
