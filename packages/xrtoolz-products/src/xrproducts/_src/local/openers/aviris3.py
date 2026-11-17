"""AVIRIS-3 L2B methane-enhancement opener.

The AVIRIS-3 L2B greenhouse-gas collection (ORNL DAAC, doi:10.3334/ORNLDAAC/2358)
ships one orthocorrected COG per quantity and flight scene:
``<flight>_<scene>_L2B_GHG_<ver>_CH4_ORT.tif`` (enhancement, ppm m),
``..._CH4_SNS_ORT.tif`` (sensitivity, unitless) and ``..._CH4_UNC_ORT.tif``
(uncertainty, ppm m), in UTM on WGS 84 with -9999 as no data, and -1 in the
sensitivity and uncertainty where a gas flare or its buffer sits. As for
EMIT, the GeoTIFFs are not read here: convert them to NetCDF first (e.g.
with ``rioxarray.open_rasterio(...).to_netcdf``).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from xrproducts._src.local.openers._common import (
    attach_companions,
    attach_scene_time,
    finish_rasters,
    open_raster,
    pick_raster,
)
from xrproducts.types import CH4_ENHANCEMENT


#: Flare marker of the AVIRIS-3 sensitivity and uncertainty rasters.
FLARE_VALUE = -1.0
# Flight prefix of the AVIRIS-3 file names: AV3YYYYMMDDthhmmss.
_FLIGHT = re.compile(r"AV3(\d{8})t(\d{6})")


def open_av3_ch4_l2b(
    path: str | Path,
    *,
    sensitivity_path: str | Path | None = None,
    uncertainty_path: str | Path | None = None,
) -> xr.Dataset:
    """Open an AVIRIS-3 L2B CH4 enhancement scene as a flat CF Dataset.

    Returns the variables, dims and units of
    :func:`~xrproducts.open_emit_ch4_l2b`: ``ch4_enhancement`` (ppm m, as
    stored, never divided by the sensitivity) and, from their own files,
    ``ch4_sensitivity`` (unitless) and ``ch4_uncertainty`` (ppm m,
    ``uncertainty_of = "truth"``, unverified as for EMIT) on one
    ``(time, y, x)`` grid. Each file holds the registry name or a single
    ``(y, x)`` variable (a converted GeoTIFF band); a length-1 ``band``
    axis is squeezed out. The companions must have the enhancement's
    ``(y, x)`` shape and the same ``lon`` / ``lat`` / ``x`` / ``y`` to
    1e-6 where both carry them; nothing is resampled. The scenes are
    orthocorrected already, so there is no lookup table.

    All three rasters are floats with -9999 as ``NaN``
    (``source_fill_value``); the companions' flare marker -1 is ``NaN``
    too (``flare_value``). The scene time is the file's own (a scalar
    ``time`` or ``time_coverage_start``) or else the flight prefix of the
    file name (``AV3YYYYMMDDthhmmss``, UTC).

    Args:
        path: NetCDF enhancement raster (from ``*_CH4_ORT.tif``).
        sensitivity_path: Optional sensitivity raster (``*_CH4_SNS_ORT``).
        uncertainty_path: Optional uncertainty raster (``*_CH4_UNC_ORT``).

    Returns:
        Scene Dataset on ``(time, y, x)`` (``time`` of length 1 when a
        timestamp could be found).

    Raises:
        KeyError: If a file has neither the registry name nor a single
            ``(y, x)`` variable.
        ProductGridError: If a companion is not on the enhancement's grid.

    Example:
        >>> prefix = "AV320231006t190153_020_L2B_GHG_6395d427"
        >>> scene = open_av3_ch4_l2b(  # doctest: +SKIP
        ...     f"{prefix}_CH4_ORT.nc",
        ...     sensitivity_path=f"{prefix}_CH4_SNS_ORT.nc",
        ... )
        >>> sorted(scene.data_vars)  # doctest: +SKIP
        ['ch4_enhancement', 'ch4_sensitivity']
    """
    ds = open_raster(path)
    name = pick_raster(ds, (CH4_ENHANCEMENT.for_source("av3"),), path)
    ds = ds.rename({name: CH4_ENHANCEMENT.name}) if name != CH4_ENHANCEMENT.name else ds
    ds = attach_companions(
        ds, sensitivity_path=sensitivity_path, uncertainty_path=uncertainty_path
    )
    ds = finish_rasters(ds, flare_value=FLARE_VALUE)
    ds = attach_scene_time(ds)
    if "time" in ds.dims:
        return ds
    return _attach_flight_time(ds, Path(path).name)


def _attach_flight_time(ds: xr.Dataset, filename: str) -> xr.Dataset:
    """Length-1 ``time`` from the ``AV3YYYYMMDDthhmmss`` file-name prefix."""
    match = _FLIGHT.search(filename)
    if match is None:
        return ds
    stamp = pd.to_datetime(match.group(1) + match.group(2), format="%Y%m%d%H%M%S")
    return ds.expand_dims(time=[np.datetime64(stamp, "ns")])
