"""GHGSat CH4 L2 per-plume opener."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from xrreader._src.local.openers._common import attach_scene_time, promote_lonlat
from xrreader.types import CH4_ENHANCEMENT, XCH4, apply_cf_attrs


def open_ghgsat_ch4_l2(path: str | Path) -> xr.Dataset:
    """Open a GHGSat per-plume NetCDF as a flat CF Dataset.

    Releases carry ``xch4`` (ppb) and/or ``ch4_enhancement`` on ``(y, x)``
    with 2-D ``latitude`` / ``longitude``; both are renamed through the
    registry (``ghgsat`` alias, falling back to the canonical name) and
    the geolocation becomes ``lon`` / ``lat`` coordinates.

    Args:
        path: Per-plume NetCDF file.

    Returns:
        Scene Dataset on ``(time, y, x)`` (``time`` of length 1 when the
        file exposes a timestamp) with 2-D ``lon`` / ``lat`` coordinates.

    Raises:
        KeyError: If neither ``xch4`` nor ``ch4_enhancement`` is present.
    """
    ds = xr.open_dataset(path)
    found = False
    for var in (XCH4, CH4_ENHANCEMENT):
        raw = var.for_source("ghgsat")
        if raw not in ds.data_vars:
            continue
        found = True
        if raw != var.name:
            ds = ds.rename({raw: var.name})
        ds[var.name] = apply_cf_attrs(ds[var.name], var)
    if not found:
        raise KeyError(
            f"{path!s} has neither {XCH4.for_source('ghgsat')!r} nor "
            f"{CH4_ENHANCEMENT.for_source('ghgsat')!r}"
        )
    return attach_scene_time(promote_lonlat(ds))
