"""Helpers shared by the scene-style (EMIT / GHGSat) openers."""

from __future__ import annotations

import pandas as pd
import xarray as xr


_COORD_RENAMES = {"latitude": "lat", "longitude": "lon"}


def promote_lonlat(ds: xr.Dataset) -> xr.Dataset:
    """Rename ``latitude``/``longitude`` to ``lat``/``lon`` and make them coords.

    Args:
        ds: Dataset whose geolocation may sit in data variables under
            either the CF long names or the short ``lat``/``lon`` names.

    Returns:
        Dataset with ``lon``/``lat`` as coordinates (when present).
    """
    ds = ds.rename({k: v for k, v in _COORD_RENAMES.items() if k in ds.variables})
    promote = [n for n in ("lon", "lat") if n in ds.data_vars]
    return ds.set_coords(promote) if promote else ds


def attach_scene_time(ds: xr.Dataset) -> xr.Dataset:
    """Give a single-scene Dataset a length-1 ``time`` dimension.

    The scene timestamp is taken from a scalar ``time`` variable when the
    file carries one, otherwise from the ``time_coverage_start`` global
    attribute. Datasets that already have a ``time`` dimension, or expose
    neither, are returned unchanged. The length-1 axis is what lets
    :meth:`~xrreader.LocalL2Source.open` concatenate several scenes
    along ``time``.

    Args:
        ds: Scene dataset on ``(y, x)``.

    Returns:
        Dataset with a ``time`` dimension of length 1 when a timestamp
        could be found.
    """
    if "time" in ds.dims:
        return ds
    if "time" in ds.variables and ds["time"].ndim == 0:
        return ds.set_coords("time").expand_dims("time")
    start = ds.attrs.get("time_coverage_start")
    if start is None:
        return ds
    ts = pd.Timestamp(start)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ds.expand_dims(time=[ts.to_datetime64()])
