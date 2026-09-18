"""Sentinel-5P TROPOMI CH4 L2 opener.

The product (S5P-L2-SRON-PUM-400F) spreads one retrieval over four NetCDF
groups on a shared ``(time, scanline, ground_pixel)`` swath::

    PRODUCT/                               time, scanline, ground_pixel
    PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/ + layer
    PRODUCT/SUPPORT_DATA/INPUT_DATA/       + layer / level
    PRODUCT/SUPPORT_DATA/GEOLOCATIONS/     + corner

:func:`open_tropomi_ch4_l2` flattens them into one Dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from xrreader.types import (
    CH4_PRIOR_PROFILE,
    COLUMN_AK,
    DRY_AIR_SUBCOLUMNS,
    QA_VALUE,
    SP,
    XCH4,
    XCH4_BC,
    XCH4_PRECISION,
    Variable,
    apply_cf_attrs,
    subset_where,
)


_PRODUCT = "PRODUCT"
_SUPPORT_GROUPS = (
    "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS",
    "PRODUCT/SUPPORT_DATA/INPUT_DATA",
    "PRODUCT/SUPPORT_DATA/GEOLOCATIONS",
)
_REGISTRY_VARIABLES: tuple[Variable, ...] = (
    XCH4,
    XCH4_BC,
    XCH4_PRECISION,
    COLUMN_AK,
    CH4_PRIOR_PROFILE,
    DRY_AIR_SUBCOLUMNS,
    QA_VALUE,
    SP,
)
_COORD_RENAMES = {"latitude": "lat", "longitude": "lon"}
# ``PRODUCT/time`` is "seconds since 2010-01-01 00:00:00 UTC" (PUM §7);
# only needed when the file lacks CF ``units`` and xarray left it numeric.
_TROPOMI_EPOCH = np.datetime64("2010-01-01T00:00:00", "s")


def open_tropomi_ch4_l2(
    path: str | Path,
    *,
    qa_min: float | None = 0.5,
    keep_groups: bool = False,
) -> xr.Dataset:
    """Flatten a TROPOMI CH4 L2 granule into one CF Dataset.

    The ``PRODUCT`` group and its three ``SUPPORT_DATA`` children are read
    with :func:`xarray.open_datatree` and merged on the shared swath dims
    ``(time, scanline, ground_pixel)`` (plus ``layer`` / ``level`` /
    ``corner`` where a variable carries them). Variables are renamed
    through the registry's ``tropomi`` aliases (``methane_mixing_ratio``
    -> ``xch4``, ``..._bias_corrected`` -> ``xch4_bias_corrected``,
    ``..._precision`` -> ``xch4_precision``, ``column_averaging_kernel``,
    ``methane_profile_apriori`` -> ``ch4_profile_apriori``,
    ``dry_air_subcolumns``, ``qa_value``, ``surface_pressure`` -> ``sp``);
    ``pressure_interval`` keeps its product name. ``latitude`` /
    ``longitude`` become the 2-D ``lat`` / ``lon`` coordinates.

    Time is decoded as in the product: the length-1 ``time`` dimension
    keeps the granule reference time (``PRODUCT/time``, seconds since
    2010-01-01) as its index coordinate, and the per-scanline
    ``delta_time`` offset is folded into a ``scanline_time`` coordinate on
    ``(time, scanline)`` (``datetime64``), replacing ``delta_time``. The
    reference index is what multi-granule concatenation and
    ``sel(time=...)`` use; ``scanline_time`` is the measurement time of
    each scanline.

    Args:
        path: Path to the ``S5P_*_L2__CH4____*.nc`` granule.
        qa_min: Pixels with ``qa_value < qa_min`` are set to NaN in every
            data variable (the PUM recommends ``0.5``). ``None`` keeps
            all pixels.
        keep_groups: When ``True`` the flattened Dataset keeps the raw
            product variable names (no registry renaming, no CF attribute
            stamping); the time decoding and QA screening still apply.

    Returns:
        Flat Dataset on ``(time, scanline, ground_pixel[, layer, level,
        corner])`` with ``lon`` / ``lat`` / ``scanline_time`` coordinates.

    Raises:
        KeyError: If the file has no ``PRODUCT`` group.
    """
    tree = xr.open_datatree(path)
    groups = {g.lstrip("/") for g in tree.groups}
    if _PRODUCT not in groups:
        raise KeyError(f"{path!s} has no {_PRODUCT!r} group")
    parts = [tree[_PRODUCT].to_dataset()]
    parts.extend(tree[g].to_dataset() for g in _SUPPORT_GROUPS if g in groups)
    ds = xr.merge(parts, compat="override", join="exact", combine_attrs="override")
    ds.attrs = dict(tree.attrs)
    ds = _decode_scanline_time(ds)
    if not keep_groups:
        ds = _rename_canonical(ds)
    if qa_min is not None:
        ds = subset_where(
            ds, QA_VALUE.for_source("tropomi"), qa_min, np.inf, drop=False
        )
    return ds


def _decode_scanline_time(ds: xr.Dataset) -> xr.Dataset:
    """Replace ``delta_time`` by a ``(time, scanline)`` datetime coordinate."""
    ref = ds["time"]
    if not np.issubdtype(ref.dtype, np.datetime64):
        ref = _TROPOMI_EPOCH + ref.astype("timedelta64[s]")
    delta = ds["delta_time"]
    if np.issubdtype(delta.dtype, np.datetime64):
        scanline_time = delta
    elif np.issubdtype(delta.dtype, np.timedelta64):
        scanline_time = ref + delta
    else:
        scanline_time = ref + delta.astype("timedelta64[ms]")
    scanline_time = scanline_time.rename("scanline_time")
    scanline_time.attrs = {"long_name": "Scanline measurement time (UTC)"}
    return ds.drop_vars("delta_time").assign_coords(
        time=ref, scanline_time=scanline_time
    )


def _rename_canonical(ds: xr.Dataset) -> xr.Dataset:
    """Rename product names to registry names and stamp CF attributes."""
    rename = {v.for_source("tropomi"): v.name for v in _REGISTRY_VARIABLES}
    rename.update(_COORD_RENAMES)
    ds = ds.rename({k: v for k, v in rename.items() if k in ds.variables and k != v})
    promote = [n for n in ("lon", "lat") if n in ds.data_vars]
    if promote:
        ds = ds.set_coords(promote)
    for var in _REGISTRY_VARIABLES:
        if var.name in ds.data_vars:
            ds[var.name] = apply_cf_attrs(ds[var.name], var)
    return ds
