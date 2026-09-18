"""Spatial, temporal, and variable subsetting.

``subset_bbox``, ``subset_where`` and ``subset_time`` are defined in
:mod:`xrreader.types` (pure xarray, shared with the archive readers) and
re-exported here unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import regionmask
import xarray as xr
from xrreader.types import subset_bbox, subset_time, subset_where

from xrtoolz.geo._src.regions import RegionSpec, resolve_region


__all__ = [
    "select_variables",
    "subset_bbox",
    "subset_time",
    "subset_to_region",
    "subset_where",
]


def subset_to_region(
    ds: xr.Dataset,
    region: str | RegionSpec | regionmask.Regions,
    *,
    lon: str = "lon",
    lat: str = "lat",
    validate: bool = True,
) -> xr.Dataset:
    """Subset a Dataset to a named, custom, or polygon region."""
    import warnings

    if isinstance(region, str):
        region = resolve_region(region)
    if isinstance(region, RegionSpec):
        region = region.regions

    # regionmask emits a FutureWarning about the default mask method;
    # pin it explicitly so the choice is stable across regionmask
    # versions and the warning doesn't leak to callers.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        mask = region.mask(ds[lon], ds[lat], method="shapely").notnull()
    if validate and not bool(mask.any().item()):
        raise ValueError(
            "Region does not overlap dataset coordinates. "
            "Pass validate=False to allow empty results."
        )
    return ds.where(mask, drop=True)


def select_variables(
    ds: xr.Dataset,
    variables: str | Sequence[str],
) -> xr.Dataset:
    """Return a dataset with only the requested variables.

    Args:
        ds: Input dataset.
        variables: A single variable name or a sequence of names.

    Returns:
        Dataset restricted to ``variables``.
    """
    names = [variables] if isinstance(variables, str) else list(variables)
    return ds.drop_vars([name for name in ds.data_vars if name not in names])
