"""Land / ocean / country masks via ``regionmask``.

Each mask function adds a boolean-like ``int16`` mask as a coordinate
(not a data variable) so that downstream :func:`xarray.Dataset.where`
calls pick it up naturally. The helper :func:`apply_mask` is a thin
wrapper around ``ds.where(mask, drop=drop)``.

Requires :mod:`regionmask`, which itself downloads and caches Natural
Earth shapefiles on first use.
"""

from __future__ import annotations

import numpy as np
import regionmask
import xarray as xr


def add_land_mask(ds: xr.Dataset, name: str = "land_mask") -> xr.Dataset:
    """Attach a 1/0 land mask (Natural Earth 110m) as a coordinate.

    Args:
        ds: Input dataset with ``lon`` and ``lat`` coordinates.
        name: Name of the mask coordinate to add.

    Returns:
        Dataset with a new ``name`` coordinate that is 1 over land and
        0 over ocean.
    """
    regions = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = regions.mask_3D(ds).squeeze(drop=True).astype(np.int16)
    mask.attrs.update(
        standard_name="land_mask",
        long_name="Land Mask",
        source="Natural Earth 110m",
    )
    return _attach_mask(ds, mask, name)


def add_ocean_mask(
    ds: xr.Dataset,
    ocean: str = "global",
    name: str = "ocean_mask",
) -> xr.Dataset:
    """Attach a 1/0 ocean mask as a coordinate.

    Args:
        ds: Input dataset with ``lon`` and ``lat`` coordinates.
        ocean: Ocean basin name (e.g. ``"indian"``, ``"north atlantic"``)
            or ``"global"`` to union all Natural Earth ocean basins.
        name: Name of the mask coordinate to add.

    Returns:
        Dataset with a new ``name`` coordinate that is 1 inside the
        requested ocean region(s) and 0 elsewhere.
    """
    regions = regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50
    mask_3d = regions.mask_3D(ds)

    if ocean == "global":
        mask = mask_3d.any(dim="region").astype(np.int16)
    else:
        selector = [i for i, n in enumerate(mask_3d.names.values) if n == ocean]
        if not selector:
            available = sorted(set(mask_3d.names.values))
            raise ValueError(
                f"Ocean basin {ocean!r} not found. Available: {available}."
            )
        mask = mask_3d.isel(region=selector[0]).astype(np.int16)

    mask.attrs.update(
        standard_name="ocean_mask",
        long_name="Ocean Mask",
        source="Natural Earth ocean_basins_50",
        basin=ocean,
    )
    return _attach_mask(ds, mask, name)


def add_country_mask(
    ds: xr.Dataset,
    country: str,
    name: str = "country_mask",
) -> xr.Dataset:
    """Attach a 1/0 country mask as a coordinate.

    Args:
        ds: Input dataset with ``lon`` and ``lat`` coordinates.
        country: Country name or 3-letter abbreviation as recognised by
            Natural Earth 110m country regions.
        name: Name of the mask coordinate to add.

    Returns:
        Dataset with a new ``name`` coordinate that is 1 inside
        ``country`` and 0 elsewhere.
    """
    regions = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    mask_3d = regions.mask_3D(ds)
    names = list(mask_3d.names.values)
    abbrevs = list(mask_3d.abbrevs.values)

    if country in names:
        idx = names.index(country)
    elif country in abbrevs:
        idx = abbrevs.index(country)
    else:
        raise ValueError(
            f"Country {country!r} not found in Natural Earth 110m regions."
        )

    mask = mask_3d.isel(region=idx).astype(np.int16)
    mask.attrs.update(
        standard_name="country_mask",
        long_name="Country Mask",
        country=country,
        source="Natural Earth countries_110",
    )
    return _attach_mask(ds, mask, name)


def apply_mask(
    ds: xr.Dataset,
    mask: xr.DataArray | str,
    drop: bool = False,
) -> xr.Dataset:
    """Apply a boolean mask to every variable in the dataset.

    Args:
        ds: Input dataset.
        mask: Either a DataArray to mask by, or the name of a
            coordinate / variable in ``ds`` (e.g. ``"ocean_mask"``).
        drop: If ``True``, drop coordinates with no valid values.

    Returns:
        Masked dataset.
    """
    mask_da = ds[mask] if isinstance(mask, str) else mask
    return ds.where(mask_da.astype(bool), drop=drop)


def _attach_mask(ds: xr.Dataset, mask: xr.DataArray, name: str) -> xr.Dataset:
    return ds.assign_coords({name: mask})
