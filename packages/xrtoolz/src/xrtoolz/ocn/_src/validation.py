"""Ocean-specific variable validation (attribute harmonization)."""

from __future__ import annotations

import xarray as xr


def validate_ssh(ds: xr.Dataset, variable: str = "ssh") -> xr.Dataset:
    """Attach CF-style attrs to an SSH variable."""
    ds = ds.copy()
    ds[variable] = ds[variable].assign_attrs(
        units="m",
        standard_name="sea_surface_height",
        long_name="Sea Surface Height",
    )
    return ds


def validate_velocity(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Attach CF-style attrs to zonal / meridional velocity variables."""
    ds = ds.copy()
    ds[u] = ds[u].assign_attrs(
        units="m s-1",
        standard_name="sea_water_x_velocity",
        long_name="Zonal Velocity",
    )
    ds[v] = ds[v].assign_attrs(
        units="m s-1",
        standard_name="sea_water_y_velocity",
        long_name="Meridional Velocity",
    )
    return ds
