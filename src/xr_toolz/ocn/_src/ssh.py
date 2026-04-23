"""Sea-surface height helpers."""

from __future__ import annotations

import xarray as xr


def calculate_ssh_alongtrack(
    ds: xr.Dataset,
    variable: str = "ssh",
    sla: str = "sla_filtered",
    mdt: str = "mdt",
    lwe: str = "lwe",
) -> xr.Dataset:
    """Compose along-track SSH from SLA + MDT minus LWE.

    Equivalent altimetry-convention formula:

    ``ssh = sla_filtered + mdt - lwe``

    Args:
        ds: Dataset containing ``sla``, ``mdt``, ``lwe``.
        variable: Name under which to store the SSH output.
        sla: Sea-level anomaly variable name.
        mdt: Mean dynamic topography variable name.
        lwe: Land-water equivalent correction variable name.

    Returns:
        ``ds`` with ``variable`` added.
    """
    ds = ds.copy()
    ds[variable] = ds[sla] + ds[mdt] - ds[lwe]
    ds[variable].attrs.update(
        units="m",
        standard_name="sea_surface_height",
        long_name="Sea Surface Height",
    )
    return ds


def calculate_ssh_unfiltered(
    ds: xr.Dataset,
    variable: str = "ssh",
    sla: str = "sla_unfiltered",
    mdt: str = "mdt",
    lwe: str = "lwe",
) -> xr.Dataset:
    """Same as :func:`calculate_ssh_alongtrack` but from the unfiltered SLA."""
    return calculate_ssh_alongtrack(ds, variable=variable, sla=sla, mdt=mdt, lwe=lwe)
