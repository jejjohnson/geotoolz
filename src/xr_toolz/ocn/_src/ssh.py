"""Sea-surface height helpers."""

from __future__ import annotations

import xarray as xr


def calculate_ssh_alongtrack(
    ds: xr.Dataset,
    variable: str = "ssh",
    sla: str = "sla_filtered",
    mdt: str = "mdt",
    lwe: str | None = "lwe",
) -> xr.Dataset:
    """Compose along-track SSH from SLA + MDT (- LWE, optional).

    Equivalent altimetry-convention formula:

    ``ssh = sla + mdt - lwe``  (when *lwe* is given)

    ``ssh = sla + mdt``        (when *lwe* is ``None``)

    Upstream ``sla_to_ssh`` convention (no LWE correction) is reproduced by
    passing ``lwe=None``::

        calculate_ssh_alongtrack(ds, sla="sla", mdt="mdt", lwe=None)

    When MDT lives in a separate file it must first be regridded onto the
    SLA grid before the datasets can be merged.  The canonical pattern is::

        from xr_toolz.interpolate import regrid_like
        from xr_toolz.ocn import calculate_ssh_alongtrack

        ds_mdt_on_sla = regrid_like(ds_mdt, ds_sla, dims=("lat", "lon"))
        ds = ds_sla.assign(mdt=ds_mdt_on_sla["mdt"])
        ds = calculate_ssh_alongtrack(ds, sla="sla", mdt="mdt", lwe=None)

    Args:
        ds: Dataset containing the ``sla``, ``mdt``, and (optionally) ``lwe``
            variables.
        variable: Name under which to store the SSH output.
        sla: Sea-level anomaly variable name.
        mdt: Mean dynamic topography variable name.
        lwe: Land-water equivalent correction variable name.  Pass ``None``
            to skip the LWE correction (simple ``sla + mdt`` convention).

    Returns:
        ``ds`` with ``variable`` added.
    """
    ds = ds.copy()
    ssh = ds[sla] + ds[mdt]
    if lwe is not None:
        ssh = ssh - ds[lwe]
    ds[variable] = ssh
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
    lwe: str | None = "lwe",
) -> xr.Dataset:
    """Same as :func:`calculate_ssh_alongtrack` but from the unfiltered SLA."""
    return calculate_ssh_alongtrack(ds, variable=variable, sla=sla, mdt=mdt, lwe=lwe)
