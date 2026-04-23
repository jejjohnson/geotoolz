"""Spectral analysis — Power Spectral Density on xarray data."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from typing import Any

import xarray as xr
import xrft


_DEFAULT_PSD_KWARGS: dict[str, Any] = {
    "scaling": "density",
    "detrend": "linear",
    "window": "tukey",
    "nfactor": 2,
    "window_correction": True,
    "true_amplitude": True,
    "truncate": True,
}


def psd_spacetime(
    ds: xr.Dataset,
    variable: str,
    dims: Sequence[str],
    **kwargs: Any,
) -> xr.Dataset:
    """Compute the multi-dimensional power spectrum of ``ds[variable]``.

    Args:
        ds: Input dataset.
        variable: Name of the variable to transform.
        dims: Dimensions over which to compute the spectrum.
        **kwargs: Forwarded to :func:`xrft.power_spectrum`; sensible
            defaults (Tukey window, linear detrend, density scaling) are
            used for keys that are not provided.

    Returns:
        Dataset containing the power spectrum, stored under the same
        ``variable`` name.
    """
    opts = {**_DEFAULT_PSD_KWARGS, **kwargs}
    psd = xrft.power_spectrum(ds[variable], dim=list(dims), **opts)
    return psd.to_dataset(name=variable)


def psd_isotropic(
    ds: xr.Dataset,
    variable: str,
    dims: Sequence[str],
    **kwargs: Any,
) -> xr.Dataset:
    """Compute the isotropic (radially averaged) power spectrum.

    Args:
        ds: Input dataset.
        variable: Name of the variable to transform.
        dims: Spatial dimensions to treat isotropically (e.g.
            ``["lat", "lon"]``).
        **kwargs: Forwarded to :func:`xrft.isotropic_power_spectrum`.

    Returns:
        Dataset containing the isotropic power spectrum, stored under
        the same ``variable`` name.
    """
    opts = {**_DEFAULT_PSD_KWARGS, **kwargs}
    psd = xrft.isotropic_power_spectrum(ds[variable], dim=list(dims), **opts)
    return psd.to_dataset(name=variable)


def cross_spectrum(
    ds: xr.Dataset,
    var_a: str,
    var_b: str,
    dims: Sequence[str],
    **kwargs: Any,
) -> xr.Dataset:
    """Cross-power spectrum of two variables over ``dims``."""
    opts = {**_DEFAULT_PSD_KWARGS, **kwargs}
    cps = xrft.cross_spectrum(ds[var_a], ds[var_b], dim=list(dims), **opts)
    return cps.to_dataset(name=f"{var_a}_{var_b}_cross")


def conditional_average(
    ds: xr.Dataset,
    dims: Sequence[str],
    drop: bool = True,
) -> xr.Dataset:
    """Average over non-frequency dims, keeping only positive support.

    Used to collapse a multidimensional spectrum onto a subset of
    frequency axes without contaminating the sum with the zero/negative
    halves of the remaining axes.
    """
    dims = list(dims)
    remaining = [d for d in ds.dims if d not in dims]
    if not remaining:
        return ds.mean(dim=dims)

    cond = reduce(lambda x, y: x & (ds[y] > 0.0), remaining, ds[remaining[0]] > 0.0)
    return ds.mean(dim=dims).where(cond, drop=drop)
