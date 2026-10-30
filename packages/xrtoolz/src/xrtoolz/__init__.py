"""xrtoolz — the xarray side of the geostack in one install.

This distribution holds no code. It installs the family, each package
imported under its own name:

- ``xrtoolz-ops`` (``xrops``): operator families on xarray (atm, ocn,
  budgets, geo, interpolate, metrics, transforms, viz);
- ``xrtoolz-products`` (``xrproducts``): readers for CMEMS, CDS, AEMET and
  local L2 products;
- ``xrtoolz-grad`` (``xrgrad``): finite-difference calculus on xarray.

Example:
    >>> import xrops
    >>> import xrproducts
"""

from __future__ import annotations


__version__ = "0.3.0"  # x-release-please-version

__all__ = ["__version__"]
