"""xrtoolz — the xarray side of the geostack in one install.

This distribution holds no code. It installs the family, each package
imported under its own name:

- ``xrtoolz-ops`` (``xrops``): operator families on xarray (atm, ocn,
  budgets, geo, interpolate, metrics, transforms, viz);
- ``xrtoolz-reader`` (``xrreader``): readers for CMEMS, CDS, AEMET and
  local L2 products;
- ``xrtoolz-core`` (``xrcore``): the xarray ``Operator`` base and
  ``Signature``;
- ``xrtoolz-grad`` (``xrgrad``): finite-difference calculus on xarray;
- ``xrtoolz-einx`` (``xreinx``): the einx bridge;
- ``xrtoolz-sklearn`` (``xrsklearn``): the scikit-learn bridge.

Example:
    >>> import xrops
    >>> import xrreader
"""

from __future__ import annotations


__version__ = "0.0.6"  # x-release-please-version

__all__ = ["__version__"]
