"""xrops — xarray operators for Earth System Data Cubes.

The operator package of the xrtoolz family (``pip install xrtoolz-ops``).
Each family is a subpackage: :mod:`xrops.atm`, :mod:`xrops.budgets`,
:mod:`xrops.geo`, :mod:`xrops.inference`, :mod:`xrops.interpolate`,
:mod:`xrops.metrics`, :mod:`xrops.ocn`, :mod:`xrops.transforms`,
:mod:`xrops.utils` and :mod:`xrops.viz`. The composition layer
(``Sequential``, ``Graph``, ``Input``, ``Tap``) is :mod:`pipekit`, and the
xarray ``Operator`` base and ``Signature`` are :mod:`xrcore`.
"""

from xrops.combinators import ApplyToEach, Augment


__version__ = "0.0.0"  # x-release-please-version

__all__ = [
    "ApplyToEach",
    "Augment",
    "__version__",
]
