"""xrops — xarray operators for Earth System Data Cubes.

The operator package of the xrtoolz family (``pip install xrtoolz-ops``).
Each family is a subpackage: :mod:`xrops.atm`, :mod:`xrops.budgets`,
:mod:`xrops.geo`, :mod:`xrops.inference`, :mod:`xrops.interpolate`,
:mod:`xrops.metrics`, :mod:`xrops.ocn`, :mod:`xrops.transforms`,
:mod:`xrops.utils` and :mod:`xrops.viz`. The composition layer
(``Sequential``, ``Graph``, ``Input``, ``Tap``) and ``Signature`` are
:mod:`pipekit`; :class:`Operator` is the xarray-aware base every xrops
operator subclasses (``DataTree`` dispatch on top of ``pipekit.Operator``).
"""

from xrops._src.operator import Operator
from xrops.combinators import ApplyToEach, Augment
from xrops.grad import Difference, Gradient


__version__ = "0.3.0"  # x-release-please-version

__all__ = [
    "ApplyToEach",
    "Augment",
    "Difference",
    "Gradient",
    "Operator",
    "__version__",
]
