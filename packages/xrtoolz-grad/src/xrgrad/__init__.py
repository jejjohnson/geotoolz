"""Finite-difference calculus operators for xarray data.

Pure-function API for partial derivatives and vector-calculus operators
(``∇``, ``∇·``, ``∇×``, ``Δ``) on three coordinate geometries:

- ``"cartesian"`` — uniform spacing in each dimension.
- ``"rectilinear"`` — non-uniform 1-D coordinate per dimension.
- ``"spherical"`` — longitude/latitude in degrees, with the metric
  factors ``1/(R cos φ)`` and ``1/R`` applied automatically.

The underlying stencils come from :mod:`finitediffx`. Constants used by
the spherical metric and downstream physics live in
:mod:`xrgrad._src.constants`.
"""

from xrgrad._src.constants import EARTH_RADIUS, GRAVITY, OMEGA
from xrgrad._src.grid_metrics import grid_metrics_from_coords
from xrgrad._src.operators import (
    curl,
    divergence,
    gradient,
    laplacian,
    partial,
)


__version__ = "0.1.0"

__all__ = [
    "EARTH_RADIUS",
    "GRAVITY",
    "OMEGA",
    "__version__",
    "curl",
    "divergence",
    "gradient",
    "grid_metrics_from_coords",
    "laplacian",
    "partial",
]
