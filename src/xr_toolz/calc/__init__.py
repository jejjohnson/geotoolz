"""Finite-difference calculus operators for xarray data.

Pure-function API for partial derivatives and vector-calculus operators
(``∇``, ``∇·``, ``∇×``, ``Δ``) on three coordinate geometries:

- ``"cartesian"`` — uniform spacing in each dimension.
- ``"rectilinear"`` — non-uniform 1-D coordinate per dimension.
- ``"spherical"`` — longitude/latitude in degrees, with the metric
  factors ``1/(R cos φ)`` and ``1/R`` applied automatically.

The underlying stencils come from :mod:`finitediffx`. Constants used by
the spherical metric and downstream physics live in
:mod:`xr_toolz.calc._src.constants`.
"""

from xr_toolz.calc._src.constants import EARTH_RADIUS, GRAVITY, OMEGA
from xr_toolz.calc._src.operators import (
    curl,
    divergence,
    gradient,
    laplacian,
    partial,
)


__all__ = [
    "EARTH_RADIUS",
    "GRAVITY",
    "OMEGA",
    "curl",
    "divergence",
    "gradient",
    "laplacian",
    "partial",
]
