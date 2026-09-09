r"""Finite-difference calculus on labeled xarray grids.

Pure-function API for partial derivatives and vector-calculus operators
(``∇``, ``∇·``, ``∇×``, ``Δ``) on three coordinate geometries, selected
by the ``geometry=`` keyword every operator shares:

- ``"cartesian"`` — uniform spacing in each dimension; the spacing is
  read from the dimension coordinate and validated as uniform.
- ``"rectilinear"`` — non-uniform 1-D coordinate per dimension;
  differences are taken against the true coordinate values.
- ``"spherical"`` — longitude/latitude in degrees. Derivatives are
  taken in radians and scaled by the metric factors

  .. math::

      \frac{\partial}{\partial x} =
      \frac{1}{R\cos\varphi} \frac{\partial}{\partial \lambda},
      \qquad
      \frac{\partial}{\partial y} =
      \frac{1}{R} \frac{\partial}{\partial \varphi},

  and the vector operators add the curvature corrections
  (``∓ tan φ / R`` terms) so results match the equivalent
  ``metpy.calc`` operators on lon/lat fields.

The underlying stencils come from :mod:`finitediffx` (the sole carrier
of the JAX stack in the xrtoolz workspace). Physical constants used by
the spherical metric and downstream physics are exported as
:data:`EARTH_RADIUS`, :data:`GRAVITY`, and :data:`OMEGA`.

**NaN and land masks.** By default a stencil that reads a NaN returns
NaN, so every masked cell costs a halo of roughly ``accuracy`` valid
cells in each differentiated direction — on an ocean field that is a
strip of water lost around every coastline. Pass
``nan_policy="adaptive"`` to any operator to select, per point, the
widest stencil that lies wholly on finite data; the recovered cells
carry one order lower accuracy and masked cells stay NaN. See
``docs/design/xrgrad-nan-masks.md`` for the measurements behind that
default.

Example:
    Relative vorticity of a rigid-rotation velocity field:

    ```pycon
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrgrad
    >>> x = np.linspace(0.0, 3.0, 4)
    >>> X, Y = np.meshgrid(x, x, indexing="xy")
    >>> coords = {"y": x, "x": x}
    >>> ds = xr.Dataset(
    ...     {
    ...         "u": xr.DataArray(-Y, dims=("y", "x"), coords=coords),
    ...         "v": xr.DataArray(X, dims=("y", "x"), coords=coords),
    ...     }
    ... )
    >>> zeta = xrgrad.curl(ds, ("u", "v"), dims=("x", "y"))
    >>> np.unique(zeta.values)
    array([2.])

    ```
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


__version__ = "0.0.2"  # x-release-please-version

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
