# Calculus

Finite-difference vector calculus on xarray datasets. Each operator works
on cartesian, rectilinear, or spherical geometries by deriving the grid
metrics from the coordinates. For a scalar $\phi$ and vector field
$\mathbf{u}=(u,v)$:

$$
\nabla\phi=\left(\frac{\partial\phi}{\partial x},\frac{\partial\phi}{\partial y}\right),\quad
\nabla\cdot\mathbf{u}=\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y},\quad
\nabla\times\mathbf{u}=\frac{\partial v}{\partial x}-\frac{\partial u}{\partial y},\quad
\Delta\phi=\nabla\cdot\nabla\phi.
$$

The stencils are geoops's diagnostic difference kernel (`geoops.geom.difference`, geostack OP-O7), so a field gives the same numbers in `xrops.grad` and `geoops.geom`. On a lon/lat grid the metric is a sphere of radius `EARTH_RADIUS` by default; `ellipsoid="WGS84"` on `partial`, `gradient`, `Gradient` and `Difference` uses that ellipsoid's radii, as geoops does on a geographic CRS.

## Operators

::: xrops.grad.Gradient

::: xrops.grad.Difference

## Differential operators

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.grad.partial

::: xrops.grad.gradient

::: xrops.grad.divergence

::: xrops.grad.curl

::: xrops.grad.laplacian

::: xrops.grad.grid_metrics_from_coords

## Physical constants

::: xrops.grad.EARTH_RADIUS

::: xrops.grad.GRAVITY

::: xrops.grad.OMEGA
