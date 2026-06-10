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

## Differential operators

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.calc.partial

::: xrtoolz.calc.gradient

::: xrtoolz.calc.divergence

::: xrtoolz.calc.curl

::: xrtoolz.calc.laplacian

::: xrtoolz.calc.grid_metrics_from_coords

## Physical constants

::: xrtoolz.calc.EARTH_RADIUS

::: xrtoolz.calc.GRAVITY

::: xrtoolz.calc.OMEGA
