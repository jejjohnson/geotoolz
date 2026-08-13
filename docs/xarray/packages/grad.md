# xrtoolz-grad — finite-difference calculus

```bash
# Pre-PyPI, install from the workspace via git — xrgrad has no internal
# deps, so this resolves on its own; once published this becomes a
# plain `pip install xrtoolz-grad`.
uv pip install "xrtoolz-grad @ git+https://github.com/jejjohnson/xrtoolz@main#subdirectory=packages/xrtoolz-grad"
```

`xrgrad` provides partial derivatives and vector-calculus operators
(`∇`, `∇·`, `∇×`, `Δ`) on labeled xarray grids. It is a pure-function
package — no pipekit, no other xrtoolz dependencies — and the only
member of the workspace that carries the
[finitediffx](https://github.com/ASEM000/finitediffx) (JAX-backed)
stencil engine.

## Geometry dispatch

Every operator takes `geometry=`, which selects how coordinate values
become physical distances:

| Geometry | Assumes | Spacing |
|---|---|---|
| `"cartesian"` | uniform grid, coords in metres | single Δ read from the coord, validated uniform (`uniform_rtol`) |
| `"rectilinear"` | non-uniform 1-D coord per dim, in metres | true coordinate differences |
| `"spherical"` | lon/lat in **degrees** | metric factors below |

For the spherical case, derivatives are taken in radians and scaled by
the metric factors

$$
\frac{\partial}{\partial x}
  = \frac{1}{R\cos\varphi}\frac{\partial}{\partial\lambda},
\qquad
\frac{\partial}{\partial y}
  = \frac{1}{R}\frac{\partial}{\partial\varphi},
$$

with $\lambda$ longitude, $\varphi$ latitude, and $R$ the Earth radius
(`xrgrad.EARTH_RADIUS`, 6 371 000 m).

## Curvature corrections

On the sphere, divergence and curl are not plain sums of partials —
the unit vectors themselves rotate with longitude. `xrgrad` adds the
curvature terms so results match the equivalent `metpy.calc` operators
on lon/lat fields:

$$
\nabla\cdot\mathbf{F}
  = \frac{1}{R\cos\varphi}\frac{\partial F_x}{\partial\lambda}
  + \frac{1}{R}\frac{\partial F_y}{\partial\varphi}
  - \frac{F_y\tan\varphi}{R},
$$

$$
\zeta
  = \frac{1}{R\cos\varphi}\frac{\partial v}{\partial\lambda}
  - \frac{1}{R}\frac{\partial u}{\partial\varphi}
  + \frac{u\tan\varphi}{R}.
$$

The Laplacian is composed as $\Delta f = \nabla\cdot\nabla f$, so it
inherits the corrections automatically.

## Accuracy and stencils

`accuracy=` is finitediffx's accuracy order; `method=` selects
`"central"` (default), `"forward"`, or `"backward"` stencils. Interior
points use the requested central stencil; boundary points fall back to
one-sided differences of the same order. Two practical notes:

- First derivatives of polynomials up to the stencil order are exact on
  interior points at `accuracy=1`.
- `laplacian` composes two first-derivative stencils, so use
  `accuracy=2` (or higher) when the interior must reproduce quadratic
  fields exactly.

```python
import numpy as np
import xarray as xr
import xrgrad

x = np.linspace(0.0, 3.0, 4)
X, Y = np.meshgrid(x, x, indexing="xy")
coords = {"y": x, "x": x}
ds = xr.Dataset(
    {
        "u": xr.DataArray(-Y, dims=("y", "x"), coords=coords),
        "v": xr.DataArray(X, dims=("y", "x"), coords=coords),
    }
)

zeta = xrgrad.curl(ds, ("u", "v"), dims=("x", "y"))       # rigid rotation → 2
div = xrgrad.divergence(ds, ("v", "u"), dims=("x", "y"))  # ∇·(x, -y) → 0
```

For a worked spherical example (geostrophic currents and vorticity from
sea-surface height), see the
[quickstart notebook](../notebooks/xrgrad_quickstart.ipynb).

## Grid metrics and constants

`grid_metrics_from_coords` derives the cell-width / face-area Datasets
consumed by `xrtoolz.budgets` from a Dataset's lon/lat (and optional
depth) coordinates. The physical constants `EARTH_RADIUS`, `GRAVITY`,
and `OMEGA` are exported at the top level and replace the equivalent
`metpy.constants` lookups.

## API reference

- [Calculus](../api/calc.md)
