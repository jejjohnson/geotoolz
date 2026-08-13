# xrtoolz-grad

Finite-difference calculus on labeled xarray grids (import name:
`xrgrad`).

Pure-function API for partial derivatives and vector-calculus operators
(`∇`, `∇·`, `∇×`, `Δ`) with three coordinate geometries:

- `"cartesian"` — uniform spacing in each dimension.
- `"rectilinear"` — non-uniform 1-D coordinate per dimension.
- `"spherical"` — longitude/latitude in degrees, with the metric factors
  `1/(R cos φ)` and `1/R` (and the curvature corrections) applied
  automatically, matching the equivalent `metpy.calc` operators.

Plus `grid_metrics_from_coords` for cell volumes/face areas, and the
physical constants (`EARTH_RADIUS`, `GRAVITY`, `OMEGA`).

The stencils come from [finitediffx](https://github.com/ASEM000/finitediffx);
this package has no dependency on the rest of the xrtoolz stack.

```bash
pip install xrtoolz-grad
```

```python
import xrgrad

zeta = xrgrad.curl(ds, ("u", "v"), dims=("lon", "lat"), geometry="spherical")
```
