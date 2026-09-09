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

### Mixed geometry

`divergence` also accepts a **per-dim sequence** of geometries, one per
entry in `dims`. The common case is a 3-D ocean budget: spherical
horizontal, rectilinear vertical.

```python
div = xrgrad.divergence(
    ds,
    ("flux_e", "flux_n", "flux_w"),
    dims=("lon", "lat", "depth"),
    geometry=("spherical", "spherical", "rectilinear"),
)
```

The `-(F_y \tan\varphi)/R` curvature term is applied when **both** the
longitude and latitude axes are spherical. Marking only one of them (or
marking the vertical axis spherical) raises — the metric couples the
pair, so neither is meaningful alone. Geometry-specific keywords are
routed only to the backends that accept them, so `radius=` may
accompany a mixed call without reaching the vertical axis.

### Temporal coordinates

`datetime64` and `timedelta64` coordinates work anywhere a numeric one
does; they are converted to **seconds**, so `∂c/∂t` comes back per
second. Values are anchored at the first sample before conversion, which
keeps nanosecond steps representable in float64. A uniform time axis
takes the cartesian path; an irregular one (a skipped day, monthly
means) needs `geometry="rectilinear"`. `cftime`/object-dtype calendars
are rejected with an explicit error rather than silently mis-stepped.

All geometries expect the differentiated dimension to carry a **1-D
coordinate**: step sizes come from coordinate values, so a dimension
without one raises `ValueError`, and 2-D (curvilinear) coordinates are
rejected rather than silently producing a nonsense step. `"rectilinear"`
additionally requires the coordinate to be **strictly monotonic** —
repeated or reversing values would divide by a zero coordinate
difference. Strictly descending coordinates are fine, as are scalar
coordinates on axes a given derivative does not use (a `lat` derivative
still works after `da.sel(lon=...)`).

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

On spherical and rectilinear geometries the Laplacian is composed as
$\Delta f = \nabla\cdot\nabla f$, so it inherits the corrections
automatically. Only on Cartesian grids, where there is no curvature term
to inherit, does `laplacian` instead sum direct second-derivative
stencils $\sum_d \partial^2 f/\partial d^2$ — see below.

## Accuracy and stencils

`accuracy=` is finitediffx's accuracy order; `method=` selects
`"central"` (default), `"forward"`, or `"backward"` stencils. Interior
points use the requested central stencil; boundary points fall back to
one-sided differences of the same order. Two practical notes:

- First derivatives of polynomials up to the stencil order are exact on
  interior points at `accuracy=1`.
- `partial` takes an `order=` argument for higher derivatives. `order=2`
  uses a single second-derivative stencil rather than two composed
  passes. It is available on Cartesian grids (and on rectilinear ones
  whose coordinate turns out to be uniform); spherical geometry and
  genuinely non-uniform coordinates raise `NotImplementedError`.
- `laplacian` sums `order=2` stencils **only** for
  `geometry="cartesian"`, so there it is already exact for quadratic
  fields at the default `accuracy=1` and degrades only a single boundary
  ring. Every other geometry — spherical, and *all* rectilinear grids,
  including ones whose coordinate is uniform enough that each partial
  delegates to the Cartesian backend — still composes two
  first-derivative stencils, so pass `accuracy=2` (or higher) there when
  the interior must reproduce quadratic fields exactly.
- `divergence`, `curl`, and `laplacian` accept a per-dim `accuracy`
  tuple as well as a scalar, matching `gradient`. The tuple pairs with
  `dims` in order, which is useful on anisotropic grids.

## Periodic boundaries

By default the first and last points of an axis fall back to one-sided
stencils. On a genuinely periodic axis — global longitude being the
usual case — that leaves a visible seam at the dateline. Pass
`periodic=` to wrap instead, so those points are differentiated against
the opposite edge and are as accurate as the interior:

```python
zeta = xrgrad.curl(
    ds, ("u", "v"), dims=("lon", "lat"),
    geometry="spherical", periodic="lon",
)
```

`partial` takes a bool (it acts on one axis); the multi-dim operators
take a dimension name or a sequence of names, which must be among the
dims being differentiated.

The axis is treated as **endpoint exclusive** — the period is `n * step`,
so the last sample sits one step before the first repeats. This is what
`np.linspace(0, L, n, endpoint=False)` and a `0…355` global longitude
axis give you. A grid that stores *both* ends of the period (numpy's
default `endpoint=True`) repeats a physical location, which makes the
seam point its own neighbour and roughly halves its derivative; drop the
duplicate with `da.isel(x=slice(0, -1))` first.

Which convention a grid uses cannot be read off the coordinate — both
are uniform with the same step — so on the Cartesian path this is only a
`UserWarning`, raised when the field happens to take the same value at
both ends. A genuinely endpoint-exclusive field can do that too (period
3 over `[0, 1, 2]` with values `[0, 1, 0]`), so the warning is advisory
and the computation proceeds. Spherical longitude is the exception: its
period *is* known, so a grid that does not span exactly 360° is a hard
error.

Further restrictions:

- **Spherical longitude** must span the full 360°; a regional grid has
  no wrap-around neighbour and raises.
- **Latitude never wraps** — the poles are a separate problem.
- **Non-uniform rectilinear** axes raise: wrapping the index grid would
  also have to wrap the spacing, which has no single period.

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

## NaN and land masks

A stencil that reads a NaN returns NaN. By default that is exactly what
happens, so each masked cell costs a halo of roughly `accuracy` valid
cells in every differentiated direction — on a masked ocean field, a
strip of water lost all the way around every coastline, silently.

`nan_policy="adaptive"` picks, per point, the widest stencil lying
wholly on finite data: centred where the full support is valid, then
forward, then backward. Masked cells stay NaN.

```python
zeta = xrgrad.curl(
    ds, ("u", "v"), dims=("lon", "lat"),
    geometry="spherical", nan_policy="adaptive",
)
```

Measured against an analytic field with a synthetic coastline
(`docs/design/examples/xrgrad-nan-mask-comparison.py`), at the cells
adjacent to land:

| | cells lost | RMS relative error |
|---|---|---|
| `propagate` (default) | 62% | 5.5e-05 on the survivors |
| fill, differentiate, re-mask | 0% | 6.5e-01 |
| `nan_policy="adaptive"` | 0% | 7.2e-03 (`accuracy=1`), 1.4e-04 (`accuracy=2`) |

Two things to take from that:

- **Do not gap-fill a field to fix its derivative.** Harmonic filling
  solves `∇²u = 0`, so the filled gradient is an artefact of the fill —
  four orders of magnitude worse than losing the cells, and invisible
  because the output looks complete. The `fillnan_*` family in
  `xrtoolz.interpolate` is the right tool when you want a gap-free
  *field*, not for derivative accuracy near a mask.
- **Coastal cells cost one order of accuracy**, since they use a
  one-sided stencil. Raising `accuracy` buys it back, as the table
  shows. Interior cells are bit-identical to `propagate`.
- **Raising `accuracy` never loses cells.** A region narrower than the
  requested stencil — a two-cell channel between islands, say — falls
  back to the widest *lower* accuracy that fits there, so recovery is
  monotone in `accuracy` and only the local error order varies. The same
  descent applies when the whole axis is shorter than the stencil. A
  single isolated cell still has no neighbour to difference against and
  stays NaN.
- **Cost is up to `3 * accuracy` stencil passes** on the requested axes,
  against one for `propagate`. The sweep exits as soon as no valid cell
  is unresolved, so an unmasked field costs a single pass and an ordinary
  coastline costs three; the upper bound is only reached on masks
  fragmented into strips narrower than the requested stencil.

`method=` cannot be combined with `nan_policy="adaptive"` — the fallback
chain is itself method selection — and the option composes with
`periodic=`, so a masked global field can wrap and adapt in one call.

## Grid metrics and constants

`grid_metrics_from_coords` derives the cell-width / face-area Datasets
consumed by `xrtoolz.budgets` from a Dataset's lon/lat (and optional
depth) coordinates. The physical constants `EARTH_RADIUS`, `GRAVITY`,
and `OMEGA` are exported at the top level and replace the equivalent
`metpy.constants` lookups.

## API reference

- [Calculus](../api/calc.md)
