# When to jit

Goal: decide which steps of a pipeline run eagerly on NumPy and which
compile with JAX, and cross between them on purpose.

## The rule

```text
PRE · NumPy, eager, once per file
read COG / NetCDF · reproject · mask clouds
rasterize polygons · split into patches
        │ to_jax()
        ▼
COMPILED · JAX, jit + grad
normalize · augment · model forward
H(x) · averaging kernel · loss · ∇ · optimizer
        │ to_numpy()
        ▼
POST · NumPy, eager, once per result
threshold · vectorize · stitch · metrics · write
```

| Put it inside `jit` when | Keep it NumPy when |
| --- | --- |
| it runs many times on the same shapes | it runs once per file or per result |
| a gradient must flow through it | nothing downstream needs its gradient |
| shapes are fixed or can be padded | shapes depend on the data |
| it is array math (elementwise, einx, linear algebra) | it calls GDAL, rasterio, georeader, scipy, skimage or sklearn |

A carrier crosses with `to_jax()` (needs `jax_enable_x64`: the transform is
float64) and `to_numpy()`. Inside `jit` the transform, the swath lat/lon,
the point positions and the time stamps are leaves, so a new scene of the
same shape reuses the compiled program; dims, coordinates, CRS, fill and
attrs are static, so changing them compiles again.

## Training a methane plume detector (Sentinel-2)

Preprocessing reads and patches eagerly; the training step is compiled.
Every patch keeps its own transform, so the step can use the geometry (here
the pixel size) without a retrace for each patch.

```python
import jax
import jax.numpy as jnp
import numpy as np

import geoarray
from geoarray import GeoArray
from geoarray.coords import SpectralAxis

jax.config.update("jax_enable_x64", True)

# PRE · NumPy: 64 patches of 32 × 32, bands B11 and B12, each with its own grid
rng = np.random.default_rng(0)
origins = np.arange(64) * 640.0
transforms = np.stack(
    [[20.0, 0.0, 600000.0 + x0, 0.0, -20.0, 3550000.0] for x0 in origins]
)                                                    # (P, 6) float64
patches: GeoArray = GeoArray(
    rng.random((64, 2, 32, 32), dtype=np.float32),   # (P, C, H, W) float32
    transforms,
    "EPSG:32613",
    dims=("patch", "band", "y", "x"),
    coords=(SpectralAxis(names=("B11", "B12")),),
)
labels = (rng.random((64, 32, 32)) > 0.9).astype(np.float32)  # (P, H, W) float32


# COMPILED · JAX: a per-pixel logistic detector on the B12 / B11 ratio
def loss(w: jnp.ndarray, batch: GeoArray, y: jnp.ndarray) -> jnp.ndarray:
    def per_patch(p: GeoArray) -> GeoArray:
        ratio = p.sel(band="B12").data / (p.sel(band="B11").data + 1e-6)  # (H, W)
        return p.sel(band="B12").with_data(w[0] * ratio + w[1])
    logits = geoarray.map_over(batch, per_patch, "patch").data  # (P, H, W)
    # binary cross-entropy, averaged over pixels
    return jnp.mean(jnp.logaddexp(0.0, logits) - y * logits)


step = jax.jit(jax.value_and_grad(loss))
on_device: GeoArray = patches.to_jax()               # data and transforms on JAX
w = jnp.zeros(2)                                     # (2,) float64
for _ in range(3):
    value, grad = step(w, on_device, jnp.asarray(labels))  # one trace for every call
    w = w - 0.1 * grad
```

`geoarray.map_over` over a batch dim maps the data and the matching
transform rows together: 64 patches with 64 transforms are one `jax.vmap`,
one trace.

## One 4D-Var cost (along-track SSH)

The observation operator H(x) is `geoarray.interp_at`: the state read at the
observation points. Observations come in sets of different sizes; pad them
to a fixed N with `pad_points`, and mask with `valid`, so every window
reuses one compiled cost and the padding changes neither J nor ∇J.

```python
import jax
import jax.numpy as jnp
import numpy as np

from geoarray import GeoArray, GeoPoints, interp_at, pad_points

jax.config.update("jax_enable_x64", True)

# the state: SSH on a 1/4° grid, (H, W) = (40, 60)
ssh: GeoArray = GeoArray(
    np.zeros((40, 60)),                              # (H, W) float64
    (0.25, 0.0, -30.0, 0.0, -0.25, 50.0),            # lon, lat in degrees
    "EPSG:4326",
    dims=("y", "x"),
).to_jax()

# one pass of along-track SLA: 37 observations, padded to N = 64
rng = np.random.default_rng(0)
track: GeoPoints = GeoPoints(
    rng.normal(0.0, 0.05, 37),                       # (N_obs,) float64, metres
    xy=np.stack([np.linspace(-25.0, -20.0, 37), np.linspace(45.0, 40.0, 37)], -1),
    dims=("obs",),
    crs="EPSG:4326",
    fill_value=np.nan,
)
obs: GeoPoints = pad_points(track, 64).to_jax()      # (64,) float64, valid on 37


@jax.jit
def cost(state: GeoArray, observations: GeoPoints, sigma: float) -> jnp.ndarray:
    # H(x): the state at the observation points
    predicted: GeoPoints = interp_at(state, observations)
    # d = y − H(x), zero on the padded slots
    d = jnp.where(observations.valid, observations.data - predicted.data, 0.0)
    # J = ½ dᵀ R⁻¹ d, R = σ² I
    return 0.5 * (d @ d) / sigma**2


value = cost(ssh, obs, 0.03)                         # () float64
gradient: GeoArray = jax.grad(cost)(ssh, obs, 0.03)  # ∇ₓJ: a GeoArray on the grid
```

## Pitfalls

- **A bare `jax.vmap` maps the first dim only.** Static dims cannot follow
  `in_axes`; map a named dim with `geoarray.map_over`.
- **Host reads raise under a trace.** `g.grid`, `g.to_numpy()` and
  `np.asarray(g)` need concrete values and raise `GeoArrayTraceError` inside
  `jit`; read the transform as `g.transform` there.
- **Changing static fields recompiles.** A new CRS, new band names or new
  attrs compile again; new pixels, transforms and dates do not.
