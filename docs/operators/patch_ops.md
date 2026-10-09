# Train-tile / inference-stitch with `patch_ops`

`geotoolz.patch_ops` is the bridge between the four-axis Patcher
framework ([`geopatcher`](https://github.com/jejjohnson/geotoolz/tree/main/packages/geotoolz-patcher)) and
the operator graph: extract well-behaved chips from a scene, run an
operator per chip, and stitch the outputs back into a full scene — with
the train-time and inference-time data flow expressed as the same
operator graph, just with different endpoints.

Install the optional `[patch]` extra to pull in `geotoolz-patcher[pipekit]`:

```bash
pip install 'geotoolz[patch]'
```

`GridSampler`, `ApplyToChips` and `MergePatches` are re-exported from
`geopatcher.integrations.pipekit` — the two module paths return the same
class objects. `MergePatches` merges patches into a field;
`geotoolz.geom.Stitch` (also top-level `geotoolz.Stitch`) mosaics
georeferenced GeoTensor tiles.
`GridSampler` and `MergePatches` hold runtime objects (a patcher, a
domain), so both are `forbid_in_yaml`; their `get_config()` is a debug
record, not a replay recipe.

## The pieces

| Operator | Signature | Purpose |
|----------|-----------|---------|
| `GridSampler(patcher=patcher)` | `Field → list[Patch]` | Drive a `SpatialPatcher` and materialise its chips |
| `ApplyToChips(operator=op)` | `list[Patch] → list[Patch]` | Map any operator over each chip's data |
| `MergePatches(aggregation=aggregation, domain)` | `list[Patch] → field` | Merge chips back into a global field |
| `TriangularWindow(width)` | window axis | Linear feather ramp (`float64`) matching `geom.Stitch(blend="feather")` |
| `StratifiedSample(...)` | `scene → list[Patch]` | Chips with class proportions matching a target distribution |
| `BalancedSampler(...)` | `scene → list[Patch]` | Exactly N chips per class label |

`Patch` is geopatcher's chip carrier: `data` (the chip, a `GeoTensor`
with a correctly shifted transform), `anchor` (upper-left pixel), and
`indices` (the `rasterio` window it was cut from).

## Inference: tile → model → feather-stitched scene

Overlapping tiles plus a tapered window and overlap-add aggregation
give seam-free full-scene predictions:

```python
import geopatcher as gp
from geotoolz import Sequential
from geotoolz.patch_ops import ApplyToChips, GridSampler, MergePatches, TriangularWindow

field = gp.RasterField(scene)          # scene: GeoTensor or RasterioReader

patcher = gp.SpatialPatcher(
    geometry    = gp.spatial.geometry.Rectangular(size=(256, 256)),
    sampler     = gp.spatial.sampler.RegularStride(step=(192, 192)),   # 64 px overlap
    window      = TriangularWindow(width=32),                 # feather ramp
    aggregation = gp.spatial.aggregation.OverlapAdd(),
)

pipe = Sequential([
    GridSampler(patcher=patcher),
    ApplyToChips(operator=cloud_segmentation_model),   # any Operator, e.g. gz.learn.ModelOp
    MergePatches(aggregation=gp.spatial.aggregation.OverlapAdd(), domain=field.domain),
])
prediction = pipe(field)       # GeoTensor on the scene's grid, with the model's band axes
```

Swap `TriangularWindow` for `gp.spatial.window.Hann` / `gp.spatial.window.Tukey` for
smoother tapers, or use `gp.spatial.window.Boxcar` with non-overlapping strides
for exact tiling.

## Training: label-aware chip sampling

Random crops over-represent the majority class. The label-aware
samplers classify each candidate chip by the label under its **centre
pixel** and draw within each class:

```python
from geotoolz.patch_ops import BalancedSampler, StratifiedSample

# Class proportions matching a target distribution. The total is split
# across classes with the largest-remainder method, so the realised
# counts always sum to n_samples (when every class has enough chips).
sampler = StratifiedSample(
    labels=land_cover,                       # single-band GeoTensor / (H, W) array
    target_proportions={0: 0.5, 1: 0.3, 2: 0.2},
    n_samples=500,
    size=(128, 128),
    seed=42,
)
train_patches = sampler(scene)

# Or: exactly N chips per class.
sampler = BalancedSampler(labels=land_cover, n_per_class=50, size=(128, 128), seed=42)
train_patches = sampler(scene)
```

Both samplers:

- require `labels` to share the scene's pixel grid (chips are cut from
  the scene at the anchors chosen on the label raster);
- warn and return fewer chips when a class has fewer candidate
  positions than requested;
- are reproducible for a fixed `seed`;
- emit `list[Patch]`, so augmentation or feature extraction composes
  directly: `Sequential([StratifiedSample(...), ApplyToChips(operator=gz.augment.RandomFlip())])`.

Because train-time sampling and inference-time tiling both speak
`list[Patch]`, the per-chip part of the graph (`ApplyToChips(operator=model)`)
is identical in both settings — only the endpoints differ.

## Along-track and point sampling

For altimetry ground tracks, flight lines, and station lists, the
sampling axes live upstream in `geopatcher`:

```python
# Chips centred along a track, resampled to a fixed along-track spacing.
patcher = gp.SpatialPatcher(
    geometry    = gp.spatial.geometry.Rectangular(size=(64, 64)),
    sampler     = gp.spatial.sampler.AlongTrack(track=track_xy, spacing=5_000.0),
    window      = gp.spatial.window.Boxcar(),
    aggregation = gp.spatial.aggregation.Mean(),
)
chips = list(patcher.split(field))

# Raster values at scattered points — nearest or bilinear.
domain = gp.fields.PointDomain(coords=points_xy, kdtree=tree, interp="bilinear")
values = domain.sample(scene)             # (N,) or (bands, N)
```

For CRS-aware point extraction into a vector cube, see
`geotoolz.geom.coregister.RasterToPoints`.

## Further reading

- [geopatcher's docs](https://github.com/jejjohnson/geotoolz/tree/main/packages/geotoolz-patcher) — the
  four axes (Geometry × Sampler × Window × Aggregation), boundary
  policies, streaming aggregation, async splits.
- [Integration with geocatalog & geopatcher](how-to/integration-with-geocatalog-and-geopatcher.md)
  — wiring catalog queries into patched pipelines.
