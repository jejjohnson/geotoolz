# Train-tile / inference-stitch with `patch_ops`

`geotoolz.patch_ops` is the bridge between the four-axis Patcher
framework ([`geopatcher`](https://github.com/jejjohnson/geotoolz/tree/main/packages/geotoolz-patcher)) and
the operator graph: extract well-behaved chips from a scene, run an
operator per chip, and stitch the outputs back into a full scene — with
the train-time and inference-time data flow expressed as the same
operator graph, just with different endpoints.

Install the optional `[patch]` extra to pull in `geopatcher[pipekit]`:

```bash
pip install 'geotoolz[patch]'
```

The same classes are also importable from
`geopatcher.integrations.pipekit`; the two module paths return the same
objects — use whichever reads better in your code.

## The pieces

| Operator | Signature | Purpose |
|----------|-----------|---------|
| `GridSampler(patcher)` | `Field → list[Patch]` | Drive a `SpatialPatcher` and materialise its chips |
| `ApplyToChips(op)` | `list[Patch] → list[Patch]` | Map any operator over each chip's data |
| `Stitch(aggregation, domain)` | `list[Patch] → field` | Merge chips back into a global field |
| `SpatialTriangular(width)` | window axis | Linear feather ramp matching `geom.Stitch(blend="feather")` |
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
from geotoolz.patch_ops import ApplyToChips, GridSampler, SpatialTriangular, Stitch

field = gp.RasterField(scene)          # scene: GeoTensor or RasterioReader

patcher = gp.SpatialPatcher(
    geometry    = gp.SpatialRectangular(size=(256, 256)),
    sampler     = gp.SpatialRegularStride(step=(192, 192)),   # 64 px overlap
    window      = SpatialTriangular(width=32),                # feather ramp
    aggregation = gp.SpatialOverlapAdd(),
)

pipe = Sequential([
    GridSampler(patcher),
    ApplyToChips(cloud_segmentation_model),   # any Operator, e.g. gz.learn.ModelOp
    Stitch(gp.SpatialOverlapAdd(), domain=field.domain),
])
prediction = pipe(field)
```

Swap `SpatialTriangular` for `gp.SpatialHann` / `gp.SpatialTukey` for
smoother tapers, or use `gp.SpatialBoxcar` with non-overlapping strides
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
  directly: `Sequential([StratifiedSample(...), ApplyToChips(gz.augment.RandomFlip())])`.

Because train-time sampling and inference-time tiling both speak
`list[Patch]`, the per-chip part of the graph (`ApplyToChips(model)`)
is identical in both settings — only the endpoints differ.

## Along-track and point sampling

For altimetry ground tracks, flight lines, and station lists, the
sampling axes live upstream in `geopatcher`:

```python
# Chips centred along a track, resampled to a fixed along-track spacing.
patcher = gp.SpatialPatcher(
    geometry    = gp.SpatialRectangular(size=(64, 64)),
    sampler     = gp.SpatialAlongTrack(track=track_xy, spacing=5_000.0),
    window      = gp.SpatialBoxcar(),
    aggregation = gp.SpatialMean(),
)
chips = list(patcher.split(field))

# Raster values at scattered points — nearest or bilinear.
domain = gp.PointDomain(coords=points_xy, kdtree=tree, interp="bilinear")
values = domain.sample(scene)             # (N,) or (bands, N)
```

For CRS-aware point extraction into a vector cube, see
`geotoolz.geom.RasterToPoints`.

## Further reading

- [geopatcher's docs](https://github.com/jejjohnson/geotoolz/tree/main/packages/geotoolz-patcher) — the
  four axes (Geometry × Sampler × Window × Aggregation), boundary
  policies, streaming aggregation, async splits.
- [Integration with geocatalog & geopatcher](recipes/integration-with-geocatalog-and-geopatcher.md)
  — wiring catalog queries into patched pipelines.
