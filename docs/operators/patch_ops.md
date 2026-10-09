# Tile → operate → stitch

Run any operator or model tile by tile over a scene too large for memory,
and get one result on the scene's grid. `geotoolz.patch_ops` turns the
[geopatcher](../patcher/index.md) steps into operators, so the whole flow
is one `Sequential`.

Install the `[patch]` extra (see [Install](index.md#install)):

```bash
pip install 'geotoolz[patch]'
```

## Run a pipeline tile by tile

Wrap a geopatcher `SpatialPatcher` in `GridSampler`, map your pipeline
over the chips with `ApplyToChips`, and merge with `MergePatches`.
Overlapping tiles with a tapered window give a result without seams.

```python
import numpy as np
from georeader.geotensor import GeoTensor
from rasterio.transform import from_origin

import geopatcher as gp
import geotoolz as gz
from geotoolz.patch_ops import ApplyToChips, GridSampler, MergePatches

rng: np.random.Generator = np.random.default_rng(0)
scene: GeoTensor = GeoTensor(
    rng.integers(200, 4000, size=(2, 512, 512), dtype=np.uint16),  # (2, 512, 512) uint16 · red, NIR
    transform=from_origin(750_000, 4_350_000, 10, 10), crs="EPSG:32610",
    fill_value_default=0, attrs={"band_names": ["B4", "B8"]},
)
field: gp.RasterField = gp.RasterField(scene)

patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(128, 128)),
    sampler=gp.spatial.sampler.RegularStride(step=(96, 96)),        # 32 px overlap
    window=gp.spatial.window.Hann(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
ndvi: gz.Sequential = gz.DNToReflectance(scale=1e-4) | gz.NDVI(red="B4", nir="B8")
tiled: gz.Sequential = gz.Sequential([
    GridSampler(patcher=patcher),                                   # field → list[Patch], (2, 128, 128) uint16 each
    ApplyToChips(operator=ndvi),                                    # list[Patch] → list[Patch], (128, 128) float64 each
    MergePatches(aggregation=gp.spatial.aggregation.OverlapAdd(), domain=field.domain),
])
result: GeoTensor = tiled(field)                                    # (2, 512, 512) uint16 → (512, 512) float64
assert result.transform == scene.transform
```

`MergePatches` places the chips on the domain's grid and keeps the band
axes they carry, so NDVI's one-band chips merge into an `(H, W)` result.
It needs the output `domain` when you build it, so build the field first.

## Run a model on each tile

`gz.ModelOp` wraps a trained model, here any callable on a plain array.
It returns the model's output as it is; each `Patch` keeps its footprint,
so `MergePatches` can still place it. `TriangularWindow` is a linear
feather ramp that matches `geom.Stitch(blend="feather")`.

```python
import numpy as np
from georeader.geotensor import GeoTensor
from rasterio.transform import from_origin

import geopatcher as gp
import geotoolz as gz
from geotoolz.patch_ops import ApplyToChips, GridSampler, MergePatches, TriangularWindow


def water_model(chip: np.ndarray) -> np.ndarray:                    # (2, h, w) → (1, h, w) float32
    """Stand-in for a trained network: a soft water score from two bands."""
    green, nir = chip.astype(np.float32)
    return (1 / (1 + np.exp((nir - green) / 500)))[None]


rng: np.random.Generator = np.random.default_rng(0)
scene: GeoTensor = GeoTensor(
    rng.integers(200, 4000, size=(2, 384, 384), dtype=np.uint16),  # (2, 384, 384) uint16 · green, NIR
    transform=from_origin(750_000, 4_350_000, 10, 10), crs="EPSG:32610", fill_value_default=0,
)
field: gp.RasterField = gp.RasterField(scene)

patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(128, 128)),
    sampler=gp.spatial.sampler.RegularStride(step=(96, 96)),
    window=TriangularWindow(width=32),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
infer: gz.Sequential = gz.Sequential([
    GridSampler(patcher=patcher),                                   # field → list[Patch], (2, 128, 128) uint16 each
    ApplyToChips(operator=gz.ModelOp(model=water_model)),           # list[Patch] → list[Patch], (1, 128, 128) float32 each
    MergePatches(aggregation=gp.spatial.aggregation.OverlapAdd(), domain=field.domain),
])
water: GeoTensor = infer(field)                                     # (2, 384, 384) uint16 → (1, 384, 384) float64
```

Fit any learned preprocessing (`StandardScaler().fit(...)`) before the
pipeline, so every tile uses the same statistics; see
[Fitted operators](concepts.md#fitted-operators).

## Start from a catalog

`geocatalog.patch.field_for` mosaics catalog rows onto a `GeoSlice` grid
and returns the `RasterField` that `GridSampler` takes. The
[stack quickstart](../index.md#quickstart-catalog-patcher-operators)
runs the whole flow on Planetary Computer Sentinel-2, with signed URLs.

## Sample training chips by label

Random crops over-represent the majority class. `StratifiedSample` and
`BalancedSampler` classify each candidate chip by the label under its
centre pixel and draw per class.

```python
import numpy as np
from georeader.geotensor import GeoTensor
from rasterio.transform import from_origin

import geotoolz as gz
from geotoolz.patch_ops import ApplyToChips, BalancedSampler, StratifiedSample

rng: np.random.Generator = np.random.default_rng(0)
grid: dict = {"transform": from_origin(750_000, 4_350_000, 10, 10), "crs": "EPSG:32610"}
scene: GeoTensor = GeoTensor(rng.random((4, 256, 256)), **grid, fill_value_default=np.nan)  # (4, 256, 256) float64
land_cover: np.ndarray = np.zeros((256, 256), dtype=np.uint8)                              # (256, 256) uint8 · classes 0, 1, 2
land_cover[:, 128:] = 1
land_cover[200:, :] = 2

stratified: StratifiedSample = StratifiedSample(
    labels=land_cover, target_proportions={0: 0.5, 1: 0.3, 2: 0.2}, n_samples=20, size=(32, 32), seed=42,
)
balanced: BalancedSampler = BalancedSampler(labels=land_cover, n_per_class=5, size=(32, 32), seed=42)

train: gz.Sequential = stratified | ApplyToChips(operator=gz.RandomFlip(seed=0))
chips: list = train(scene)                                          # 20 Patch, (4, 32, 32) float64 each
per_class: list = balanced(scene)                                   # 15 Patch, (4, 32, 32) float64 each
```

Both samplers need `labels` on the scene's pixel grid, are reproducible
for a fixed `seed`, and warn when a class has too few positions. They
emit `list[Patch]`, like `GridSampler`, so the per-chip step is the same
for training and inference.

## The pieces

| Operator | Takes → returns | Purpose |
|---|---|---|
| `GridSampler(patcher=patcher)` | field → `list[Patch]` | run a `SpatialPatcher` and materialise its chips |
| `ApplyToChips(operator=op)` | `list[Patch]` → `list[Patch]` | map any operator over each chip's data |
| `MergePatches(aggregation=aggregation, domain=domain)` | `list[Patch]` → field | merge chips onto the domain's grid |
| `TriangularWindow(width)` | window axis | linear feather ramp, `float64` |
| `StratifiedSample(labels=..., target_proportions=..., n_samples=..., size=...)` | scene → `list[Patch]` | chips in target class proportions |
| `BalancedSampler(labels=..., n_per_class=..., size=...)` | scene → `list[Patch]` | the same number of chips per class |

A `Patch` holds `data` (the chip, with a shifted transform), `anchor`
(upper-left pixel) and `indices` (the window it was cut from). The same
classes are importable from `geopatcher.integrations.pipekit`.

## Pitfalls

- **Not YAML-safe.** `GridSampler` and `MergePatches` hold a patcher and
  a domain, so they are `forbid_in_yaml`; rebuild the pipeline in code.
- **Two different stitchers.** `MergePatches` merges patches into a
  field; `gz.Stitch` mosaics georeferenced GeoTensor tiles.
- **Other sampling axes** (along-track, points, temporal) live in
  geopatcher; see [Patcher concepts](../patcher/concepts.md).
