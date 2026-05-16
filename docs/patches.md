# Train-tile / inference-stitch workflow

`geotoolz.patches` provides first-class operators for the common remote-sensing
ML loop: cut scenes into model-sized patches, sample training examples, and
stitch patch predictions back onto the original grid.

## Training patches

```python
import numpy as np
import geotoolz as gz

patches = gz.patches.ExtractPatches(
    size=(256, 256),
    stride=(128, 128),
    nan_cutoff=0.1,
    drop_incomplete=True,
)(scene)
```

Use `RandomCrop`, `StratifiedSample`, and `BalancedSampler` when training needs
stochastic crops or class-aware sampling. All random samplers accept `seed`; a
per-call `seed=` overrides the constructor seed for reproducible experiments.

## Inference stitching

```python
tiles = gz.patches.ExtractPatches(size=(256, 256), stride=(192, 192))(scene)
preds = [model(tile) for tile in tiles]

result = gz.patches.StitchPatches(
    target_shape=scene.shape[-2:],
    target_transform=scene.transform,
    target_crs=str(scene.crs),
    blend="feather",
    feather_width=32,
)(preds)
```

Patch transforms and CRS are preserved, so stitched outputs land exactly on the
target scene grid. `blend="average"`, `"feather"`, `"max"`, and `"first"` are
available for overlapping predictions.
