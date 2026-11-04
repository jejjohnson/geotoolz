# Compositing

Per-pixel composites over co-registered frame stacks and matched multi-source tuples.

- **Temporal composites:** `MedianComposite`, `MaxNDVIComposite` (NDVI via
  `geoops.indices.ndvi`), `CloudFreeComposite`, `MinCloudComposite`, `BAPComposite`
- **Multi-source fusion:** `StackMatched`, `BlendMatched`
- **Calling convention:** every composite is an N-ary reducer — frames as positional arguments
  (`MedianComposite()(f1, f2, f3)`, so `op(Input("a"), Input("b"))` works in a `Graph`) or one
  sequence; `MedianComposite` / `MaxNDVIComposite` also take one `(T, C, H, W)` stack, the
  cloud-aware composites take `(frame, mask_or_metadata)` pairs, `StackMatched` / `BlendMatched` also
  take one `Mapping`, and `BlendMatched(method="ivw")` takes `(tensor, variance)` pairs
- **Tier-A primitives** (plain `(T, ..., H, W)` numpy stacks, NaN = missing): `median_composite`,
  `mean_composite`, `take_by_spatial_index`, `mask_frames`, `blend_weighted`, `bap_scores` and the
  per-criterion BAP scores `doy_distance`, `doy_score`, `view_angle_score`, `cloud_distance_score`,
  `opacity_score`
- **Nodata-aware reductions:** `reduce`, `argreduce` (a `GeoArray`, `GeoPoints` or `GeoSwath`, any dim)

## Reductions that leave nodata out

`reduce(x, fn, dim)` and `argreduce(x, fn, dim)` are pipekit-array's
`reduce` / `argreduce` with the carrier's nodata read first, by the
stack's rule (`GeoArray.valid_mask`): a value that is non-finite or equals
the fill, every band of a pixel where any band is, and on `GeoPoints` a
sample that is not `valid`, never count. They agree with
`MedianComposite` on a `(time, band, y, x)` stack. A named reduction runs as its
NaN-skipping form, a position with no valid entry is NaN, and the result's
fill is NaN. `pipekit_array.reduce` reduces the raw values, so on a scene
with fill 0 a time mean averages the zeros in. geoarray does not re-export
either (GA-B6). They run on NumPy and under `jax.jit`.

```python
import numpy as np
import pipekit_array as pa
from geoarray import GeoArray

from geoops.compositing import argreduce, reduce

stack: GeoArray = GeoArray(
    np.array([[[1.0]], [[0.0]], [[3.0]]]),             # (T, H, W) float64, 0 = nodata
    (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0),
    "EPSG:32613",
    fill_value=0.0,
    dims=("time", "y", "x"),
)
mean: GeoArray = reduce(stack, "mean", "time")         # (H, W) float64: 2.0
raw: GeoArray = pa.reduce(stack, "mean", "time")       # (H, W) float64: 1.33, fill counted
best = argreduce(stack, "argmax", "time")              # (H, W) int: 2, never the fill
picked = pa.take_along(stack, best, "time")            # (H, W) float64: 3.0
```

::: geoops.compositing
