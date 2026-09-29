# Einx

`geotoolz.einx` wraps [einx](https://github.com/fferflo/einx) — universal
einstein-notation tensor ops — as carrier-aware Operators. einx is a core
dependency: the same notation also powers the internal linear algebra of
the Tier-A primitives (covariance products, matched-filter scoring, PCA
projections, channel-order flips).

The **spatial-survival rule** decides what a pattern does to geospatial
metadata: if the output expression ends in the bare spatial axes
(`... y x`) and neither axis is composed anywhere in the pattern, a
`GeoTensor` input returns a `GeoTensor` (transform/CRS/fill preserved).
Any pattern that consumes, moves, or recomposes a spatial axis returns a
plain `np.ndarray`. `SpatialPool` is the deliberate exception — it
*rescales* the transform to the pooled grid.

**Nodata.** `SpatialPool` and `PerBandReduce` exclude invalid pixels
(non-finite, or equal to the carrier's `fill_value_default` in any band)
with NaN-aware reductions; `SpatialPool` writes its output fill into
blocks with no valid pixel (`NaN` for float results of an integer input,
otherwise the input's fill). `Einx` runs einx on the raw values, and a
value-changing, spatially-surviving output declares an explicit fill
(`NaN` for floats, `False` for booleans) written into the carrier's
invalid pixels. As in georeader, `fill_value_default=0` means 0 is
nodata — set it to `None` or `NaN` when 0 is real data.

All presets broadcast over leading axes, so `(T, C, H, W)` time stacks
work too.

```python
import geotoolz as gz

mean_map = gz.Einx(op="mean", pattern="c y x -> y x")     # GeoTensor -> GeoTensor
band_stats = gz.PerBandReduce(reduce="std")               # GeoTensor -> (C,) ndarray
scores = gz.Einx(op="dot", pattern="band y x, sig band -> sig y x")
coarse = gz.SpatialPool(reduce="mean", factor=4)          # transform rescaled
pooled = gz.Einx(op="mean", pattern="c (y py) (x px) -> c y x", op_kwargs={"py": 2, "px": 2})
```

::: geotoolz.einx.Einx

::: geotoolz.einx.CHWtoHWC

::: geotoolz.einx.HWCtoCHW

::: geotoolz.einx.PerBandReduce

::: geotoolz.einx.SpatialPool

## Pattern analysis

These helpers are pure string processing over pattern text.

::: geotoolz.einx.spatial_survives

::: geotoolz.einx.output_axes
