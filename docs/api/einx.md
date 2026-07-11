# Einx

`geotoolz.einx` wraps [einx](https://github.com/fferflo/einx) — universal
einstein-notation tensor ops — as carrier-aware Operators. Requires the
`[einx]` extra: `pip install 'geotoolz[einx]'`.

The **spatial-survival rule** decides what a pattern does to geospatial
metadata: if the output expression ends in the bare spatial axes
(`... y x`) and neither axis is composed anywhere in the pattern, a
`GeoTensor` input returns a `GeoTensor` (transform/CRS/fill preserved).
Any pattern that consumes, moves, or recomposes a spatial axis returns a
plain `np.ndarray`. `SpatialPool` is the deliberate exception — it
*rescales* the transform to the pooled grid.

```python
import geotoolz as gz

mean_map = gz.Einx(op="mean", pattern="c y x -> y x")     # GeoTensor -> GeoTensor
band_stats = gz.PerBandReduce(reduce="std")               # GeoTensor -> (C,) ndarray
scores = gz.Einx(op="dot", pattern="band y x, sig band -> sig y x")
coarse = gz.SpatialPool(reduce="mean", factor=4)          # transform rescaled
```

::: geotoolz.einx.Einx

::: geotoolz.einx.CHWtoHWC

::: geotoolz.einx.HWCtoCHW

::: geotoolz.einx.PerBandReduce

::: geotoolz.einx.SpatialPool

## Pattern analysis

These helpers are pure string processing — importable without the extra.

::: geotoolz.einx.spatial_survives

::: geotoolz.einx.output_axes
