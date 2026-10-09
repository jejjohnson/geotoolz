# Normalise for training and inference

Fit per-band statistics once on training data, save them as plain
configuration, and apply the same scaling at inference. The contract
behind `fit` / `transform` is in [Fitted operators](concepts.md#fitted-operators).

## Fit on training data, apply at inference

`fit` learns `mean_` and `std_` per band. They are not part of
`get_config()`, so pass them back as constructor arguments to get an
operator whose state is JSON and rebuilds anywhere.

```python
import numpy as np
from georeader.geotensor import GeoTensor
from rasterio.transform import from_origin

import geotoolz as gz

rng: np.random.Generator = np.random.default_rng(0)
grid: dict = {"transform": from_origin(750_000, 4_350_000, 10, 10), "crs": "EPSG:32610", "fill_value_default": np.nan}
train: GeoTensor = GeoTensor(rng.normal(0.2, 0.05, (4, 64, 64)), **grid)  # (4, 64, 64) float64
test: GeoTensor = GeoTensor(rng.normal(0.2, 0.05, (4, 64, 64)), **grid)   # (4, 64, 64) float64

fitted: gz.StandardScaler = gz.normalize.StandardScaler().fit(train)
frozen: gz.StandardScaler = gz.normalize.StandardScaler(mean=fitted.mean_, std=fitted.std_)
state: dict = frozen.state                                                # JSON: class name + per-band mean / std

deployed: gz.Operator = gz.Operator.from_state(state)
normed: GeoTensor = deployed(test)                                        # (4, 64, 64) float64 → (4, 64, 64) float64
restored: GeoTensor = deployed.inverse(normed)                            # (4, 64, 64) float64 → (4, 64, 64) float64
assert np.allclose(restored.values, test.values)
```

`RobustScaler` (`median_` / `iqr_`) and `MinMaxScaler` (`vmin_` /
`vmax_`) follow the same pattern. Statistics skip `NaN` and the carrier's
fill, and outputs keep the input's CRS and transform.

## Learn on the first call

`fit_on_call=True` fits on the first scene the operator sees and reuses
those statistics afterwards. Use it in a notebook; in a parallel
pipeline call `fit` first, so the statistics do not depend on which
scene arrives first.

```python
import numpy as np

import geotoolz as gz

rng: np.random.Generator = np.random.default_rng(0)
first: np.ndarray = rng.normal(0.2, 0.05, (4, 32, 32))   # (4, 32, 32) float64
second: np.ndarray = rng.normal(0.3, 0.05, (4, 32, 32))  # (4, 32, 32) float64

scaler: gz.StandardScaler = gz.normalize.StandardScaler(fit_on_call=True)
a: np.ndarray = scaler(first)                             # (4, 32, 32) float64 → (4, 32, 32) float64 · fits here
b: np.ndarray = scaler(second)                            # (4, 32, 32) float64 → (4, 32, 32) float64 · reuses the fit
assert np.allclose(a.mean(axis=(1, 2)), 0) and not np.allclose(b.mean(axis=(1, 2)), 0)
```

## Stretch for display

For one-off display, a percentile stretch per band handles outliers
better than min / max scaling.

```python
import numpy as np

import geotoolz as gz

rgb: np.ndarray = np.random.default_rng(0).gamma(2.0, 0.05, (3, 64, 64))           # (3, 64, 64) float64
stretched: np.ndarray = gz.normalize.HistogramStretch(lower=2, upper=98)(rgb)       # (3, 64, 64) float64 → (3, 64, 64) float64 · in [0, 1]
display: np.ndarray = gz.normalize.HistogramStretch(out_range=(0, 255))(rgb)        # (3, 64, 64) float64 → (3, 64, 64) float64 · in [0, 255]
```

For a `uint8` image ready to save, use `gz.viz.StretchToUint8`.

## Further reading

- [Normalize reference](api/normalize.md): every scaler, the non-linear
  scalings and histogram matching.
- [Tile → operate → stitch](patch_ops.md): apply one fitted scaler to
  every tile of a large scene.
