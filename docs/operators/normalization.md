# Train-time vs inference-time normalisation

`geotoolz.normalize` provides GeoTensor-aware Operators for common remote-sensing normalisation workflows.

## Train-time statistics

Fit statistics once on your training data, then freeze them into a fixed-stats operator whose JSON-compatible state round-trips for inference:

```python
import geotoolz as gz

scaler = gz.normalize.StandardScaler().fit(train_scene)
train_scene_normalized = scaler.transform(train_scene)

frozen = gz.normalize.StandardScaler(mean=scaler.mean_, std=scaler.std_)
restored = gz.Operator.from_state(frozen.state)
test_scene_normalized = restored(test_scene)
original = restored.inverse(test_scene_normalized)
```

Learned statistics (`mean_` / `std_`, `median_` / `iqr_`, `vmin_` / `vmax_`) are never part of `get_config()` — only constructor arguments are — so pass them back as constructor arguments to persist them. `fit_on_call=True` fits on the first call instead of an explicit `fit`; in a parallel pipeline (`pipekit.ThreadMap`) call `fit` first so the statistics don't depend on which scene arrives first (see [Fitted operators](concepts.md#fitted-operators-fit-transform)).

## Inference-time scaling

For deployed pipelines, pass cached per-band arrays directly:

```python
import geotoolz as gz

scaler = gz.normalize.StandardScaler(mean=mean, std=std)
normalized = scaler(scene)
```

## Visualisation

For per-scene display, percentile stretching is usually more robust than raw min/max scaling:

```python
import geotoolz as gz

rgb = gz.normalize.HistogramStretch(lower=2, upper=98, out_range=(0, 255))(scene)
```

All statistics are NaN-aware by default, and GeoTensor shape, transform, and CRS are preserved by the operators.
