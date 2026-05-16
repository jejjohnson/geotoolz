# Train-time vs inference-time normalisation

`geotoolz.normalize` provides GeoTensor-aware Operators for common remote-sensing normalisation workflows.

## Train-time statistics

Fit statistics once across your training catalogue, persist them as JSON-compatible operator state, and reuse them for inference:

```python
import geotoolz as gz

scaler = gz.normalize.StandardScaler(fit_on_call=True)
train_scene_normalized = scaler(train_scene)
state = scaler.state
restored = gz.Operator.from_state(state)
test_scene_normalized = restored(test_scene)
```

## Inference-time scaling

For deployed pipelines, pass cached per-band arrays directly:

```python
scaler = gz.normalize.StandardScaler(mean=mean, std=std)
normalized = scaler(scene)
```

## Visualisation

For per-scene display, percentile stretching is usually more robust than raw min/max scaling:

```python
rgb = (
    gz.normalize.PercentileClip(lower=2, upper=98)
    | gz.normalize.MinMaxScaler(vmin=0.0, vmax=1.0, out_range=(0, 255))
)(scene)
```

All statistics are NaN-aware by default, and GeoTensor shape, transform, and CRS are preserved by the operators.
