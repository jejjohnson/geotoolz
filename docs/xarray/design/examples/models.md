---
status: draft
version: 0.1.0
---

!!! note "Imports in this page are from the original `geo_toolz` layout"
    These design docs were adapted from the `geo_toolz` design study.
    Code snippets use the original feature-based import paths
    (`geo_toolz.<module>`). In `xr_toolz`, the domain-agnostic
    operations live under `xr_toolz.geo.<module>`; physics-specific
    operations live under `xr_toolz.ocn` / `xr_toolz.atm` /
    `xr_toolz.rs`. See `xr_toolz/__init__.py` and
    `xr_toolz/geo/__init__.py` for the current export surface.

# Layer 2 — Model Examples

Graph API, inference, and model comparison. *(P4: bring your own model, P2: progressive disclosure)*

---

## End-to-End Graph (Preprocessing + Evaluation)

### Branching DAG with shared preprocessing, multi-input metrics

```python
from geo_toolz.core import Input, Graph, Sequential
from geo_toolz.validation import ValidateCoords
from geo_toolz.regrid import Regrid
from geo_toolz.detrend import RemoveClimatology
from geo_toolz.metrics import RMSE, PSDScore

# Declare symbolic inputs
raw = Input("satellite_data")
model = Input("model_forecast")
ref = Input("reference")

# Shared preprocessing — P1: Sequential is an Operator
preprocess = Sequential([
    ValidateCoords(),
    Regrid(target_lon, target_lat),
    RemoveClimatology(clim),
])

obs_clean = preprocess(raw)
model_clean = preprocess(model)

# Multi-input evaluation (D3: dual-mode __call__ — symbolic here)
obs_rmse = RMSE(variable="ssh", dims=["time"])(obs_clean, ref)
model_rmse = RMSE(variable="ssh", dims=["time"])(model_clean, ref)
model_psd = PSDScore(variable="ssh", dims=["lat", "lon"])(model_clean, ref)

# Compile into a single callable — P1: Graph is itself an Operator
pipeline = Graph(
    inputs={"satellite_data": raw, "model_forecast": model, "reference": ref},
    outputs={
        "obs_cleaned": obs_clean,
        "model_cleaned": model_clean,
        "obs_rmse": obs_rmse,
        "model_rmse": model_rmse,
        "model_psd": model_psd,
    },
)

results = pipeline(
    satellite_data=ds_satellite,
    model_forecast=ds_model,
    reference=ds_reference,
)
print(pipeline.describe())
```

---

## Inference — sklearn in a Pipeline

### P4: ModelOp turns any model into an Operator

```python
from geo_toolz.core import Sequential
from geo_toolz.inference import SklearnModelOp
import joblib

rf = joblib.load("trained_random_forest.pkl")

# P4: SklearnModelOp wraps the model — same Operator interface
pipeline = Sequential([
    ValidateCoords(),
    Regrid(target_lon=target_lon, target_lat=target_lat),
    RemoveClimatology(clim),
    SklearnModelOp(rf, sample_dim="time"),
])

# Raw satellite data → cleaned → predictions — P3: xarray in, xarray out
ds_pred = pipeline(ds_raw)
```

---

## Inference — JAX/Equinox Model

### JIT-compiled inference as a pipeline step

```python
import equinox as eqx
from geo_toolz.inference import JaxModelOp

model = eqx.tree_deserialise_leaves("ssh_emulator.eqx", model_skeleton)

# P4: JaxModelOp — jit-compiled, same Operator interface (D4: no framework import)
pipeline = Sequential([
    RemoveClimatology(clim),
    JaxModelOp(model, sample_dim="time", jit=True),
])

ds_pred = pipeline(ds_features)
```

---

## Inference — NumPyro Posterior Predictive

### Any callable works via ModelOp

```python
from geo_toolz.inference import ModelOp

def posterior_predict(X):
    return predictive(jr.PRNGKey(0), X)["obs"].mean(axis=0)

# P4: plain callable → Operator
pipeline = Sequential([
    RemoveClimatology(clim),
    ModelOp(posterior_predict, sample_dim="time"),
])

ds_pred = pipeline(ds_features)
```

---

## Model Comparison Graph

### Preprocess → infer (2 models) → evaluate — all in one DAG

```python
from geo_toolz.core import Input, Graph, Sequential
from geo_toolz.inference import SklearnModelOp, JaxModelOp
from geo_toolz.metrics import RMSE, PSDScore

raw = Input("features")
ref = Input("ground_truth")

cleaned = Sequential([ValidateCoords(), Regrid(target_lon, target_lat), RemoveClimatology(clim)])(raw)

# Two competing models — P4: both are Operators
sklearn_pred = SklearnModelOp(fitted_rf, sample_dim="time")(cleaned)
jax_pred = JaxModelOp(neural_net, sample_dim="time", jit=True)(cleaned)

sklearn_rmse = RMSE(variable="ssh", dims=["time"])(sklearn_pred, ref)
jax_rmse = RMSE(variable="ssh", dims=["time"])(jax_pred, ref)
jax_psd = PSDScore(variable="ssh", dims=["lat", "lon"])(jax_pred, ref)

pipeline = Graph(
    inputs={"features": raw, "ground_truth": ref},
    outputs={
        "sklearn_pred": sklearn_pred, "jax_pred": jax_pred,
        "sklearn_rmse": sklearn_rmse, "jax_rmse": jax_rmse, "jax_psd": jax_psd,
    },
)

results = pipeline(features=ds_satellite, ground_truth=ds_reference)
print(f"RF RMSE:  {float(results['sklearn_rmse'].values):.4f}")
print(f"JAX RMSE: {float(results['jax_rmse'].values):.4f}")
```
