# xrtoolz-sklearn — the scikit-learn bridge

```bash
# Pre-PyPI, install from the workspace via git (with its xrcore/pipekit
# base); once published this becomes a plain `pip install xrtoolz-sklearn`.
uv pip install \
  "xrtoolz-sklearn @ git+https://github.com/jejjohnson/xrtoolz@main#subdirectory=packages/xrtoolz-sklearn" \
  "xrtoolz-core @ git+https://github.com/jejjohnson/xrtoolz@main#subdirectory=packages/xrtoolz-core" \
  "pipekit @ git+https://github.com/jejjohnson/pipekit@main#subdirectory=packages/pipekit"
```

`xrsklearn` lets any sklearn-style estimator operate on N-D labeled
data. One marshalling core — **stack → delegate → unstack** — exposed
through three surfaces.

## The lifecycle

sklearn estimators want a 2-D `(n_samples, n_features)` matrix.
`XarrayEstimator` gets there and back:

1. **Stack**: the input is transposed so `sample_dim` leads, and every
   other dim is flattened into a feature `MultiIndex`
   (`(time, lat, lon)` → `(time, lat×lon)`). Dataset inputs
   column-concatenate their data_vars into one feature matrix.
2. **Delegate**: the wrapped estimator's own
   `fit / transform / predict / …` runs on the 2-D view.
3. **Unstack**: outputs are re-labeled. Same-feature-count outputs
   rebuild the original feature dims; reduced outputs (PCA scores,
   cluster labels) come back as `(sample_dim, component)` or
   `(sample_dim,)`.

The fitted wrapper stores the feature-grid metadata, which is what lets
`inverse_transform` rebuild the original `(sample, *feature_dims)`
layout. Fitted-estimator attributes (`components_`,
`cluster_centers_`, …) pass through untouched.

```python
from sklearn.decomposition import PCA
from xrsklearn import XarrayEstimator

wrap = XarrayEstimator(PCA(n_components=3), sample_dim="time")
scores = wrap.fit_transform(da)        # (time, lat, lon) → (time, component)
recon = wrap.inverse_transform(scores)  # back to (time, lat, lon)
```

## NaN policy

Real gridded data has land masks and gaps; sklearn raises on NaN.
`nan_policy=` decides what happens at the 2-D boundary:

| Policy | Behaviour |
|---|---|
| `"propagate"` (default) | hand NaNs to the estimator unchanged — fine for NaN-aware estimators, raises inside sklearn otherwise |
| `"mask"` | drop sample rows containing any NaN before delegating, re-insert NaN rows in the output |
| `"raise"` | fail fast with a labeled error before sklearn sees the data |

## The `.sklearn` accessors

Importing `xrsklearn` registers `da.sklearn` / `ds.sklearn` accessors —
thin sugar that constructs an `XarrayEstimator` and forwards:

```python
import xrsklearn  # registration side effect

scaled = da.sklearn.fit_transform(StandardScaler(), sample_dim="time")
```

Methods that need a *fitted* estimator (`transform`, `predict`,
`score`, `inverse_transform`) accept either a raw fitted sklearn
estimator or a fitted `XarrayEstimator` — but only the wrapper carries
the metadata to rebuild the original feature grid on
`inverse_transform`.

## SklearnOp — the pipeline operator

`SklearnOp` wraps an estimator as an `xrcore.Operator` so fit/transform
steps compose into `pipekit.Sequential` chains next to any other
operator, reading and writing named Dataset variables. For a pipeline
you intend to *reuse* (validation, inference), fit the estimators up
front and let `SklearnOp` apply them with its default
`method="transform"` — training-time statistics are then baked in:

```python
from pipekit import Sequential
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xrsklearn import SklearnOp, XarrayEstimator

scaler = XarrayEstimator(StandardScaler(), sample_dim="time").fit(train["ssh"])
pca = XarrayEstimator(PCA(n_components=3), sample_dim="time").fit(
    scaler.transform(train["ssh"])
)

pipeline = Sequential(
    [
        SklearnOp(scaler, variable="ssh", sample_dim="time"),
        SklearnOp(pca, variable="ssh", output_variable="pcs",
                  sample_dim="time"),
    ]
)

pipeline(test)  # applies the *training-time* scaling and projection
```

`method="fit_transform"` exists for one-shot exploratory runs, but note
that it clones and refits the estimator on **every** invocation — in a
reused pipeline that leaks evaluation statistics into the transform and
feeds downstream steps values normalised against a different
distribution than they were fitted on.

## Why not the split-object pattern?

Stateful xrtoolz operations normally split into `CalculateX` (returns
state) and `ApplyX(state)`. The sklearn bridge is the deliberate
exception: sklearn's own fit/transform API *is* the state contract, and
wrapping it twice would force every estimator through a second,
redundant state object. `XarrayEstimator` keeps sklearn's lifecycle;
`SklearnOp` adapts it to pipelines — and sets `forbid_in_yaml`
(pipekit convention for operators holding live state), because its
config records only the estimator class and constructor parameters:
a fitted model's learned state is not YAML-serializable, so a config
round-trip cannot rebuild a fitted pipeline. Persist fitted estimators
with joblib/pickle instead.

## API reference

- [Utilities (`XarrayEstimator`)](../api/utils.md)
- [Transforms (`SklearnOp`)](../api/transforms.md)
