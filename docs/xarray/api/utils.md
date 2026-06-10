# Utilities

Cross-cutting helpers. `XarrayEstimator` is the scikit-learn ↔ xarray
bridge: it wraps any scikit-learn estimator so that `fit` / `transform` /
`predict` accept and return xarray objects, preserving coordinates and
dimension names. It backs the decomposition helpers in
[Transforms](transforms.md).

::: xrtoolz.utils.XarrayEstimator
