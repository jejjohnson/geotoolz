# Learn

scikit-learn integration. Wraps any sklearn-compatible estimator (`fit` / `predict` / `transform` /
`decision_function` / `fit_predict`) as a carrier-aware Operator.

!!! warning "Phase-1 API — subject to change"
    The current `mode=` / `nan_fit=` / `nan_transform=` kwargs are provisional. The full design
    (`PixelTable` carrier, type-named wrappers like `PixelwiseClassifier`, `NanPolicy` dataclass)
    will replace this surface in a follow-up with deprecation shims. Supervised estimators
    (classifiers / regressors) must be pre-fit out-of-graph and loaded via `state_path=` —
    in-graph supervised fit helpers are tracked for v0.2.

- **Universal adapter:** `SklearnOp`, `GeoTensorEstimator`
- **Convenience wrappers** (named algorithm, sensible defaults). Each is prefixed `Pixelwise` so it
  never shadows the scikit-learn class it wraps, and takes that estimator by keyword —
  `gz.learn.PixelwisePCA(estimator=PCA(n_components=3))`:
  - Decomposition: `PixelwisePCA`, `PixelwiseIPCA`, `PixelwiseNMF`
  - Clustering: `PixelwiseKMeans`, `PixelwiseMiniBatchKMeans`, `PixelwiseGMM`
  - Anomaly: `PixelwiseIsolationForest`, `PixelwiseOneClassSVM`, `PixelwiseLocalOutlierFactor`
  - Imputation: `PixelwiseKNNImputer`, `PixelwiseIterativeImputer`
- **Framework-agnostic inference:** `ModelOp(model=...)`

::: geotoolz.learn
