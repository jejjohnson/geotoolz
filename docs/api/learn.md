# Learn

scikit-learn integration. Wraps any sklearn-compatible estimator (`fit` / `predict` / `transform` /
`decision_function` / `fit_predict`) as a carrier-aware Operator.

!!! warning "Phase-1 API — subject to change"
    The current `mode=` / `nan_fit=` / `nan_transform=` kwargs are provisional. The full design
    (`PixelTable` carrier, type-named wrappers like `PixelwiseClassifier`, `NanPolicy` dataclass)
    will replace this surface in a follow-up (a breaking change, without aliases). Supervised estimators
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

## Saved state

`SklearnOp.save_state(path)` (and `GeoTensorEstimator.save_state`) writes one joblib pickle holding
the fitted estimator and imputer. Pass `write_meta=True` to also write a `<path>.meta.json` sidecar
with the scikit-learn version, fit input shape / sample count and a UTC `fit_timestamp`; it is off
by default so a save is deterministic and writes a single file.

!!! danger "Only load state files you trust"
    `load_state(path)` and the `state_path=` constructor argument unpickle the file with
    `joblib.load`, and unpickling can execute arbitrary code. Only load files you created yourself
    or that come from a source you trust as much as the code you run — never one from an
    untrusted upload, download or share. `state_path=` loads at construction, so this applies to
    instantiating a pipeline config too.

## API reference

::: geotoolz.learn
