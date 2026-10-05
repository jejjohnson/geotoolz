# xrtoolz-sklearn

The scikit-learn ↔ xarray bridge for the xrtoolz stack (import name:
`xrsklearn`).

Three surfaces over one stack → delegate → unstack marshalling core:

- **`XarrayEstimator`** — wraps any sklearn-style estimator so
  `fit / transform / fit_transform / inverse_transform / predict /
  predict_proba / score` operate on N-D `DataArray` / `Dataset` /
  `DataTree` inputs.
- **`da.sklearn` / `ds.sklearn` / `dt.sklearn` accessors** — registered as
  a side effect of `import xrsklearn`.
- **`SklearnOp`** — the `xrcore.Operator` wrapper for composing
  estimators into `pipekit.Sequential` chains and `Graph` pipelines.

What the bridge takes care of:

- **Feature validation** — inputs are checked against the grid seen at
  fit time; permuted dims / coordinates are reordered, a different grid
  raises instead of being scored column-for-column.
- **Labeled outputs per method** — scaler output back on the input grid
  (a Dataset stays a Dataset), PCA scores as `(sample, component)`,
  predictions on the target's grid, probabilities labeled by class.
- **NaN policies** — `nan_policy="mask"` drops land-mask columns
  (learned at fit) and gappy rows (with `y` and `sample_weight`), then
  restores both as NaN.
- **DataTrees** — `tree_mode` fits one estimator per node, one pooled
  over nodes, or one on all nodes' variables side by side.

```bash
pip install xrtoolz-sklearn
```

```python
from sklearn.decomposition import PCA
from xrsklearn import XarrayEstimator

wrap = XarrayEstimator(PCA(n_components=5), sample_dim="time", nan_policy="mask")
pcs = wrap.fit_transform(ssh)           # land + gaps handled
recon = wrap.inverse_transform(pcs)     # back on the (time, lat, lon) grid
```
