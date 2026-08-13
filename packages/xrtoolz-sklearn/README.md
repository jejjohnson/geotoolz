# xrtoolz-sklearn

The scikit-learn ↔ xarray bridge for the xrtoolz stack (import name:
`xrsklearn`).

Three surfaces over one stack → delegate → unstack marshalling core:

- **`XarrayEstimator`** — wraps any sklearn-style estimator so
  `fit / transform / fit_transform / inverse_transform` operate on N-D
  `DataArray` / `Dataset` inputs, with `NanPolicy` controlling NaN
  handling around the delegate.
- **`da.sklearn` / `ds.sklearn` accessors** — registered as a side
  effect of `import xrsklearn`.
- **`SklearnOp`** — the `xrcore.Operator` wrapper for composing
  estimators into `pipekit.Sequential` chains and `Graph` pipelines.

```bash
pip install xrtoolz-sklearn
```

```python
from sklearn.decomposition import PCA
from xrsklearn import XarrayEstimator

pcs = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit_transform(da)
```
