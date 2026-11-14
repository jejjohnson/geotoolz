"""scikit-learn on xarray: Datasets, DataTrees and the ``.sklearn`` accessor.

The one scikit-learn bridge of the stack is pipekit-array's
`pipekit_array.Estimator` with `pipekit_array.to_rows` / `from_rows`
(geostack OP-O3); use it directly on a DataArray in a pipeline. This module
adds what is xarray's own:

- `XarrayEstimator`: a Dataset's variables side by side as one feature
  table, and a DataTree fitted per node or pooled over its nodes;
- the ``.sklearn`` accessor on DataArray, Dataset and DataTree, registered
  when this module is imported.

scikit-learn is imported when an estimator is fitted: install the
``xrtoolz-ops[sklearn]`` extra.
"""

from xrops.learn._src import accessor as _accessor  # noqa: F401  (registers .sklearn)
from xrops.learn._src.estimator import TreeMode, XarrayEstimator


__all__ = ["TreeMode", "XarrayEstimator"]
