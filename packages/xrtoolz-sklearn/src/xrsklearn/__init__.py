"""xrsklearn — the scikit-learn ↔ xarray bridge for the xrtoolz stack.

Three surfaces over one marshalling core (stack → delegate → unstack):

- :class:`XarrayEstimator` — wraps any sklearn-style estimator so
  ``fit / transform / fit_transform / inverse_transform`` operate on
  N-D :class:`xr.DataArray` / :class:`xr.Dataset` inputs, with
  :class:`NanPolicy` controlling NaN handling around the delegate.
- ``da.sklearn`` / ``ds.sklearn`` accessors — registered as a side
  effect of importing this package; thin sugar over
  :class:`XarrayEstimator`.
- :class:`SklearnOp` — the Layer-1 :class:`xrcore.Operator` wrapper, for
  composing fitted (or fit-on-first-call) estimators into
  ``pipekit.Sequential`` chains and ``Graph`` pipelines.
"""

from xrsklearn._src import accessor as _accessor  # noqa: F401  (registers .sklearn)
from xrsklearn._src.operator import SklearnOp
from xrsklearn._src.wrap import NanPolicy, XarrayEstimator


__version__ = "0.0.1"  # x-release-please-version

__all__ = [
    "NanPolicy",
    "SklearnOp",
    "XarrayEstimator",
    "__version__",
]
