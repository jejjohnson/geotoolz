"""Cross-cutting utilities — domain-agnostic helpers reused across modules.

Currently hosts the scikit-learn ↔ xarray bridge (:class:`XarrayEstimator`),
which lets any sklearn estimator operate on N-D :class:`xr.DataArray` /
:class:`xr.Dataset` inputs via stack→delegate→unstack marshalling.
Importing this module also registers the thin ``da.sklearn`` /
``ds.sklearn`` accessors, which construct an ``XarrayEstimator`` and
delegate to the same implementation.
"""

from xrtoolz.utils._src import sklearn_accessor as _sklearn_accessor  # noqa: F401
from xrtoolz.utils._src.sklearn_wrap import XarrayEstimator


__all__ = [
    "XarrayEstimator",
]
