"""Cross-cutting utilities — domain-agnostic helpers reused across modules.

Currently hosts the scikit-learn ↔ xarray bridge (:class:`XarrayEstimator`),
which lets any sklearn estimator operate on N-D :class:`xr.DataArray` /
:class:`xr.Dataset` inputs via stack→delegate→unstack marshalling.
"""

from xr_toolz.utils._src.sklearn_wrap import XarrayEstimator


__all__ = [
    "XarrayEstimator",
]
