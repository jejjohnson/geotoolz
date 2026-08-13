"""Cross-cutting utilities — domain-agnostic helpers reused across modules.

Hosts the internal finite-mask, grid-spacing, validation, and
optional-import helpers under :mod:`xrtoolz.utils._src`.

The scikit-learn ↔ xarray bridge (``XarrayEstimator``, ``NanPolicy``, the
``da.sklearn`` / ``ds.sklearn`` accessors, and ``SklearnOp``) moved to the
:mod:`xrsklearn` package. ``XarrayEstimator`` is re-exported here for
backward compatibility, and importing this module still registers the
accessors (via the :mod:`xrsklearn` import below) — but new code should
import from :mod:`xrsklearn` directly.
"""

from xrsklearn import XarrayEstimator


__all__ = [
    "XarrayEstimator",
]
