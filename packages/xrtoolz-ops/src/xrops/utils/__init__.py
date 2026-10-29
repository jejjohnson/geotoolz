"""Cross-cutting utilities — domain-agnostic helpers reused across modules.

Hosts the internal finite-mask, grid-spacing, validation, and
optional-import helpers under :mod:`xrops.utils._src`.

The scikit-learn ↔ xarray bridge (``XarrayEstimator``, ``NanPolicy``, the
``da.sklearn`` / ``ds.sklearn`` accessors, and ``SklearnOp``) is the
:mod:`xrsklearn` package; import it from there.
"""

__all__: list[str] = []
