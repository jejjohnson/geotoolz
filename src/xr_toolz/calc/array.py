"""Tier A — array-tier entry points for :mod:`xr_toolz.calc`.

Per design decision D11, every arithmetic submodule grows a duck-array
``axis=`` entry point under ``<module>/array.py``. This module re-exports
the pilot finite-difference kernels (``partial``, ``gradient``) for raw
numpy arrays under the default 2nd-order central-difference scheme.

Tier B wrappers in :mod:`xr_toolz.calc._src.cartesian` (and the
geometry-dispatched :mod:`xr_toolz.calc._src.operators`) keep the
``finitediffx``-backed higher-order / non-uniform / spherical paths;
this module is the simple numpy-only computational core that callers
can drop down to when they have raw arrays and want the standard
2nd-order centred stencil.
"""

from __future__ import annotations

from xr_toolz.calc._src.array_calc import (
    gradient,
    partial,
)


__all__ = [
    "gradient",
    "partial",
]
