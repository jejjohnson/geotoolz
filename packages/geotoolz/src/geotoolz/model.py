"""DEPRECATED compatibility alias — `ModelOp` moved to `geotoolz.learn`.

The framework-agnostic inference wrapper lives with the rest of the
model-integration operators. Import ``from geotoolz.learn import
ModelOp`` (or use the top-level ``geotoolz.ModelOp`` re-export) — this
alias will be removed in a future release.
"""

from __future__ import annotations

from geotoolz.learn import ModelOp


__all__ = ["ModelOp"]
