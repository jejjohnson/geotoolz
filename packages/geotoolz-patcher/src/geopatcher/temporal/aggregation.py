"""`geopatcher.temporal.aggregation` — How temporal patch outputs combine.

`Fold`, `Mean`, `HierarchicalCombine` and `Forecast`; `Aggregation` is
the base.
"""

from __future__ import annotations

from geopatcher._src.temporal.aggregation import (
    Aggregation,
    Fold,
    Forecast,
    HierarchicalCombine,
    Mean,
)


__all__ = [
    "Aggregation",
    "Fold",
    "Forecast",
    "HierarchicalCombine",
    "Mean",
]
