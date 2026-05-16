"""Operator wrappers — `GridSampler`, `ApplyToChips`, `Stitch` — around `geopatcher`.

Thin glue between the four-axis Patcher framework (which lives in the
standalone ``geopatcher`` package) and `geotoolz.core.Operator`, so a
sliding-window inference pipeline composes inside a `Sequential` /
`Graph`::

    pipe = Sequential([
        GridSampler(patcher),
        ApplyToChips(model_op),
        Stitch(SpatialOverlapAdd()),
    ])

Optional extra: ``pip install 'geotoolz[patch]'`` to pull in geopatcher.
Importing this module without geopatcher installed raises a friendly
``ImportError`` pointing at the right extra.
"""

from __future__ import annotations

from typing import Any, ClassVar

from geotoolz.core._src.operator import Operator


try:
    from geopatcher import Patch, SpatialAggregation, SpatialPatcher
except ImportError as _e:  # pragma: no cover - exercised when [patch] is missing
    raise ImportError(
        "geotoolz.patch_ops requires the `geopatcher` package. "
        "Install with `pip install 'geotoolz[patch]'` (or `pip install geopatcher`)."
    ) from _e


class GridSampler(Operator):
    """Operator: ``Field → list[Patch]`` — yields the Patcher's patches.

    Materialises the iterator into a list so downstream operators don't
    need to know about lazy iteration; users who want streaming should
    consume ``patcher.split`` directly.

    Args:
        patcher: The `SpatialPatcher` to drive.
    """

    forbid_in_yaml: ClassVar[bool] = False

    def __init__(self, patcher: SpatialPatcher) -> None:
        self.patcher = patcher

    def _apply(self, field: Any) -> list[Patch]:
        return list(self.patcher.split(field))

    def get_config(self) -> dict[str, Any]:
        return {"patcher": self.patcher.get_config()}


class ApplyToChips(Operator):
    """Operator: ``list[Patch] → list[Patch]`` — map ``operator`` over each patch.

    The inner operator runs against each ``patch.data`` and the result
    replaces ``patch.data``; ``anchor`` / ``indices`` / ``weights`` are
    preserved so downstream `Stitch` can reconstruct the field.

    Args:
        operator: The per-chip operator (a `ModelOp`, an `NDVI`, …).
    """

    forbid_in_yaml: ClassVar[bool] = False

    def __init__(self, operator: Operator) -> None:
        self.operator = operator

    def _apply(self, patches: list[Patch]) -> list[Patch]:
        out: list[Patch] = []
        for p in patches:
            out.append(
                Patch(
                    data=self.operator(p.data),
                    anchor=p.anchor,
                    indices=p.indices,
                    weights=p.weights,
                )
            )
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "operator": {
                "class": type(self.operator).__name__,
                "config": self.operator.get_config(),
            }
        }


class Stitch(Operator):
    """Operator: ``list[Patch] → field`` — wraps an `SpatialAggregation`.

    Pairs with `GridSampler` + `ApplyToChips` to express ``split →
    operator → merge`` as a three-step `Sequential`. The ``domain``
    argument is supplied at construction (commonly ``field.domain``) so
    the resulting `Operator` has a single positional input (the list of
    patches) and slots into the linear pipeline.

    Args:
        aggregation: The `SpatialAggregation` to apply.
        domain: The `Domain` the patches were drawn from. Required
            because the aggregation's output shape is fixed by the
            domain.
    """

    forbid_in_yaml: ClassVar[bool] = False

    def __init__(self, aggregation: SpatialAggregation, domain: Any) -> None:
        self.aggregation = aggregation
        self.domain = domain

    def _apply(self, patches: list[Patch]) -> Any:
        return self.aggregation.merge(patches, self.domain)

    def get_config(self) -> dict[str, Any]:
        return {
            "aggregation": {
                "class": type(self.aggregation).__name__,
                "config": self.aggregation.get_config(),
            }
        }
