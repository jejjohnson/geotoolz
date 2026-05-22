"""Operator wrappers — `GridSampler`, `ApplyToChips`, `Stitch` — around `geopatcher`.

Thin glue between the four-axis Patcher framework (which lives in the
standalone ``geopatcher`` package) and `pipekit.Operator`, so a
sliding-window inference pipeline composes inside a `Sequential` /
`Graph`::

    pipe = Sequential([
        GridSampler(patcher),
        ApplyToChips(model_op),
        Stitch(SpatialOverlapAdd(), domain=field.domain),
    ])

Optional extra: ``pip install 'geotoolz[patch]'`` to pull in
``geopatcher[pipekit]`` (which transitively installs `pipekit`).
Importing this module without geopatcher installed raises a friendly
``ImportError`` pointing at the right extra.

The same wrappers are also reachable as ``geopatcher.integrations.pipekit``
once the ``[patch]`` extra is installed — both module paths re-import
the same classes. Use whichever location reads better in your code; we
keep both available rather than picking a winner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from pipekit import Operator

from geotoolz._src.blending import triangular_weights


try:
    from geopatcher import Patch, SpatialAggregation, SpatialPatcher, SpatialWindow
except ImportError as _e:  # pragma: no cover - exercised when [patch] is missing
    raise ImportError(
        "geotoolz.patch_ops requires the `geopatcher` package. "
        "Install with `pip install 'geotoolz[patch]'` (or `pip install geopatcher`)."
    ) from _e


@dataclass(eq=False)
class SpatialTriangular(SpatialWindow):
    """Linear-ramp triangular spatial window for overlap-add blending."""

    width: int = 16

    def weights(self, geometry: Any) -> np.ndarray:
        size = getattr(geometry, "size", None)
        if size is None:
            raise TypeError(
                f"SpatialTriangular weights aren't defined for "
                f"{type(geometry).__name__}; expected a fixed-size geometry."
            )
        return triangular_weights(tuple(int(s) for s in size), self.width).astype(
            np.float64
        )

    def get_config(self) -> dict[str, Any]:
        return {"width": self.width}


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
