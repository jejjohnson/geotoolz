"""Matched-stack operators — multi-source fusion of co-registered tensors.

Sibling to the existing composites (``MaxNDVIComposite``,
``CloudFreeComposite``, etc.) but for a *matched tuple* of tensors
from different sources rather than a temporal stack of one sensor.

* `StackMatched` — concatenate N aligned tensors along a new band axis.
* `BlendMatched` — weighted mean across N aligned tensors, with
  optional inverse-variance weighting when uncertainty maps are
  present.

Inputs are assumed already co-registered (typically by a
``geotoolz.geom.coregister`` operator before this op). Scaffolding
only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pipekit import Operator


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from georeader.geotensor import GeoTensor


class StackMatched(Operator):
    """Concatenate aligned tensors along a new band axis.

    Args:
        order: Optional list of source names fixing the output band
            order. ``None`` keeps the input order.
        fill: Value substituted where a source is missing. NaN means
            "downstream model handles it"; a numeric value means
            "imputed here".
    """

    def __init__(
        self,
        *,
        order: list[str] | None = None,
        fill: float = float("nan"),
    ) -> None:
        self.order = list(order) if order is not None else None
        self.fill = fill

    def get_config(self) -> dict[str, Any]:
        return {"order": self.order, "fill": self.fill}

    def __call__(self, tensors: Sequence[GeoTensor]) -> GeoTensor:
        raise NotImplementedError("Phase 3 PR — see design §5.")


class BlendMatched(Operator):
    """Weighted mean across N aligned tensors.

    Args:
        method: ``"mean"`` (equal weights), ``"weighted_mean"`` (use
            ``weights``), or ``"ivw"`` (inverse-variance weighting,
            requires per-source variance maps via ``variances``).
        weights: Per-source scalar weights. Ignored unless
            ``method == "weighted_mean"``.
        nan_policy: ``"ignore"`` (treat NaN as missing) or
            ``"propagate"`` (any NaN poisons the output pixel).
    """

    def __init__(
        self,
        *,
        method: Literal["mean", "weighted_mean", "ivw"] = "mean",
        weights: list[float] | None = None,
        nan_policy: Literal["ignore", "propagate"] = "ignore",
    ) -> None:
        self.method = method
        self.weights = list(weights) if weights is not None else None
        self.nan_policy = nan_policy

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "weights": self.weights,
            "nan_policy": self.nan_policy,
        }

    def __call__(
        self,
        tensors: Sequence[GeoTensor],
        variances: Sequence[np.ndarray] | None = None,
    ) -> GeoTensor:
        raise NotImplementedError("Phase 3 PR — see design §5.")
