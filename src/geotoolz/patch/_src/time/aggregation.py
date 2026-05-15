"""`TemporalAggregation` — time → time reconstruction.

Four aggregations:

- `TemporalFold`       - RNN-like stateful fold over patches.
- `TemporalMean`       - per-anchor mean across temporal patches.
- `TemporalHierarchicalCombine` - stitch multi-scale outputs
  (pairs with `TemporalMultiScale`).
- `TemporalForecast`   - keep only the horizon portion of each patch.

`TemporalFold` is the design's `Sequential` time aggregation, renamed to
avoid clashing with the operator `geotoolz.Sequential` already exported
at the top level.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar


class TemporalAggregation:
    """Base for time-axis merge strategies."""

    streaming_safe: ClassVar[bool] = False
    forbid_in_yaml: ClassVar[bool] = False

    def merge(self, patches: Iterable[Any]) -> Any:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass(eq=False)
class TemporalFold(TemporalAggregation):
    """Stateful left-fold across patches — the RNN / state-space shape.

    Args:
        fold_fn: ``(state, patch) -> state``. Carries closures, so
            ``forbid_in_yaml = True``.
        initial_state: Starting accumulator (default ``None``).
    """

    fold_fn: Callable[[Any, Any], Any]
    initial_state: Any = None

    streaming_safe: ClassVar[bool] = True
    forbid_in_yaml: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any]) -> Any:
        state = self.initial_state
        for p in patches:
            state = self.fold_fn(state, p)
        return state


@dataclass(eq=False)
class TemporalMean(TemporalAggregation):
    """TemporalMean across patches' data (numpy ``stack`` + ``mean``)."""

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any]) -> Any:
        import numpy as np

        data = [np.asarray(p.data, dtype=np.float64) for p in patches]
        if not data:
            return np.array(0.0)
        return np.mean(np.stack(data, axis=0), axis=0)


@dataclass(eq=False)
class TemporalHierarchicalCombine(TemporalAggregation):
    """Stitch multi-scale outputs — pairs with `TemporalMultiScale` geometry.

    Args:
        scales: List of lookback lengths matching `TemporalMultiScale.scales`.
    """

    scales: list[int] = field(default_factory=list)

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any]) -> dict[int, Any]:
        out: dict[int, Any] = {}
        for p in patches:
            out[int(p.anchor)] = p.data
        return out

    def get_config(self) -> dict[str, Any]:
        return {"scales": list(self.scales)}


@dataclass(eq=False)
class TemporalForecast(TemporalAggregation):
    """Keep only the horizon portion - pairs with `TemporalLookbackHorizon`.

    The patch's data is expected to be the lookback + horizon block; the
    aggregation returns a dict ``{anchor: horizon_block}`` so callers can
    align predictions back onto the time axis.

    Args:
        horizon: Length of the horizon block at the *end* of each patch.
    """

    horizon: int = 1

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any]) -> dict[int, Any]:
        import numpy as np

        out: dict[int, Any] = {}
        for p in patches:
            arr = np.asarray(p.data)
            out[int(p.anchor)] = arr[-int(self.horizon) :]
        return out

    def get_config(self) -> dict[str, Any]:
        return {"horizon": self.horizon}
