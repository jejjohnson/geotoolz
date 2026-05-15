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

    `TemporalMultiScale.window` emits one patch per (anchor, scale), so the
    aggregation must key on the *(anchor, scale-slice)* pair rather than on
    the anchor alone — otherwise later scales overwrite earlier ones.

    Returns ``{anchor: {(slice.start, slice.stop): data}}``: outer dict keyed
    by anchor, inner dict keyed by the patch's slice interval (the natural
    identifier for "which scale produced this"). When ``scales`` is supplied
    and matches `TemporalMultiScale.scales`, the inner key is the matching
    scale length instead; otherwise it's the slice tuple.

    Args:
        scales: List of lookback lengths matching `TemporalMultiScale.scales`.
            When supplied, the inner-dict keys become the scale lengths.
    """

    scales: list[int] = field(default_factory=list)

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any]) -> dict[int, dict[Any, Any]]:
        out: dict[int, dict[Any, Any]] = {}
        for p in patches:
            anchor = int(p.anchor)
            indices = p.indices
            scale_key: Any
            if (
                isinstance(indices, slice)
                and self.scales
                and (indices.stop - indices.start) in self.scales
            ):
                scale_key = int(indices.stop - indices.start)
            elif isinstance(indices, slice):
                scale_key = (int(indices.start), int(indices.stop))
            else:
                scale_key = repr(indices)
            out.setdefault(anchor, {})[scale_key] = p.data
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
        time_axis: Which axis of ``patch.data`` is the time axis. Must
            match the patcher's ``time_axis``. Default 0.
    """

    horizon: int = 1
    time_axis: int = 0

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any]) -> dict[int, Any]:
        import numpy as np

        out: dict[int, Any] = {}
        ax = int(self.time_axis)
        h = int(self.horizon)
        for p in patches:
            arr = np.asarray(p.data)
            # Slice the trailing `horizon` elements along `time_axis` only.
            idx: list[Any] = [slice(None)] * arr.ndim
            idx[ax] = slice(arr.shape[ax] - h, arr.shape[ax])
            out[int(p.anchor)] = arr[tuple(idx)]
        return out

    def get_config(self) -> dict[str, Any]:
        return {"horizon": self.horizon, "time_axis": self.time_axis}
