"""`TemporalAggregation` — time → time reconstruction.

Four aggregations:

- `TemporalFold`       - RNN-like stateful fold over patches.
- `TemporalMean`       - per-time-step mean of the overlapping patches.
- `TemporalHierarchicalCombine` - stitch multi-scale outputs
  (pairs with `TemporalMultiScale`).
- `TemporalForecast`   - keep only the post-anchor horizon of each patch.

`TemporalFold` is the design's `Sequential` time aggregation, renamed to
avoid clashing with operator-graph `Sequential` types in downstream
composition libraries (e.g. `geotoolz.Sequential`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from geopatcher._src._serialize import config_from_fields


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
    """Per-time-step mean of the overlapping patches — the `SpatialMean` of time.

    Each patch is placed at its ``indices`` (a ``slice`` along the time
    axis) and accumulated into a running per-step sum and count, so the
    merge holds one output-sized buffer, never the patch stream, and
    shrunk edge windows (``boundary="shrink"``) of different lengths merge
    fine. NaN samples are not counted; steps no patch reached get
    ``fill_value``.

    Args:
        time_axis: Which axis of ``patch.data`` is the time axis. Must
            match the patcher's ``time_axis``. Default 0.
        time_len: Length of the output time axis. ``None`` (default) uses
            the furthest patch ``indices.stop``; pass the series length to
            cover trailing steps no patch reached (they get ``fill_value``).
        fill_value: Written into steps with no valid sample. Default NaN.

    Raises:
        ValueError: On an empty patch stream without ``time_len`` (the
            output shape is unknown), a patch whose ``indices`` is not a
            unit-step ``slice`` matching its data, or patches whose
            non-time shapes differ.
    """

    time_axis: int = 0
    time_len: int | None = None
    fill_value: float = math.nan

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any]) -> np.ndarray:
        ax = int(self.time_axis)
        total: np.ndarray | None = None
        count: np.ndarray | None = None
        for p in patches:
            x = np.moveaxis(np.asarray(p.data, dtype=np.float64), ax, 0)
            sl = _time_slice(p, x.shape[0], type(self).__name__)
            if total is None or count is None:
                n = self.time_len if self.time_len is not None else sl.stop
                total = np.zeros((max(int(n), sl.stop), *x.shape[1:]))
                count = np.zeros_like(total)
            elif x.shape[1:] != total.shape[1:]:
                raise ValueError(
                    f"TemporalMean: patch at anchor {p.anchor} has non-time "
                    f"shape {x.shape[1:]}, expected {total.shape[1:]}"
                )
            if sl.stop > total.shape[0]:
                grow = [(0, sl.stop - total.shape[0])] + [(0, 0)] * (x.ndim - 1)
                total, count = np.pad(total, grow), np.pad(count, grow)
            valid = ~np.isnan(x)
            total[sl] += np.where(valid, x, 0.0)
            count[sl] += valid
        if total is None or count is None:
            if self.time_len is None:
                raise ValueError(
                    "TemporalMean.merge got no patches; pass time_len= to "
                    "get an all-fill output for an empty stream."
                )
            return np.full(int(self.time_len), self.fill_value, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(count > 0, total / count, self.fill_value)
        return np.moveaxis(out, 0, ax)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


def _time_slice(patch: Any, length: int, owner: str) -> slice:
    """``patch.indices`` as a validated unit-step slice of ``length`` steps."""
    s = patch.indices
    if not isinstance(s, slice) or s.start is None or s.stop is None:
        raise ValueError(
            f"{owner} needs patch.indices to be a bounded time slice; got "
            f"{s!r} at anchor {patch.anchor}"
        )
    if s.step not in (None, 1) or int(s.stop) - int(s.start) != length:
        raise ValueError(
            f"{owner}: patch at anchor {patch.anchor} has indices {s!r} but "
            f"{length} steps of data along the time axis"
        )
    return slice(int(s.start), int(s.stop))


@dataclass(eq=False)
class TemporalHierarchicalCombine(TemporalAggregation):
    """Stitch multi-scale outputs — pairs with `TemporalMultiScale` geometry.

    `TemporalMultiScale.window` emits one patch per (anchor, scale), so the
    aggregation keys on the *(anchor, scale)* pair rather than on the
    anchor alone. The scale is the patch's `TemporalPatch.window_index`
    (set by `TemporalPatcher`), never its realised slice — two scales
    shrunk to the same edge slice stay distinct. Hand-built patches with
    ``window_index=None`` are numbered in arrival order per anchor (the
    order `split` yields them).

    Returns ``{anchor: {scale_key: data}}``: ``scale_key`` is
    ``scales[k]`` when ``scales`` is supplied (it should equal
    `TemporalMultiScale.scales`), else the window index ``k``.

    Args:
        scales: List of lookback lengths matching `TemporalMultiScale.scales`.
            When supplied, the inner-dict keys become the scale lengths.

    Raises:
        ValueError: If ``scales`` is supplied and a window index has no
            matching entry, or two patches claim the same (anchor, scale).
    """

    scales: list[int] = field(default_factory=list)

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any]) -> dict[int, dict[Any, Any]]:
        out: dict[int, dict[Any, Any]] = {}
        for p in patches:
            anchor = int(p.anchor)
            inner = out.setdefault(anchor, {})
            k = getattr(p, "window_index", None)
            k = len(inner) if k is None else int(k)
            if self.scales:
                if not 0 <= k < len(self.scales):
                    raise ValueError(
                        f"TemporalHierarchicalCombine: window index {k} at "
                        f"anchor {anchor} has no entry in scales={self.scales}"
                    )
                key: Any = int(self.scales[k])
            else:
                key = k
            if key in inner:
                raise ValueError(
                    f"TemporalHierarchicalCombine: two patches for anchor "
                    f"{anchor}, scale {key!r}"
                )
            inner[key] = p.data
        return out

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class TemporalForecast(TemporalAggregation):
    """Keep only the horizon portion - pairs with `TemporalLookbackHorizon`.

    The horizon is the ``horizon`` steps *after the anchor*,
    ``[anchor + 1, anchor + 1 + horizon)``, located from the anchor rather
    than taken as the patch's tail. A patch's data is either the whole
    lookback + horizon block covering ``patch.indices`` (the horizon is
    sliced out of it) or, for a model that predicts only the future, a
    ``horizon``-step block (kept as is). Patches whose window does not
    contain the full horizon (a ``boundary="shrink"`` window at the end
    of the axis) are skipped, never mislabelled. The aggregation returns
    ``{anchor: horizon_block}`` so callers can align predictions back
    onto the time axis.

    Args:
        horizon: Number of steps after the anchor to keep.
        time_axis: Which axis of ``patch.data`` is the time axis. Must
            match the patcher's ``time_axis``. Default 0.

    Raises:
        ValueError: A patch whose ``indices`` is not a bounded slice, or
            whose time length is neither the window's nor ``horizon``.
    """

    horizon: int = 1
    time_axis: int = 0

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any]) -> dict[int, Any]:
        out: dict[int, Any] = {}
        ax = int(self.time_axis)
        h = int(self.horizon)
        for p in patches:
            arr = np.asarray(p.data)
            n = arr.shape[ax]
            s = p.indices
            if not isinstance(s, slice) or s.start is None or s.stop is None:
                raise ValueError(
                    "TemporalForecast needs patch.indices to be a bounded time "
                    f"slice; got {s!r} at anchor {p.anchor}"
                )
            first = int(p.anchor) + 1 - int(s.start)
            if first < 0 or first + h > int(s.stop) - int(s.start):
                continue  # the window does not hold the whole horizon
            if n == int(s.stop) - int(s.start):
                idx: list[Any] = [slice(None)] * arr.ndim
                idx[ax] = slice(first, first + h)
                out[int(p.anchor)] = arr[tuple(idx)]
            elif n == h:
                out[int(p.anchor)] = arr
            else:
                raise ValueError(
                    f"TemporalForecast: patch at anchor {p.anchor} has {n} "
                    f"steps along time_axis={ax}; expected the window's "
                    f"{int(s.stop) - int(s.start)} or horizon={h}"
                )
        return out

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)
