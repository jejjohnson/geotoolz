"""Temporal matchup strategies.

A `TemporalStrategy` filters a parallel list of candidate intervals
against a single primary interval. Three families are persisted
across the ``matchups.parquet`` ``strategy`` / ``tolerance_json``
columns:

* `NearestInTime` — pick the secondary nearest in time within Δt;
  produces at most one secondary per primary.
* `WithinWindow` — every secondary whose midpoint falls within a
  ``[t + start, t + end]`` window around the primary.
* `Synchronous` — overlapping observation intervals (within an
  optional tolerance).

The return shape is a ``pd.IntervalIndex`` containing the surviving
candidates *in input position order*, so the matchup engine can map
positions back to the underlying SourceRow list. NearestInTime
returns either an empty index or a single-element one.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class TemporalStrategy(Protocol):
    """Selector over candidate intervals.

    Concrete strategies are either *predicates* (Synchronous,
    WithinWindow — emit every candidate that satisfies the
    condition) or *selectors* (NearestInTime — emit at most the
    single best). The engine treats both uniformly via ``filter``.
    """

    def filter(
        self,
        primary: pd.Interval,
        candidates: pd.IntervalIndex,
    ) -> pd.IntervalIndex:
        """Return the subset of candidates that match the primary."""
        ...


def _to_timedelta(value: timedelta | str) -> pd.Timedelta:
    """Coerce ``timedelta`` / ISO-like string to a `pd.Timedelta`."""
    import pandas as pd

    return pd.Timedelta(value)


def _midpoint(interval: pd.Interval) -> pd.Timestamp:
    """Interval midpoint as a Timestamp (tz-aware if the input is)."""
    import pandas as pd

    left = pd.Timestamp(interval.left)
    right = pd.Timestamp(interval.right)
    return left + (right - left) / 2


@dataclasses.dataclass(frozen=True)
class NearestInTime:
    """Pick the secondary nearest in time within ``dt``.

    "Nearest" is measured between interval midpoints. If the
    nearest is still further than ``dt`` away, the result is empty.

    Args:
        dt: Maximum allowed time offset, e.g. ``timedelta(hours=6)``
            or ``"6h"`` (parsed via `pd.Timedelta`).
    """

    dt: timedelta | str

    def filter(
        self,
        primary: pd.Interval,
        candidates: pd.IntervalIndex,
    ) -> pd.IntervalIndex:
        import pandas as pd

        if len(candidates) == 0:
            return candidates[:0]
        dt_limit = _to_timedelta(self.dt)
        primary_mid = _midpoint(primary)
        # Build a parallel array of midpoints + |delta| seconds.
        mids = pd.Series(
            [_midpoint(iv) for iv in candidates], index=range(len(candidates))
        )
        deltas = (mids - primary_mid).abs()
        in_range = deltas <= dt_limit
        if not in_range.any():
            return candidates[:0]
        # `idxmin` over the in-range subset returns the position of
        # the smallest |delta|; ties broken by first occurrence.
        best_pos = int(deltas[in_range].astype("int64").idxmin())
        return candidates[best_pos : best_pos + 1]


@dataclasses.dataclass(frozen=True)
class WithinWindow:
    """Candidates whose midpoint falls in ``[primary.mid + start, primary.mid + end]``.

    Useful for "give me everything within ±12 h of each primary".
    ``start`` is typically negative (look back); ``end`` positive
    (look forward).

    Args:
        start: Offset from the primary midpoint. Negative looks back.
        end: Offset from the primary midpoint. Positive looks forward.
    """

    start: timedelta | str
    end: timedelta | str

    def filter(
        self,
        primary: pd.Interval,
        candidates: pd.IntervalIndex,
    ) -> pd.IntervalIndex:
        import pandas as pd

        if len(candidates) == 0:
            return candidates[:0]
        primary_mid = _midpoint(primary)
        lower = primary_mid + _to_timedelta(self.start)
        upper = primary_mid + _to_timedelta(self.end)
        mids = pd.Series(
            [_midpoint(iv) for iv in candidates], index=range(len(candidates))
        )
        keep = (mids >= lower) & (mids <= upper)
        positions = [i for i, k in enumerate(keep) if k]
        return candidates[positions]


@dataclasses.dataclass(frozen=True)
class Synchronous:
    """Candidates whose intervals overlap the primary's, within tolerance.

    Equivalent to `WithinWindow(start=-tolerance, end=+tolerance)`
    applied to interval overlap (not midpoints) — useful for
    matching observations that should genuinely co-occur in time
    (e.g. simultaneous flyovers).

    Args:
        tolerance: Slack on either side of the primary interval.
            ``"0s"`` (default) enforces strict overlap.
    """

    tolerance: timedelta | str = "0s"

    def filter(
        self,
        primary: pd.Interval,
        candidates: pd.IntervalIndex,
    ) -> pd.IntervalIndex:
        import pandas as pd

        if len(candidates) == 0:
            return candidates[:0]
        tol = _to_timedelta(self.tolerance)
        primary_left = pd.Timestamp(primary.left) - tol
        primary_right = pd.Timestamp(primary.right) + tol
        positions = []
        for i, iv in enumerate(candidates):
            cand_left = pd.Timestamp(iv.left)
            cand_right = pd.Timestamp(iv.right)
            # Intervals overlap iff each starts before the other ends.
            if cand_left <= primary_right and cand_right >= primary_left:
                positions.append(i)
        return candidates[positions]
