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


# The engine prefers a strategy's ``select(primary, candidates) ->
# list[int]`` (positions of the matching candidates) when it has one:
# positions identify candidates exactly, whereas mapping the intervals
# ``filter`` returns back to rows is ambiguous when two candidates share
# an interval. Every shipped strategy defines both; ``filter`` stays the
# Protocol so a custom strategy only needs one.


def _to_timedelta(value: timedelta | str) -> pd.Timedelta:
    """Coerce ``timedelta`` / ISO-like string to a `pd.Timedelta`."""
    import pandas as pd

    return pd.Timedelta(value)


def _midpoint(interval: pd.Interval) -> pd.Timestamp:
    """Interval midpoint as a UTC-aware Timestamp (naive inputs are UTC).

    The one midpoint helper: strategies and the engine's
    ``time_offset_sec`` both use it, so naive and aware intervals can be
    compared without a ``TypeError``.
    """
    import pandas as pd

    from geocatalog._src._timeutil import to_utc_ts

    left = to_utc_ts(pd.Timestamp(interval.left))
    right = to_utc_ts(pd.Timestamp(interval.right))
    return left + (right - left) / 2


def _present(candidates: pd.IntervalIndex) -> list[tuple[int, pd.Interval]]:
    """``(position, interval)`` of the candidates that have times.

    A missing interval (``NaT`` endpoints) reads back as NaN and never
    matches.
    """
    import pandas as pd

    return [
        (i, iv)
        for i, iv in enumerate(candidates)
        if isinstance(iv, pd.Interval)
        and not pd.isna(iv.left)
        and not pd.isna(iv.right)
    ]


def _by_positions(
    candidates: pd.IntervalIndex, positions: list[int]
) -> pd.IntervalIndex:
    return candidates[positions]


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
        return _by_positions(candidates, self.select(primary, candidates))

    def select(self, primary: pd.Interval, candidates: pd.IntervalIndex) -> list[int]:
        """Position of the nearest candidate within ``dt``; ties → first."""
        dt_limit = _to_timedelta(self.dt)
        primary_mid = _midpoint(primary)
        best: tuple[pd.Timedelta, int] | None = None
        for i, iv in _present(candidates):
            delta = abs(_midpoint(iv) - primary_mid)
            # Strict `<` keeps the first of equally near candidates; the
            # engine orders candidates stably, so the choice is
            # deterministic.
            if delta <= dt_limit and (best is None or delta < best[0]):
                best = (delta, i)
        return [] if best is None else [best[1]]


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
        return _by_positions(candidates, self.select(primary, candidates))

    def select(self, primary: pd.Interval, candidates: pd.IntervalIndex) -> list[int]:
        """Positions of the candidates whose midpoint is in the window."""
        primary_mid = _midpoint(primary)
        lower = primary_mid + _to_timedelta(self.start)
        upper = primary_mid + _to_timedelta(self.end)
        return [i for i, iv in _present(candidates) if lower <= _midpoint(iv) <= upper]


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
        return _by_positions(candidates, self.select(primary, candidates))

    def select(self, primary: pd.Interval, candidates: pd.IntervalIndex) -> list[int]:
        """Positions of the candidates overlapping the widened primary."""
        import pandas as pd

        from geocatalog._src._timeutil import to_utc_ts

        tol = _to_timedelta(self.tolerance)
        primary_left = to_utc_ts(pd.Timestamp(primary.left)) - tol
        primary_right = to_utc_ts(pd.Timestamp(primary.right)) + tol
        positions = []
        for i, iv in _present(candidates):
            cand_left = to_utc_ts(pd.Timestamp(iv.left))
            cand_right = to_utc_ts(pd.Timestamp(iv.right))
            # Intervals overlap iff each starts before the other ends.
            if cand_left <= primary_right and cand_right >= primary_left:
                positions.append(i)
        return positions
