"""Matchup engine and `MatchupRow` carrier.

Joins iterables of `SourceRow`s on space + time:

* Build a spatial index (STRtree) over the secondary footprints.
* For each primary, query the index for spatial pre-candidates.
* Apply the user's temporal strategy to narrow by time.
* Apply the user's spatial strategy to confirm the match
  (the index pre-filter accepts envelope overlap; the strategy
  is the truth gate).
* Emit a `MatchupRow` per surviving secondary, plus N-way fan-out
  when ``secondary`` is a mapping of role → iterable.

The shipped ingestion API — `CatalogBundle.ingest` — is the thin
wrapper that takes a catalog + selector and streams rows into this
engine.

See ``docs/design/query-matchup.md`` §4.4 / §4.6.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from geocatalog._src._timeutil import to_utc_ts
from geocatalog._src.matchup.temporal import _midpoint


if TYPE_CHECKING:
    import shapely.geometry.base

    from geocatalog._src.matchup.spatial import SpatialStrategy
    from geocatalog._src.matchup.temporal import TemporalStrategy
    from geocatalog._src.sources._base import SourceRow


# A `Selector` filters a catalog before joining: ``{"source":
# "earthaccess", "collection": "MOD09GA"}``. Empty dict matches all.
# Kept here for the future catalog-driven wrapper; the engine itself
# takes iterables.
Selector = Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class MatchupRow:
    """A single matched tuple persisted to ``matchups.parquet``.

    The catalog of MatchupRows is itself a `GeoCatalog` — its
    geometry column is the common-footprint intersection, and its
    interval is the union of member intervals — so downstream code
    can query it with the same ``query(bounds, interval)`` calls as
    any other catalog. See design §4.4.

    Attributes:
        matchup_id: Stable identifier (uuid4 hex).
        strategy: Concise label naming the spatial + temporal
            strategies used (``"iou>=0.2 & nearest_in_time<=6h"``).
        member_ids: Parallel arrays with ``member_sources`` and
            ``member_roles``. ``member_ids[0]`` is always the primary.
        member_sources: ``SourceRow.source`` values for each member.
        member_roles: Role tags — ``"primary"``, ``"secondary"``, or
            user-defined names for N-way matchups.
        geometry_intersect: Common footprint (in catalog target CRS).
        time_reference: Reference timestamp the offsets are measured
            from — by convention the primary's interval midpoint.
        time_offset_sec: Parallel to ``member_ids``; offset of each
            member's interval midpoint relative to ``time_reference``.
        tolerance: Serialized strategy parameters, suitable for
            re-running the matchup deterministically.
        query_set: Optional user label, persisted as the
            ``query_set`` column in ``matchups.parquet`` so
            ``geocatalog stage --matchup-tag <name>`` can select a
            named set. Mirrors the ``tag`` argument of `matchup()`.
    """

    matchup_id: str
    strategy: str
    member_ids: tuple[str, ...]
    member_sources: tuple[str, ...]
    member_roles: tuple[str, ...]
    geometry_intersect: shapely.geometry.base.BaseGeometry
    time_reference: datetime
    time_offset_sec: tuple[float, ...]
    tolerance: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    query_set: str | None = None


def matchup(
    primary: Iterable[SourceRow],
    secondary: Iterable[SourceRow] | Mapping[str, Iterable[SourceRow]],
    *,
    spatial: SpatialStrategy,
    temporal: TemporalStrategy,
    join: Literal["all", "any"] = "all",
    tag: str | None = None,
) -> Iterator[MatchupRow]:
    """Find matching tuples of `SourceRow`s.

    Args:
        primary: Iterable of primary rows. Materialised lazily;
            each row is processed once.
        secondary: Either one iterable (pairwise — every secondary
            gets the role ``"secondary"``) or a mapping of role →
            iterable (N-way — each role indexed independently).
            Secondaries are materialised into a spatial index so
            each iterable is fully read before iteration begins.
        spatial: Strategy deciding spatial matches
            (e.g. ``Intersects()``, ``IouAtLeast(0.2)``).
        temporal: Strategy deciding temporal matches
            (e.g. ``NearestInTime(dt="6h")``).
        join: ``"all"`` (default) requires every secondary role to
            contribute a member; primaries with any empty role are
            skipped. ``"any"`` emits matchups missing some roles —
            handy for opportunistic fusion.
        tag: Optional user label persisted as ``query_set`` so a
            CLI user can ``--matchup-tag foo`` later.

    Yields:
        `MatchupRow` instances, one per matched tuple. Each row's
        ``matchup_id`` is a fresh uuid4 hex.
    """
    from geocatalog._src.matchup._engine_impl import run_matchup

    yield from run_matchup(
        primary=primary,
        secondary=secondary,
        spatial=spatial,
        temporal=temporal,
        join=join,
        tag=tag,
    )


def _utcnow() -> datetime:
    """Hoist for monkeypatching in deterministic tests."""
    return datetime.now(tz=UTC)


def _new_matchup_id() -> str:
    """Hoist for monkeypatching in deterministic tests."""
    return uuid.uuid4().hex


def _midpoint_seconds(interval: pd.Interval, reference: datetime) -> float:
    """Return ``(midpoint - reference).total_seconds()`` for an interval.

    Used as the temporal-offset entry in `MatchupRow.time_offset_sec`.
    The midpoint convention (shared with the temporal strategies via
    `geocatalog._src.matchup.temporal._midpoint`) is symmetric across
    instantaneous and range-shaped intervals.
    """
    mid = to_utc_ts(_midpoint(interval))
    return (mid.to_pydatetime() - reference).total_seconds()
