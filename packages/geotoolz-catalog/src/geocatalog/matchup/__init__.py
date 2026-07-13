"""`geocatalog.matchup` — spatial + temporal joining of catalog rows.

Hybrid-layout sub-namespace. Re-exports the `matchup` engine entry
point, the `MatchupRow` carrier persisted to ``matchups.parquet``,
and the strategy classes for spatial and temporal predicates.

See ``docs/design/query-matchup.md`` §4.4 / §4.6.
"""

from __future__ import annotations

from geocatalog._src.matchup import (
    CentroidWithin,
    Contains,
    Intersects,
    IouAtLeast,
    MatchupRow,
    NearestInTime,
    SpatialStrategy,
    Synchronous,
    TemporalStrategy,
    WithinWindow,
    matchup,
)


__all__ = [
    "CentroidWithin",
    "Contains",
    "Intersects",
    "IouAtLeast",
    "MatchupRow",
    "NearestInTime",
    "SpatialStrategy",
    "Synchronous",
    "TemporalStrategy",
    "WithinWindow",
    "matchup",
]
