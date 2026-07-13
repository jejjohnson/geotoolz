"""Matchup engine — pair / N-tuple persisted catalog entries.

Given a populated `GeoCatalog` (typically grown via
``catalog.ingest(source, ...)``), find tuples of rows that overlap in
space and time within explicit tolerances. Output is a *new catalog*
of `MatchupRow` instances persisted to a sibling Parquet table next
to ``items.parquet`` — see ``docs/design/query-matchup.md`` §4.4.

Strategies are first-class objects in ``spatial.py`` and
``temporal.py``; the engine in ``engine.py`` runs the spatial /
temporal join (DuckDB SQL where possible).
"""

from __future__ import annotations

from geocatalog._src.matchup.engine import MatchupRow, matchup
from geocatalog._src.matchup.spatial import (
    CentroidWithin,
    Contains,
    Intersects,
    IouAtLeast,
    SpatialStrategy,
)
from geocatalog._src.matchup.temporal import (
    NearestInTime,
    Synchronous,
    TemporalStrategy,
    WithinWindow,
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
