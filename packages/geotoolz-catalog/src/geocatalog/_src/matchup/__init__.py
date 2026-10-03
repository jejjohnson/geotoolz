"""Matchup engine — pair / N-tuple observations in space and time.

`matchup` joins catalogs, bundles or iterables of `SourceRow` and
yields `MatchupRow`s whose tolerances are explicit and whose ids are
content hashes, so re-runs are reproducible. Persist them next to
``items.parquet`` with `CatalogBundle.write_matchups` — see
``docs/design/query-matchup.md`` §4.4.

Strategies are first-class objects in ``spatial.py`` and
``temporal.py``; the in-memory join (shapely STRtree) is in
``engine.py``.
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
