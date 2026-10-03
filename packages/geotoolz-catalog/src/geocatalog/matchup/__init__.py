"""`geocatalog.matchup` — spatial + temporal joining of catalog rows.

Hybrid-layout sub-namespace. Re-exports the `matchup` engine entry
point, the `MatchupRow` carrier persisted to ``matchups.parquet``,
and the strategy classes for spatial and temporal predicates.

The namespace is itself callable — ``geocatalog.matchup(primary,
secondary, ...)`` forwards to the `matchup` function — because the
package binds this module over the top-level ``matchup`` name.

See ``docs/design/query-matchup.md`` §4.4 / §4.6.
"""

from __future__ import annotations

import sys
import types
from typing import Any

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


class _CallableNamespace(types.ModuleType):
    """This module's type: calling it calls `matchup`."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return matchup(*args, **kwargs)


# `inspect.signature(geocatalog.matchup)` follows this to the function.
__wrapped__ = matchup
sys.modules[__name__].__class__ = _CallableNamespace
