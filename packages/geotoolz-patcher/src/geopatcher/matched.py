"""Public alias for `geopatcher._src.matched`.

Re-exports the matched-field surface so users can write
``from geopatcher.matched import MatchedField`` without reaching
into the private ``_src`` layer.

See ``docs/design/query-matchup.md`` §6 and
``docs/decisions.md`` (ADR-003) for the design.
"""

from __future__ import annotations

from geopatcher._src.matched import (
    MatchedField,
    MatchedPatch,
    MatchedSpatialPatcher,
    MatchedSpatioTemporalPatch,
    MatchedSpatioTemporalPatcher,
    MatchedTemporalPatch,
    MatchedTemporalPatcher,
)


__all__ = [
    "MatchedField",
    "MatchedPatch",
    "MatchedSpatialPatcher",
    "MatchedSpatioTemporalPatch",
    "MatchedSpatioTemporalPatcher",
    "MatchedTemporalPatch",
    "MatchedTemporalPatcher",
]
