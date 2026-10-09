"""`MatchedField` + `MatchedPatch` — co-located patches across sources.

This subpackage is the geopatcher half of the cross-package
query→matchup→patch design (see ``docs/catalog/design/query-matchup.md``
§6). It extends the single-source patching model to N co-registered
sources: a primary `Field` defines the anchor space, secondary
`Field`s are aligned per-anchor via a coregistration callable, and
the resulting `MatchedPatch` carries one local neighborhood per
source.

By construction `MatchedField` *is* a `Field` — it satisfies the
existing Protocol via its primary's domain and CRS — so the existing
samplers / geometries / windows place anchors on it unchanged. Its
``select()`` fans out to each member and returns a per-source
``dict``; the matched patchers (`MatchedSpatialPatcher`,
`MatchedTemporalPatcher`, `MatchedSpatioTemporalPatcher`) are required
to split it into matched carriers and to merge per source.

See ``docs/patcher/decisions.md`` (ADR-003) for the locked-in design
choices.
"""

from __future__ import annotations

from geopatcher._src.matched.field import MatchedField
from geopatcher._src.matched.patch import (
    MatchedPatch,
    MatchedSpatioTemporalPatch,
    MatchedTemporalPatch,
)
from geopatcher._src.matched.patcher import (
    MatchedSpatialPatcher,
    MatchedSpatioTemporalPatcher,
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
