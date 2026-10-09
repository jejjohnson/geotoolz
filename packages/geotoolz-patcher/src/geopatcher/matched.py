"""`geopatcher.matched` — patching several co-registered sources at once.

`MatchedField` pairs fields (a scene and its labels, two sensors over one
area) through a co-registration function; the `Matched*Patcher` family
splits them together, yielding one `MatchedPatch` (spatial, temporal or
spatio-temporal) holding every source's chip for the same anchor.

See ``docs/catalog/design/query-matchup.md`` §6 and
``docs/patcher/decisions.md`` (ADR-003) for the design.
"""

from __future__ import annotations

from geopatcher._src.matched import *  # noqa: F403
from geopatcher._src.matched import __all__ as __all__
