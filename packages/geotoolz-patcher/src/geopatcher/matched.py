"""Public alias for `geopatcher._src.matched`.

Re-exports the matched-field surface so users can write
``from geopatcher.matched import MatchedField`` without reaching
into the private ``_src`` layer.

See ``docs/patcher/design/query-matchup.md`` §6 and
``docs/patcher/decisions.md`` (ADR-003) for the design.
"""

from __future__ import annotations

from geopatcher._src.matched import *  # noqa: F403
from geopatcher._src.matched import __all__ as __all__
