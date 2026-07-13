"""Public alias for `geopatcher._src.time`.

Re-exports the four temporal axes and `TemporalPatcher` so users can do either
``import geopatcher.time`` / ``from geopatcher.time import TemporalPatcher``
or the top-level ``from geopatcher import TemporalPatcher``.
"""

from __future__ import annotations

from geopatcher._src.time import *  # noqa: F403
from geopatcher._src.time import __all__ as __all__
