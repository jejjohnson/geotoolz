"""Public alias for `geopatcher._src.spatial`.

Re-exports the four spatial axes (Geometry, Sampler, Window, Aggregation) and
the `SpatialPatcher` / `AsyncSpatialPatcher` so users can do either
``import geopatcher.spatial`` / ``from geopatcher.spatial import SpatialPatcher``
or the top-level ``from geopatcher import SpatialPatcher``.
"""

from __future__ import annotations

from geopatcher._src.spatial import *  # noqa: F403
from geopatcher._src.spatial import __all__ as __all__
