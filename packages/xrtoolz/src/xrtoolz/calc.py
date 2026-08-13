"""Deprecated alias — ``xrtoolz.calc`` moved to the ``xrtoolz-grad``
distribution (import name :mod:`xrgrad`).

Every public name is re-exported here unchanged, so existing
``xrtoolz.calc`` imports keep working through the 0.x series. The shim
is removed at 1.0 — switch to ``import xrgrad``.
"""

from __future__ import annotations

import warnings

from xrgrad import (
    EARTH_RADIUS,
    GRAVITY,
    OMEGA,
    curl,
    divergence,
    gradient,
    grid_metrics_from_coords,
    laplacian,
    partial,
)


warnings.warn(
    "xrtoolz.calc has moved to the xrtoolz-grad distribution; "
    "use `import xrgrad` instead. This alias is removed at xrtoolz 1.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "EARTH_RADIUS",
    "GRAVITY",
    "OMEGA",
    "curl",
    "divergence",
    "gradient",
    "grid_metrics_from_coords",
    "laplacian",
    "partial",
]
