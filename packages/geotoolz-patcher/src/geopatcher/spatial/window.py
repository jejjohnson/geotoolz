"""`geopatcher.spatial.window` — Per-pixel weights: boundary treatment for the patch.

`Boxcar` (no taper), `Hann`, `Tukey`, `Gaussian` and the `Custom`
escape hatch; `Window` is the base. `geom_shape` gives a fixed-size
geometry's weight-array shape for custom windows.
"""

from __future__ import annotations

from geopatcher._src.spatial.window import (
    Boxcar,
    Custom,
    Gaussian,
    Hann,
    Tukey,
    Window,
    geom_shape,
)


__all__ = [
    "Boxcar",
    "Custom",
    "Gaussian",
    "Hann",
    "Tukey",
    "Window",
    "geom_shape",
]
