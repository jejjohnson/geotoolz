"""Map coordinates to pixels (#522).

`to_pixel` inverts the transform: coordinate maths, so it is geoarray's.
Reading a grid at points, H(x), blends pixel values and lives in geoops
(`geoops.geom.interp_at`, GA-B6). Written against the array namespace, so
it runs on NumPy and under ``jit`` / ``grad`` on JAX.
"""

from __future__ import annotations

from typing import Any

from geoarray._src.backend import namespace


__all__ = ["to_pixel"]


def to_pixel(transform: Any, xy: Any) -> Any:
    """Map coordinates ``(x, y)`` to fractional pixel ``(row, col)`` centres.

    With T = (a, b, c, d, e, f), column i and row j map to x = a·i + b·j + c,
    y = d·i + e·j + f. Inverting the 2 × 2 part,

        (i, j)ᵀ = [[a, b], [d, e]]⁻¹ (x − c, y − f)ᵀ,

    and subtracting ½ from both puts integer values at pixel centres (area
    convention), so ``(0, 0)`` is the centre of the top-left pixel.

    Args:
        transform: ``(6,)`` affine transform (NumPy or JAX).
        xy: ``(N, 2)`` positions in the transform's CRS.

    Returns:
        ``(N, 2)`` fractional ``(row, col)``.

    Example:
        >>> import numpy as np
        >>> t = np.array([20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0])
        >>> to_pixel(t, np.array([[600010.0, 3549990.0]])).tolist()
        [[0.0, 0.0]]
    """
    xp = namespace(xy)
    a, b, c, d, e, f = (transform[k] for k in range(6))
    det = a * e - b * d
    dx = xy[..., 0] - c
    dy = xy[..., 1] - f
    # [[a, b], [d, e]]⁻¹ = [[e, −b], [−d, a]] / det
    col = (e * dx - b * dy) / det - 0.5
    row = (-d * dx + a * dy) / det - 0.5
    return xp.stack([row, col], axis=-1)
