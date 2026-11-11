"""Map coordinates to pixels (#522), with functax's affine kernel (#643).

`to_pixel` inverts the transform: coordinate maths, so it is geoarray's to
export, and functax owns the kernel (`functax.geometry.to_pixel`, the same
arithmetic, re-exported here). It takes the array namespace of its inputs,
so it runs on NumPy alone and under ``jit`` / ``grad`` on JAX, and imports
JAX only when a JAX array arrives. Reading a grid at points, H(x), blends
pixel values and lives in geoops (`geoops.geom.interp_at`, GA-B6).
"""

from __future__ import annotations

from functax.geometry import to_pixel


__all__ = ["to_pixel"]
