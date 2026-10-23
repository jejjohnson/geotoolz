"""H(x): bilinear sampling of a grid at points, with padding (geoarray #522).

Point sets are padded to a fixed N with a validity mask, so a new day of
observations never recompiles. Padded rows may hold NaN coordinates; the
double ``where`` keeps both J and ∇ₓJ finite (GA-G16).
"""

from __future__ import annotations

from typing import Any

from proto.carrier import GeoArray


def _jnp() -> Any:
    import jax.numpy as jnp

    return jnp


def bilinear_at(field: Any, transform: Any, xy: Any, valid: Any) -> Any:
    """L1: field (..., H, W) at map points xy (N, 2) → (..., N); 0 where not valid.

    transform is (a, b, c, d, e, f): x = a·col + b·row + c, y = d·col + e·row + f.
    """
    jnp = _jnp()
    a, b, c, d, e, f = (transform[k] for k in range(6))
    H, W = field.shape[-2], field.shape[-1]
    # padded rows → the first pixel centre, so no NaN enters the arithmetic
    safe_x = jnp.where(valid, xy[:, 0], c + 0.5 * a + 0.5 * b)
    safe_y = jnp.where(valid, xy[:, 1], f + 0.5 * d + 0.5 * e)
    # (col, row) = T⁻¹(x, y), then centres at +½                      (N,)
    det = a * e - b * d
    col = (e * (safe_x - c) - b * (safe_y - f)) / det - 0.5
    row = (-d * (safe_x - c) + a * (safe_y - f)) / det - 0.5
    # i₀ = ⌊row⌋ clipped so (i₀, i₀ + 1) is inside; weights in [0, 1]   (N,)
    i0 = jnp.clip(jnp.floor(row), 0, H - 2).astype(int)
    j0 = jnp.clip(jnp.floor(col), 0, W - 2).astype(int)
    wr = jnp.clip(row - i0, 0.0, 1.0)
    wc = jnp.clip(col - j0, 0.0, 1.0)
    # f(r, c) = ∑ corner values × bilinear weights                     (..., N)
    f00 = field[..., i0, j0]
    f01 = field[..., i0, j0 + 1]
    f10 = field[..., i0 + 1, j0]
    f11 = field[..., i0 + 1, j0 + 1]
    val = (1 - wr) * ((1 - wc) * f00 + wc * f01) + wr * ((1 - wc) * f10 + wc * f11)
    return jnp.where(valid, val, 0.0)


def sample(g: GeoArray, xy: Any, valid: Any) -> Any:
    """The carrier form: a GeoArray["... y x"] at points → raw (..., N)."""
    if g.transform.ndim != 1:
        raise ValueError("sample takes one transform for every frame")
    return bilinear_at(g.data, g.transform, xy, valid)
