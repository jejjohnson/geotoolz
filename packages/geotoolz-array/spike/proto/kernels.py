"""The L1 kernels the spike runs: arrays in, arrays out, no carrier, no CRS.

Each is written against the array namespace of its input (``xp``), so the
same function runs on NumPy, JAX and dask blocks. ``#516`` drafted the
matched filter with lineax and einx; lineax is JAX-only, so a kernel that
must also run on NumPy and dask uses ``xp.linalg.solve`` instead (finding
F3 in RESULTS.md).
"""

from __future__ import annotations

from typing import Any


def _xp(a: Any) -> Any:
    mod = type(a).__module__
    if mod.startswith(("jax", "jaxlib")) or "Tracer" in type(a).__name__:
        import jax.numpy as jnp

        return jnp
    import numpy as np

    return np


def normalized_difference(a: Any, b: Any, eps: float) -> Any:
    """ND = (a − b) / (a + b + ε), elementwise. Spec ``"(),()->()"``."""
    # a − b and a + b + ε
    num = a - b
    den = a + b + eps
    return num / den


def background_stats(x: Any) -> tuple[Any, Any]:
    """Per-column background mean and covariance. Spec ``"(r,c)->(c),(c,d)"``.

    x: (R, C), the R pixels of one column over C bands.
    """
    xp = _xp(x)
    # μ = (1/R) ∑ᵣ xᵣ                                  (R, C) → (C,)
    mu = xp.mean(x, axis=0)
    # x̃ᵣ = xᵣ − μ                                       (R, C)
    xc = x - mu
    # Σ = (1/(R−1)) x̃ᵀx̃ + λI, λ for a well-posed solve  (C, C)
    n = x.shape[0]
    sigma = xc.T @ xc / (n - 1) + 1e-6 * xp.eye(x.shape[1], dtype=x.dtype)
    return mu, sigma


def matched_filter_score(x: Any, mu: Any, sigma: Any, *, t: Any) -> Any:
    """α per pixel of one column. Spec ``"(r,c),(c),(c,d)->(r)"``, vectorized.

    x: (R, C) column spectra; mu: (C,); sigma: (C, C); t: (C,) CH₄ target.
    """
    xp = _xp(x)
    # x̃ᵣ = xᵣ − μ                                       (R, C)
    xc = x - mu
    # w = Σ⁻¹t, solved, never inverted                  (C,)
    w = xp.linalg.solve(sigma, t)
    # αᵣ = x̃ᵣᵀw / tᵀw                                   (R,)
    return (xc @ w) / (t @ w)


def gradient(f: Any, *, spacing: Any) -> tuple[Any, Any]:
    """∂f/∂y and ∂f/∂x by centred differences, one-sided at the edges.

    Spec ``"(y,x)->(y,x),(y,x)"``; spacing = (Δy, Δx) in map units, derived
    from the carrier's transform (geometry=), so a traced transform flows.
    """
    xp = _xp(f)
    dy, dx = spacing[0], spacing[1]
    # interior: (f[i+1] − f[i−1]) / 2Δ; edges: one-sided      (..., Y, X)
    gy = (
        xp.concatenate(
            [
                f[..., 1:2, :] - f[..., 0:1, :],
                (f[..., 2:, :] - f[..., :-2, :]) / 2.0,
                f[..., -1:, :] - f[..., -2:-1, :],
            ],
            axis=-2,
        )
        / dy
    )
    gx = (
        xp.concatenate(
            [
                f[..., 1:2] - f[..., 0:1],
                (f[..., 2:] - f[..., :-2]) / 2.0,
                f[..., -1:] - f[..., -2:-1],
            ],
            axis=-1,
        )
        / dx
    )
    return gy, gx


def spacing_of(g: Any) -> Any:
    """(Δy, Δx) = (e, a) from a GeoArray's (6,) transform; signed, map units."""
    t = g.transform
    xp = _xp(t)
    return xp.stack([t[..., 4], t[..., 0]])
