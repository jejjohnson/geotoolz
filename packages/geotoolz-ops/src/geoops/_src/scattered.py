"""functax for geoops: its import, float64 scope, IDW plan and compiled gap fill.

Every geoops gridder and gap fill that reads scattered sources
(`geom.coregister`'s IDW and nearest paths, `geom.coregister.SwathToGrid`,
`restore.gap_fill_idw` / `gap_fill_nearest`) builds the same plan,
``functax.interp.InverseDistance(k, power, radius).plan(sources, targets)``,
so they share one distance rule:

    wᵢⱼ = max(dᵢⱼ, ε)⁻ᵖ / ∑ₗ max(dᵢₗ, ε)⁻ᵖ   over the k nearest sources,

with ε = 10⁻¹², so a target on a source takes that source's value.

functax comes from the ``[interp]`` extra (JAX, and SciPy for the host k-d
tree); `functax_module` imports it lazily with the install hint. The plans run
under ``jax.enable_x64`` so NumPy float64 coordinates and values stay float64
whatever the caller's JAX setting.

`fill_gaps_plane` is the one jitted function in geoops: it takes and returns
NumPy, so no caller can trace through it, and it compiles
``functax.project.fill_gaps`` once per process (one trace per method and plane
shape). Eager, ``fill_gaps`` re-traces its conjugate-gradient solve on every
call. Everything else here stays un-jitted, and the caller jits.
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from geoops._src.optional import import_optional


if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from jaxtyping import Bool, Float


__all__ = ["fill_gaps_plane", "functax_module", "inverse_distance", "x64"]

_EXTRA = "interp"


def functax_module(name: str, *, feature: str) -> ModuleType:
    """``functax.<name>``, imported lazily with the ``[interp]`` install hint.

    Args:
        name: The functax namespace (``"interp"``, ``"geometry"``, ``"project"``).
        feature: What needs it, for the error message.

    Returns:
        The module.

    Raises:
        ImportError: functax (or its JAX / SciPy extras) is missing; the
            message names ``geotoolz-ops[interp]``.
    """
    return import_optional(f"functax.{name}", _EXTRA, feature=feature)


def x64(feature: str) -> AbstractContextManager[Any]:
    """``jax.enable_x64(True)``: float64 stays float64 inside functax."""
    jax = import_optional("jax", _EXTRA, feature=feature)
    enable = getattr(jax, "enable_x64", None)
    if enable is not None:
        return enable(True)
    from jax.experimental import enable_x64  # JAX before ``jax.enable_x64``

    return enable_x64()


def inverse_distance(
    source_xy: Float[np.ndarray, "n 2"],
    values: Float[np.ndarray, "*batch n"],
    target_xy: Float[np.ndarray, "m 2"],
    *,
    k: int,
    power: float,
    radius: float | None = None,
    source_valid: Bool[np.ndarray, " n"] | None = None,
    feature: str,
) -> tuple[Float[np.ndarray, "*batch m"], Float[np.ndarray, " m"]]:
    """IDW of ``values`` from the sources onto the targets, with the nearest distances.

    Args:
        source_xy: ``(N, 2)`` source coordinates.
        values: ``(*batch, N)`` source values; NaN propagates.
        target_xy: ``(M, 2)`` target coordinates, in the sources' units.
        k: Neighbours per target (functax clamps it to the sources).
        power: Inverse-distance exponent ``p ≥ 0``.
        radius: Neighbours count only when ``d < radius``; ``None`` for no
            limit. A target with none is ``NaN``.
        source_valid: ``(N,)`` mask; ``False`` sources are never neighbours.
        feature: What calls it, for the missing-extra message.

    Returns:
        ``(*batch, M)`` float64 values (``NaN`` where a target has no
        neighbour), and the ``(M,)`` distance to each target's nearest
        neighbour (``inf`` where it has none).

    Example:
        >>> sources = np.array([[0.0, 0.0], [1.0, 0.0]])
        >>> out, nearest = inverse_distance(
        ...     sources, np.array([10.0, 20.0]), np.array([[0.25, 0.0]]),
        ...     k=2, power=2.0, feature="doc",
        ... )
        >>> out.round(12).tolist(), nearest.tolist()
        ([11.0], [0.25])
    """
    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n_targets = target_xy.shape[0]
    if n_targets == 0:
        # functax 0.0.6 cannot plan onto zero targets
        return np.empty((*values.shape[:-1], 0)), np.empty(0)
    interp = functax_module("interp", feature=feature)
    geometry = functax_module("geometry", feature=feature)
    with x64(feature):
        sources = geometry.Points(
            source_xy, valid=None if source_valid is None else np.asarray(source_valid)
        )
        scheme = interp.InverseDistance(k=k, power=power, radius=radius)
        plan = scheme.plan(sources, geometry.Points(target_xy))
        out = np.array(plan(values), dtype=np.float64)  # a writable host copy
        first = np.asarray(plan.read.indices)[:, 0]
        row_ok = np.asarray(plan.valid)
    # the taps come nearest first; a row with no neighbour has none
    nearest = np.linalg.norm(source_xy[first] - target_xy, axis=1)
    return out, np.where(row_ok, nearest, np.inf)


@functools.cache
def _compiled_fill_gaps() -> Any:
    """``jax.jit(functax.project.fill_gaps)``, built once per process."""
    jax = import_optional("jax", _EXTRA, feature="fill_gaps")
    project = functax_module("project", feature="fill_gaps")
    return jax.jit(project.fill_gaps, static_argnames=("method",))


def fill_gaps_plane(
    image: Float[np.ndarray, "row col"],
    observed: Bool[np.ndarray, "row col"],
    *,
    method: str,
    feature: str,
) -> Float[np.ndarray, "row col"]:
    """``functax.project.fill_gaps`` on one float64 plane, back as writable NumPy.

    Args:
        image: ``(H, W)`` values; the unobserved pixels are never read.
        observed: ``(H, W)`` mask, ``True`` at the pixels kept as they are.
        method: ``"laplace"`` (Δu = 0) or ``"biharmonic"`` (Δ²u = 0) in the holes.
        feature: What calls it, for the missing-extra message.

    Returns:
        The filled ``(H, W)`` float64 plane.

    Raises:
        ImportError: Without the ``[interp]`` extra, on every call (the cached
            compiled solve does not hide it).

    Example:
        >>> image = np.array([[0.0, 1.0, np.nan, np.nan, 0.0]])
        >>> fill_gaps_plane(image, np.isfinite(image), method="laplace",
        ...                 feature="doc").round(6).tolist()
        [[0.0, 1.0, 0.666667, 0.333333, 0.0]]
    """
    # checked on every call, so a missing extra still names itself once the
    # compiled solve is cached
    functax_module("project", feature=feature)
    with x64(feature):
        filled = _compiled_fill_gaps()(
            np.asarray(image, dtype=np.float64),
            mask=np.asarray(observed, dtype=bool),
            method=method,
        )
        # a writable copy: callers write planes into it
        return np.array(filled, dtype=np.float64)
