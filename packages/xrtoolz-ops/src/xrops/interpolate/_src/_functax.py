"""Private bridge to the functax plans behind the gap fills and IDW.

functax (the ``[functax]`` extra) runs on JAX, which the base install of
xrops does not carry, so it is imported when a fill runs, never at module
import. Every call runs under ``jax.enable_x64`` so the plans work in
float64, as the NumPy kernels they replace did; the setting is thread-local
and scoped to the call, so it neither leaks into the caller's JAX session
nor races between dask threads.

xrops is the outermost caller of these plans (its callers hand it xarray,
not JAX arrays), so it is where ``fill_gaps`` is compiled: one ``jax.jit``
per method and slice shape. Eager, ``fill_gaps`` re-traces its conjugate
gradient solve on every call, about a second per slice.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from jaxtyping import Bool, Float

from xrops.utils._src.optional_imports import _require_optional


#: SciPy's ``RBFInterpolator`` kernel names that are polyharmonic splines,
#: with their order k in functax's ``Polyharmonic(k)``
POLYHARMONIC_ORDERS: dict[str, int] = {
    "linear": 1,
    "thin_plate_spline": 2,
    "cubic": 3,
    "quintic": 5,
}


def _require_functax(module: str, feature: str) -> Any:
    """``functax.<module>``, or an ImportError naming the ``[functax]`` extra."""
    return _require_optional(
        f"functax.{module}",
        extra="functax",
        feature=feature,
        package="functax[jax,scipy]",
    )


def _x64(feature: str) -> Any:
    """The ``jax.enable_x64(True)`` context, scoped to one call."""
    jax = _require_optional(
        "jax", extra="functax", feature=feature, package="functax[jax,scipy]"
    )
    return jax.enable_x64(True)


@functools.cache
def _compiled_fill_gaps() -> Any:
    """``jax.jit(functax.project.fill_gaps)``, built once per process."""
    jax = _require_optional(
        "jax", extra="functax", feature="fill_gaps", package="functax[jax,scipy]"
    )
    project = _require_functax("project", "fill_gaps")
    return jax.jit(project.fill_gaps, static_argnames=("method",))


def fill_gaps_slice(
    image: Float[np.ndarray, "lat lon"],
    observed: Bool[np.ndarray, "lat lon"],
    method: str,
    *,
    feature: str,
) -> Float[np.ndarray, "lat lon"]:
    """``functax.project.fill_gaps`` on one float64 slice, back as writable NumPy."""
    # checked on every call, so a missing extra still names itself once the
    # compiled solve is cached
    _require_functax("project", feature)
    with _x64(feature):
        filled = _compiled_fill_gaps()(
            np.asarray(image, dtype=np.float64),
            mask=np.asarray(observed, dtype=bool),
            method=method,
        )
        # a writable copy: callers patch cells back into it
        return np.array(filled)


def inverse_distance(
    src_xy: Float[np.ndarray, "src two"],
    src_values: Float[np.ndarray, "src"],
    query_xy: Float[np.ndarray, "query two"],
    *,
    k: int,
    power: float,
    max_distance: float | None,
    eps: float,
) -> Float[np.ndarray, "query"]:
    """Euclidean IDW of the ``k`` nearest sources by ``functax.interp.InverseDistance``.

    xrops keeps a neighbour at ``d <= max_distance`` while functax keeps one
    at ``d < radius``, so the radius is the next float above
    ``max_distance``. functax needs a positive distance floor, so
    ``eps = 0`` becomes the smallest normal float64.
    """
    feature = "inverse-distance interpolation"
    interp = _require_functax("interp", feature)
    geometry = _require_functax("geometry", feature)
    radius = None if max_distance is None else float(np.nextafter(max_distance, np.inf))
    floor = float(eps) if eps > 0 else float(np.finfo(np.float64).tiny)
    with _x64(feature):
        scheme = interp.InverseDistance(
            k=int(k), power=float(power), radius=radius, floor=floor
        )
        plan = scheme.plan(
            geometry.Points(np.asarray(src_xy, dtype=np.float64)),
            geometry.Points(np.asarray(query_xy, dtype=np.float64)),
        )
        return np.asarray(plan(np.asarray(src_values, dtype=np.float64)))


def polyharmonic_rbf(
    src_xy: Float[np.ndarray, "src two"],
    src_values: Float[np.ndarray, "src"],
    query_xy: Float[np.ndarray, "query two"],
    *,
    order: int,
) -> Float[np.ndarray, "query"]:
    """Global polyharmonic RBF by ``functax.interp.RBF``, SciPy's default tail."""
    feature = "fillnan_rbf(neighbors=None)"
    interp = _require_functax("interp", feature)
    geometry = _require_functax("geometry", feature)
    with _x64(feature):
        plan = interp.RBF(interp.Polyharmonic(order)).plan(
            geometry.Points(np.asarray(src_xy, dtype=np.float64)),
            geometry.Points(np.asarray(query_xy, dtype=np.float64)),
        )
        return np.asarray(plan(np.asarray(src_values, dtype=np.float64)))


__all__ = [
    "POLYHARMONIC_ORDERS",
    "fill_gaps_slice",
    "inverse_distance",
    "polyharmonic_rbf",
]
