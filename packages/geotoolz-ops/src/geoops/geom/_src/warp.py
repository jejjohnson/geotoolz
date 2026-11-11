"""A grid warped through a functax plan: ``Reproject(engine="interp")`` (#644).

A warp is one linear read-out built from the geometry. The centres c_k of
the target pixels go through φ, the map from the target CRS to the source
CRS, once on the host with pyproj, X_{*,k} = φ(c_k); a functax scheme then
plans H on the source grid at X_*, and every frame and band is
f̂(X_*) = H y. The source transform enters the plan as an array, so the
plan and its apply trace under ``jit`` on a JAX carrier.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyproj
from geoarray import GeoArray, Grid

from geoops._src.valid import invalid_elements


__all__ = ["target_coordinates", "warp"]


def target_coordinates(target: Grid, source_crs: pyproj.CRS | None) -> np.ndarray:
    """X_* = φ(c): the target pixel centres in the source CRS, ``(M, 2)``.

    Row-major over the target (M = H′·W′), x then y. A centre pyproj cannot
    map comes back non-finite, and the plan marks it invalid.

    Example:
        >>> grid = Grid((10.0, 0.0, 0.0, 0.0, -10.0, 20.0), "EPSG:32613", (2, 2))
        >>> target_coordinates(grid, grid.crs).tolist()
        [[5.0, 15.0], [15.0, 15.0], [5.0, 5.0], [15.0, 5.0]]
    """
    from functax.geometry import pixel_centres

    x, y = pixel_centres(target)
    if source_crs is not None and target.crs is not None and target.crs != source_crs:
        # φ: target CRS -> source CRS, in (x, y) order whatever the axis order
        to_source = pyproj.Transformer.from_crs(target.crs, source_crs, always_xy=True)
        x, y = to_source.transform(x, y)
    return np.stack([np.ravel(x), np.ravel(y)], axis=-1).astype(np.float64)


def warp(
    gt: GeoArray,
    target: Grid,
    scheme: Any,
    boundary: str,
    coordinates: np.ndarray,
) -> tuple[Any, Any]:
    """f̂(X_*) = H y on every frame and band, and the fill of the result.

    Nodata (non-finite or the fill) is zeroed before the plan and a target
    pixel that reads it with a non-zero weight is written with the fill, as
    is one off the source grid. A float input keeps its dtype; an integer
    one keeps it under ``nearest`` and becomes float64 otherwise.

    Args:
        gt: The source carrier ``(..., H, W)``, NumPy or JAX.
        target: The target grid.
        scheme: The functax scheme (`resolve_scheme`).
        boundary: The scheme's edge boundary (`resolve_scheme`).
        coordinates: X_*, ``(H′·W′, 2)`` in ``gt``'s CRS (`target_coordinates`).

    Returns:
        The values ``(..., H′, W′)`` on JAX, and the output fill value.
    """
    import jax.numpy as jnp
    from functax.geometry import AffineGrid, Points
    from functax.interp import Nearest, Policies

    policies = Policies(
        outside="nan", missing="propagate", boundary=boundary, extent="cells"
    )
    source = AffineGrid(jnp.asarray(gt.transform), (gt.height, gt.width))
    plan = scheme.plan(source, Points(jnp.asarray(coordinates)), policies=policies)
    data = jnp.asarray(gt.data)
    # y: every frame and band flattened once, (..., H, W) -> (..., N)
    flat = data.reshape(*data.shape[:-2], gt.height * gt.width)
    usable = ~invalid_elements(flat, gt.fill_value)
    clean = jnp.where(usable, flat, jnp.zeros((), flat.dtype))
    # f̂ = H 𝒮(y) for every frame and band at once
    values = plan(clean)
    taps = plan.read.indices
    # a nodata tap with weight > 0 leaves the target pixel unsupported
    supported = jnp.all(usable[..., taps] | (plan.read.weights == 0), axis=-1)
    valid = supported & plan.valid
    floating = flat.dtype.kind in "fc"
    if floating or isinstance(scheme, Nearest):
        # a float input keeps its dtype; nearest copies values, so any dtype
        values = values.astype(flat.dtype)
    fill = gt.fill_value
    if values.dtype.kind in "fc":
        fill = fill if floating and fill is not None else float("nan")
    elif fill is None:
        fill = 0
    out = jnp.where(valid, values, jnp.asarray(fill, dtype=values.dtype))
    return out.reshape(*data.shape[:-2], *target.shape), fill
