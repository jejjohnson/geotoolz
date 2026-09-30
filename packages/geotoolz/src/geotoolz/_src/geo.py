"""Shared georeferencing helpers.

The one implementation of three checks every geo-aware operator family
needs:

* :func:`require_geotensor` -- reject a plain array handed to an operator
  that needs an affine ``transform`` (and CRS);
* :func:`grid_matches` -- whether two rasters sit on the same pixel grid
  (spatial shape, transform, CRS);
* :func:`pixel_xy` -- vectorised CRS coordinates of pixel centres.

All three are duck-typed on the ``transform`` / ``crs`` attributes, so any
GeoTensor-compatible carrier passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import Float


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


__all__ = ["grid_matches", "pixel_xy", "require_geotensor"]


def require_geotensor(
    x: Any, op_name: str, *, arg: str = "input", hint: str = ""
) -> GeoTensor:
    """Return ``x`` if it is georeferenced, else raise a clear ``TypeError``.

    Geo-dependent operators (reprojection, rasterisation, metre-based
    measurements, ...) are only meaningful with an affine transform
    attached. The check is duck-typed: any carrier whose ``transform``
    attribute is present and not ``None`` passes.

    Args:
        x: The carrier to check.
        op_name: Operator name, used as the message prefix.
        arg: Which argument ``x`` is (``"input"``, ``"like"``, ``"src"``,
            ...), for operators with several raster arguments.
        hint: Optional extra sentence appended to the message (e.g. how
            to call the plain-array primitive instead).

    Returns:
        ``x`` unchanged.

    Raises:
        TypeError: ``"<op_name> requires a georeferenced GeoTensor
            <arg>; got a plain array (<type>)."``
    """
    if getattr(x, "transform", None) is None:
        message = (
            f"{op_name} requires a georeferenced GeoTensor {arg}; "
            f"got a plain array ({type(x).__name__})."
        )
        if hint:
            message = f"{message} {hint}"
        raise TypeError(message)
    return x


def grid_matches(
    a: Any, b: Any, *, atol: float = 0.0, spatial_only: bool = True
) -> bool:
    """Return whether two rasters share a pixel grid.

    Semantics (the one definition used package-wide):

    1. **Shape** -- the spatial shape ``shape[-2:]`` must be equal; with
       ``spatial_only=False`` the *full* shape must be equal (band / time
       axes included), for per-pixel reductions over a stack.
    2. **Georeferencing** -- only when *both* sides carry a ``transform``
       (plain arrays have no georeferencing, so the check degrades to the
       shape comparison). Then the CRS must compare equal and the six
       affine coefficients must agree within ``atol`` (absolute, no
       relative term): ``np.allclose(a.transform, b.transform, rtol=0,
       atol=atol)``.

    The default ``atol=0.0`` is exact equality: sub-pixel grid drift is a
    real bug source and fails loudly. Pass a small ``atol`` explicitly
    where float round-off in derived transforms must be tolerated.

    Args:
        a: First raster (GeoTensor or plain array).
        b: Second raster.
        atol: Absolute tolerance on each transform coefficient, in CRS
            units. ``0.0`` means exact.
        spatial_only: Compare only the trailing ``(H, W)`` axes (default)
            instead of the full shape.

    Returns:
        ``True`` when the grids match.
    """
    shape_a, shape_b = np.shape(a), np.shape(b)
    if spatial_only:
        shape_a, shape_b = shape_a[-2:], shape_b[-2:]
    if shape_a != shape_b:
        return False
    transform_a = getattr(a, "transform", None)
    transform_b = getattr(b, "transform", None)
    if transform_a is None or transform_b is None:
        return True
    if getattr(a, "crs", None) != getattr(b, "crs", None):
        return False
    return bool(
        np.allclose(
            np.asarray(tuple(transform_a), dtype=float),
            np.asarray(tuple(transform_b), dtype=float),
            rtol=0.0,
            atol=atol,
        )
    )


def pixel_xy(
    transform: Any, rows: Any, cols: Any
) -> tuple[Float[np.ndarray, ...], Float[np.ndarray, ...]]:
    """Return CRS coordinates of pixel centres, vectorised.

    Pixel ``(row, col)`` covers ``[row, row + 1) x [col, col + 1)`` in
    pixel space, so its centre is ``(row + 0.5, col + 0.5)``. Fractional
    ``rows`` / ``cols`` (e.g. sub-pixel contour vertices) are offset the
    same way.

    Args:
        transform: Affine-like geotransform (``a, b, c, d, e, f``
            attributes) mapping ``(col, row)`` to CRS ``(x, y)``.
        rows: Row indices (scalar or array).
        cols: Column indices, broadcastable against ``rows``.

    Returns:
        ``(xs, ys)`` float64 arrays of the broadcast shape of ``rows`` and
        ``cols``.
    """
    rows = np.asarray(rows, dtype=np.float64) + 0.5
    cols = np.asarray(cols, dtype=np.float64) + 0.5
    xs = transform.c + transform.a * cols + transform.b * rows
    ys = transform.f + transform.d * cols + transform.e * rows
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
