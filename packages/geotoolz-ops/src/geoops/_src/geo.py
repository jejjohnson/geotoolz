"""Shared georeferencing helpers.

The one implementation of the checks every geo-aware operator family
needs:

* :func:`require_geoarray` -- reject a plain array handed to an operator
  that needs an affine ``transform`` (and CRS);
* :func:`reject_geotensor` -- refuse a georeader ``GeoTensor``, naming
  ``from_georeader`` (geotoolz takes `geoarray.GeoArray` only);
* :func:`affine_of` -- a carrier's transform as an ``affine.Affine``;
* :func:`require_projected_crs` -- reject a geographic CRS handed to an
  operator that measures lengths, areas or slopes;
* :func:`grid_matches` -- whether two rasters sit on the same pixel grid
  (spatial shape, transform, CRS);
* :func:`require_grid_match` -- raise a ``ValueError`` naming the operator
  when the carriers of a multi-input operator sit on different grids;
* :func:`ground_pixel_size` -- the true ground length of one pixel step
  along each axis, on any rotated (not sheared) grid;
* :func:`pixel_xy` -- vectorised CRS coordinates of pixel centres.

The carrier is `geoarray.GeoArray`; plain arrays pass where an operator
accepts them (no georeferencing).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from affine import Affine
from geoarray import GeoArray, PixelWindow, as_transform
from jaxtyping import Float
from pyproj import CRS


__all__ = [
    "affine_of",
    "as_pixel_window",
    "grid_matches",
    "ground_pixel_size",
    "is_geotensor",
    "pixel_xy",
    "reject_geotensor",
    "require_geoarray",
    "require_grid_match",
    "require_projected_crs",
]

_FROM_GEOREADER = (
    "convert it at the boundary with "
    "geoarray.interop.georeader.from_georeader(gt) (zero-copy)"
)


def is_geotensor(x: Any) -> bool:
    """Whether ``x`` is a georeader ``GeoTensor`` (checked without importing it)."""
    return any(
        cls.__name__ == "GeoTensor" and cls.__module__.startswith("georeader")
        for cls in type(x).__mro__
    )


def reject_geotensor(x: Any, op_name: str = "geotoolz") -> None:
    """Raise ``TypeError`` for a georeader ``GeoTensor``; pass anything else.

    A ``GeoTensor`` is an ``np.ndarray`` subclass, so ``np.asarray`` would
    take it silently as a plain array and the output would lose its
    georeferencing. geotoolz takes `geoarray.GeoArray` instead.

    Args:
        x: The value to check.
        op_name: Who refuses it, for the message.

    Raises:
        TypeError: ``x`` is a ``GeoTensor``; the message names
            ``from_georeader``.
    """
    if is_geotensor(x):
        raise TypeError(
            f"{op_name} takes a geoarray.GeoArray, not a georeader GeoTensor; "
            f"{_FROM_GEOREADER}"
        )


def affine_of(x: Any) -> Affine:
    """The transform of ``x`` as an ``affine.Affine`` (``~T``, ``T * (c, r)``).

    Args:
        x: A `GeoArray` (concrete, not batched), an ``Affine``, six
            numbers or a ``(6,)`` array.

    Returns:
        The ``Affine``.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> g = GeoArray(np.zeros((2, 2)), (10, 0, 500000, 0, -10, 4e6), 32629)
        >>> affine_of(g).c
        500000.0
    """
    if isinstance(x, GeoArray):
        return x.grid.affine
    return Affine(*as_transform(x))


def as_pixel_window(window: Any) -> PixelWindow:
    """A rasterio ``Window`` (or a `PixelWindow`) as a `geoarray.PixelWindow`.

    Offsets and sizes must be whole pixels (round a rasterio window with
    ``round_offsets`` / ``round_lengths`` first).

    Raises:
        ValueError: A fractional offset or size.

    Example:
        >>> from rasterio.windows import Window
        >>> as_pixel_window(Window(col_off=2, row_off=3, width=4, height=5))
        PixelWindow(col_off=2, row_off=3, width=4, height=5)
    """
    if isinstance(window, PixelWindow):
        return window
    values = {}
    for name in ("col_off", "row_off", "width", "height"):
        value = float(getattr(window, name))
        if value != round(value):
            raise ValueError(f"window {name}={value} is not a whole number of pixels")
        values[name] = round(value)
    return PixelWindow(**values)


def require_geoarray(
    x: Any, op_name: str, *, arg: str = "input", hint: str = ""
) -> GeoArray:
    """Return ``x`` if it is a `GeoArray`, else raise a clear ``TypeError``.

    Geo-dependent operators (reprojection, rasterisation, metre-based
    measurements, ...) are only meaningful with an affine transform
    attached, so a plain array is refused (and a ``GeoTensor`` is refused
    naming ``from_georeader``).

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
        TypeError: ``"<op_name> requires a georeferenced GeoArray
            <arg>; got a plain array (<type>)."``
    """
    reject_geotensor(x, op_name)
    if not isinstance(x, GeoArray):
        message = (
            f"{op_name} requires a georeferenced GeoArray {arg}; "
            f"got a plain array ({type(x).__name__})."
        )
        if hint:
            message = f"{message} {hint}"
        raise TypeError(message)
    return x


def require_projected_crs(
    gt: Any,
    op_name: str,
    *,
    what: str = "areas and distances",
    metres: bool = True,
) -> None:
    """Raise unless ``gt`` is in a projected CRS (with metre units).

    Operators that treat the affine transform as a length (pixel areas,
    distances, slopes) would silently work in degrees on a geographic CRS,
    so it is rejected with a hint to reproject first. A carrier without a
    CRS is accepted as-is: its transform is then assumed to be linear.

    Args:
        gt: Georeferenced carrier (anything exposing ``.crs``).
        op_name: Operator name used in the error message.
        what: What the operator computes, for the message (``"areas and
            distances"``, ``"slopes"``, ...).
        metres: Also require metre linear units (default). Pass
            ``False`` where any linear unit works as long as the data
            agree with it (e.g. a slope from feet elevations on a
            US-feet grid).

    Raises:
        ValueError: If the CRS is geographic (or otherwise not
            projected), or -- with ``metres=True`` -- projected with
            non-metre linear units.
    """
    crs_input = getattr(gt, "crs", None)
    if crs_input is None:
        return
    crs = CRS.from_user_input(crs_input)
    hint = (
        "Reproject to a projected metric CRS (e.g. the local UTM zone) first, "
        "e.g. with geoops.geom.Reproject(dst_crs=...) or "
        "geoops.geom.ReprojectLike."
    )
    if not crs.is_projected:
        kind = "geographic" if crs.is_geographic else "not projected"
        raise ValueError(
            f"{op_name} computes {what} from linear pixel sizes and needs a "
            f"projected CRS; got {crs.name!r} ({kind}). {hint}"
        )
    if metres and any(axis.unit_conversion_factor != 1.0 for axis in crs.axis_info):
        units = sorted({axis.unit_name for axis in crs.axis_info})
        raise ValueError(
            f"{op_name} computes {what} in metres; CRS "
            f"{crs.name!r} has linear units {units}. {hint}"
        )


def ground_pixel_size(transform: Any, op_name: str) -> tuple[float, float]:
    """Return the ``(row_step, col_step)`` ground length of one pixel step.

    Moving one column shifts the CRS position by ``(a, d)`` and one row by
    ``(b, e)``, so the step lengths are ``hypot(a, d)`` and
    ``hypot(b, e)`` -- equal to ``|a|`` / ``|e|`` only on a north-up grid.
    On a rotated grid the two step vectors stay perpendicular, so per-axis
    sampling (``scipy.ndimage.distance_transform_edt(sampling=...)``,
    :func:`numpy.gradient` spacings) measures true ground distances. A
    sheared grid has non-perpendicular steps, where no per-axis sampling is
    exact; it is rejected.

    Args:
        transform: The geotransform: an ``Affine``, six numbers, a
            ``(6,)`` array, or a `GeoArray` (its transform).
        op_name: Operator name used in the error message.

    Returns:
        ``(row_step, col_step)`` in CRS units, for ``sampling`` /
        ``pixel_size`` arguments ordered ``(row, col)``.

    Raises:
        ValueError: If the transform is sheared or degenerate.
    """
    transform = affine_of(transform)
    a, b, d, e = (
        float(transform.a),
        float(transform.b),
        float(transform.d),
        float(transform.e),
    )
    col_step = math.hypot(a, d)
    row_step = math.hypot(b, e)
    if col_step == 0.0 or row_step == 0.0:
        raise ValueError(f"{op_name}: degenerate transform {tuple(transform)[:6]}")
    if abs(a * b + d * e) > 1e-9 * col_step * row_step:
        raise ValueError(
            f"{op_name} needs perpendicular pixel axes to measure ground "
            f"distances; the transform is sheared ({tuple(transform)[:6]}). "
            "Resample to a rotated or north-up grid first, e.g. with "
            "geoops.geom.Reproject."
        )
    return row_step, col_step


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
        a: First raster (GeoArray or plain array).
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
    if not (isinstance(a, GeoArray) and isinstance(b, GeoArray)):
        return True
    transform_a, transform_b = a.transform, b.transform
    if a.crs != b.crs:
        return False
    return bool(
        np.allclose(
            np.asarray(tuple(transform_a), dtype=float),
            np.asarray(tuple(transform_b), dtype=float),
            rtol=0.0,
            atol=atol,
        )
    )


def _describe_grid(x: Any, spatial_only: bool) -> str:
    shape = tuple(np.shape(x)[-2:] if spatial_only else np.shape(x))
    if not isinstance(x, GeoArray):
        return f"shape {shape}"
    coeffs = tuple(round(float(c), 6) for c in tuple(np.asarray(x.transform))[:6])
    return f"shape {shape}, transform {coeffs}, crs {getattr(x, 'crs', None)}"


def require_grid_match(
    a: Any,
    b: Any,
    op_name: str,
    *,
    names: tuple[str, str] = ("input", "other"),
    atol: float = 0.0,
    spatial_only: bool = True,
) -> None:
    """Raise unless two carriers of a multi-input operator share a pixel grid.

    The one guard behind every multi-input operator (#141): the carriers
    are combined pixel by pixel, so a second carrier on another grid (a
    20 m mask against a 10 m enhancement, a DEM over a neighbouring
    tile) would give a silently wrong answer. Semantics are those of
    :func:`grid_matches`.

    Args:
        a: The primary carrier.
        b: The second carrier.
        op_name: Operator name for the error message.
        names: How the message refers to ``a`` and ``b``.
        atol: Absolute tolerance on each transform coefficient.
        spatial_only: Compare only the trailing ``(H, W)`` axes.

    Raises:
        ValueError: The grids differ; the message names the operator and
            both grids.
    """
    if not grid_matches(a, b, atol=atol, spatial_only=spatial_only):
        raise ValueError(
            f"{op_name}: the {names[1]} pixel grid "
            f"({_describe_grid(b, spatial_only)}) does not match the "
            f"{names[0]} pixel grid ({_describe_grid(a, spatial_only)}). Resample one "
            "onto the other first, e.g. with geoops.geom.ResampleLike."
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
        transform: The geotransform mapping ``(col, row)`` to CRS
            ``(x, y)``: an ``Affine``, six numbers, a ``(6,)`` array, or a
            `GeoArray` (its transform).
        rows: Row indices (scalar or array).
        cols: Column indices, broadcastable against ``rows``.

    Returns:
        ``(xs, ys)`` float64 arrays of the broadcast shape of ``rows`` and
        ``cols``.
    """
    transform = affine_of(transform)
    rows = np.asarray(rows, dtype=np.float64) + 0.5
    cols = np.asarray(cols, dtype=np.float64) + 0.5
    xs = transform.c + transform.a * cols + transform.b * rows
    ys = transform.f + transform.d * cols + transform.e * rows
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
