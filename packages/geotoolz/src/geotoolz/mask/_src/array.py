"""Pure-numpy helpers for geometry, morphology, and algebra masks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import scipy.ndimage as ndi
from jaxtyping import Bool, Float, Shaped

from geotoolz._src.labels import (
    Connectivity,
    label_components,
    remove_small_holes as _remove_small_holes_2d,
)
from geotoolz._src.shape import single_band


#: The ``keep=`` flag shared by every geometry / DEM mask: which side of
#: the region (polygon, elevation / slope interval, distance zone) is
#: *kept*. The mask itself is True on the other side (True = drop).
Keep = Literal["inside", "outside"]

_KEEP_VALUES = ("inside", "outside")


def _check_keep(keep: str, name: str) -> str:
    """Validate a ``keep=`` flag (``"inside"`` or ``"outside"``).

    Args:
        keep: The flag value.
        name: Caller name used in the error message.

    Returns:
        ``keep`` unchanged.

    Raises:
        ValueError: If ``keep`` is not ``"inside"`` or ``"outside"``.
    """
    if keep not in _KEEP_VALUES:
        raise ValueError(f"{name}: `keep` must be 'inside' or 'outside'; got {keep!r}")
    return keep


def _drop_polarity(region: np.ndarray, keep: str, name: str) -> np.ndarray:
    """Turn a True-in-region map into a True-means-drop mask.

    ``keep="inside"`` keeps the region, so the mask is True *outside*
    it; ``keep="outside"`` keeps everything but the region, so the mask
    is True *inside* it.
    """
    return ~region if _check_keep(keep, name) == "inside" else region


def combine_masks(
    masks: Sequence[Bool[np.ndarray, "*batch h w"]], op: str = "or"
) -> Bool[np.ndarray, "*batch h w"]:
    """Combine boolean masks element-wise with an n-ary logical operator.

    All masks must have the *same* shape — they are not broadcast (a
    ``(H, W)`` mask cannot be combined with a ``(C, H, W)`` stack; index
    or ``np.broadcast_to`` it first). Non-boolean inputs are coerced with
    ``np.asarray(mask, dtype=bool)``. Use :func:`invert_mask` for the
    unary complement.

    Args:
        masks: Non-empty sequence of equally shaped boolean masks.
        op: One of ``"or"``, ``"and"``, ``"xor"`` (reductions over the
            sequence). Case-insensitive.

    Returns:
        The combined boolean mask.

    Raises:
        ValueError: If ``masks`` is empty, the shapes differ, or ``op``
            is unknown.
    """
    if len(masks) == 0:
        raise ValueError("combine_masks: `masks` must not be empty")

    bool_masks = [np.asarray(mask, dtype=bool) for mask in masks]
    shapes = {mask.shape for mask in bool_masks}
    if len(shapes) > 1:
        raise ValueError(
            f"combine_masks: all masks must share one shape; got {sorted(shapes)}"
        )
    reducers = {
        "or": np.logical_or,
        "and": np.logical_and,
        "xor": np.logical_xor,
    }
    try:
        reducer = reducers[op.lower()]
    except KeyError:
        raise ValueError(
            "combine_masks: `op` must be one of 'or', 'and', 'xor'"
        ) from None
    return reducer.reduce(bool_masks)


def invert_mask(mask: Bool[np.ndarray, "*batch h w"]) -> Bool[np.ndarray, "*batch h w"]:
    """Invert a boolean mask.

    Args:
        mask: Boolean mask (non-boolean input is coerced to bool first).

    Returns:
        The element-wise complement ``~mask``.
    """
    return ~np.asarray(mask, dtype=bool)


def dilate_mask(
    mask: Bool[np.ndarray, "*batch h w"],
    iterations: int = 1,
    structure: Bool[np.ndarray, "sh sw"] | None = None,
) -> Bool[np.ndarray, "*batch h w"]:
    """Dilate a mask over its trailing two (spatial) axes.

    Wraps :func:`scipy.ndimage.binary_dilation`. Leading (batch / band)
    axes are processed independently, slice by slice.

    Args:
        mask: Boolean mask, at least 2-D; trailing axes are ``(H, W)``.
        iterations: Number of dilation passes. ``0`` returns a copy.
        structure: 2-D structuring element. Defaults to a 3x3 box
            (8-connectivity).

    Returns:
        The dilated boolean mask, same shape as ``mask``.

    Raises:
        ValueError: If ``iterations`` is negative, ``structure`` is not
            2-D, or ``mask`` has fewer than two dimensions.
    """
    return _apply_binary_morphology(
        mask, ndi.binary_dilation, iterations=iterations, structure=structure
    )


def erode_mask(
    mask: Bool[np.ndarray, "*batch h w"],
    iterations: int = 1,
    structure: Bool[np.ndarray, "sh sw"] | None = None,
) -> Bool[np.ndarray, "*batch h w"]:
    """Erode a mask over its trailing two (spatial) axes.

    Wraps :func:`scipy.ndimage.binary_erosion`; see :func:`dilate_mask`
    for the batching and structuring-element conventions.

    Args:
        mask: Boolean mask, at least 2-D; trailing axes are ``(H, W)``.
        iterations: Number of erosion passes. ``0`` returns a copy.
        structure: 2-D structuring element. Defaults to a 3x3 box
            (8-connectivity).

    Returns:
        The eroded boolean mask, same shape as ``mask``.

    Raises:
        ValueError: If ``iterations`` is negative, ``structure`` is not
            2-D, or ``mask`` has fewer than two dimensions.
    """
    return _apply_binary_morphology(
        mask, ndi.binary_erosion, iterations=iterations, structure=structure
    )


def open_mask(
    mask: Bool[np.ndarray, "*batch h w"], iterations: int = 1
) -> Bool[np.ndarray, "*batch h w"]:
    """Apply binary opening (erosion then dilation) over the spatial axes.

    Wraps :func:`scipy.ndimage.binary_opening` with a 3x3 box structuring
    element. Removes isolated True pixels (salt) while preserving the
    extent of large True components.

    Args:
        mask: Boolean mask, at least 2-D; trailing axes are ``(H, W)``.
        iterations: Number of opening passes. ``0`` returns a copy.

    Returns:
        The opened boolean mask, same shape as ``mask``.

    Raises:
        ValueError: If ``iterations`` is negative or ``mask`` has fewer
            than two dimensions.
    """
    return _apply_binary_morphology(mask, ndi.binary_opening, iterations=iterations)


def close_mask(
    mask: Bool[np.ndarray, "*batch h w"], iterations: int = 1
) -> Bool[np.ndarray, "*batch h w"]:
    """Apply binary closing (dilation then erosion) over the spatial axes.

    Wraps :func:`scipy.ndimage.binary_closing` with a 3x3 box structuring
    element. Fills pin-holes (pepper) inside otherwise solid True regions.

    Args:
        mask: Boolean mask, at least 2-D; trailing axes are ``(H, W)``.
        iterations: Number of closing passes. ``0`` returns a copy.

    Returns:
        The closed boolean mask, same shape as ``mask``.

    Raises:
        ValueError: If ``iterations`` is negative or ``mask`` has fewer
            than two dimensions.
    """
    return _apply_binary_morphology(mask, ndi.binary_closing, iterations=iterations)


def buffer_mask(
    mask: Bool[np.ndarray, "*batch h w"],
    radius: float,
    *,
    unit: str = "pixels",
    pixel_size: tuple[float, float] = (1.0, 1.0),
) -> Bool[np.ndarray, "*batch h w"]:
    """Radially expand True pixels by a Euclidean-distance buffer.

    A pixel is True in the output when its Euclidean distance to the
    nearest originally-True pixel is at most ``radius`` (so the original
    True pixels are always kept). Each leading (batch / band) slice is
    buffered independently.

    Unit contract:

    - ``unit="pixels"`` (default): ``radius`` is measured in pixels on
      the unit grid — every pixel is treated as 1 x 1 and ``pixel_size``
      is **ignored**.
    - ``unit="meters"`` (or ``"meter"``): ``radius`` is measured in the
      same linear units as ``pixel_size`` (metres for a projected CRS).
      ``pixel_size`` must then be the per-axis pixel extent
      ``(row_height, col_width)`` — i.e. ``(abs(yres), abs(xres))`` from
      the geotransform — and is passed as the ``sampling`` of the
      distance transform so anisotropic pixels buffer correctly.

    Args:
        mask: Boolean mask, at least 2-D; trailing axes are ``(H, W)``.
        radius: Buffer distance, in the units selected by ``unit``.
            ``0`` returns a copy of the input.
        unit: ``"pixels"``, ``"meters"``, or ``"meter"``.
        pixel_size: ``(row_height, col_width)`` pixel extent in CRS
            units; only used when ``unit`` is metres. Default
            ``(1.0, 1.0)``.

    Returns:
        The buffered boolean mask, same shape as ``mask``.

    Raises:
        ValueError: If ``radius`` is negative, ``unit`` is unknown, or
            ``mask`` has fewer than two dimensions.
    """
    if radius < 0:
        raise ValueError("buffer_mask: `radius` must be non-negative")
    if unit not in {"pixels", "meters", "meter"}:
        raise ValueError("buffer_mask: `unit` must be 'pixels', 'meter', or 'meters'")
    if radius == 0:
        return np.asarray(mask, dtype=bool).copy()

    sampling = (1.0, 1.0) if unit == "pixels" else pixel_size
    return _apply_spatial(mask, _buffer_2d, radius=radius, sampling=sampling)


def remove_small_objects(
    mask: Bool[np.ndarray, "*batch h w"],
    min_area_px: int,
    *,
    connectivity: Connectivity = 4,
) -> Bool[np.ndarray, "*batch h w"]:
    """Remove connected True components smaller than ``min_area_px`` pixels.

    Delegates to :func:`geotoolz.measure.label_components`. Each leading
    (batch / band) slice is cleaned independently.

    Args:
        mask: Boolean mask, at least 2-D; trailing axes are ``(H, W)``.
        min_area_px: Minimum component area, in pixels, for a component to
            be kept. ``0`` keeps everything.
        connectivity: ``4`` (default) or ``8`` neighbourhood.

    Returns:
        The cleaned boolean mask, same shape as ``mask``.

    Raises:
        ValueError: If ``min_area_px`` is negative or ``mask`` has fewer
            than two dimensions.
    """
    if min_area_px < 0:
        raise ValueError("remove_small_objects: `min_area_px` must be non-negative")
    return _apply_spatial(
        mask,
        _remove_small_objects_2d,
        min_area_px=min_area_px,
        connectivity=connectivity,
    )


def remove_small_holes(
    mask: Bool[np.ndarray, "*batch h w"],
    max_hole_area_px: int,
    *,
    connectivity: Connectivity = 4,
) -> Bool[np.ndarray, "*batch h w"]:
    """Fill enclosed False components up to ``max_hole_area_px`` pixels.

    A hole is a False component that does not touch the image border.
    Each leading (batch / band) slice is processed independently.

    Args:
        mask: Boolean mask, at least 2-D; trailing axes are ``(H, W)``.
        max_hole_area_px: Maximum hole area, in pixels, to fill. ``0``
            fills nothing.
        connectivity: ``4`` (default) or ``8`` neighbourhood used to
            group background pixels into holes.

    Returns:
        The filled boolean mask, same shape as ``mask``.

    Raises:
        ValueError: If ``max_hole_area_px`` is negative or ``mask`` has
            fewer than two dimensions.
    """
    if max_hole_area_px < 0:
        raise ValueError("remove_small_holes: `max_hole_area_px` must be non-negative")
    return _apply_spatial(
        mask,
        _remove_small_holes_2d,
        max_area=max_hole_area_px,
        connectivity=connectivity,
        exclude_border=True,
    )


def clean_mask(
    mask: Bool[np.ndarray, "*batch h w"],
    *,
    min_area_px: int = 25,
    max_hole_area_px: int = 25,
    close_iter: int = 1,
) -> Bool[np.ndarray, "*batch h w"]:
    """Remove small objects, fill small holes, then close the mask.

    Convenience composition of :func:`remove_small_objects`,
    :func:`remove_small_holes`, and :func:`close_mask`, in that order.

    Args:
        mask: Boolean mask, at least 2-D; trailing axes are ``(H, W)``.
        min_area_px: Components smaller than this many pixels are
            removed.
        max_hole_area_px: Enclosed holes up to this many pixels are filled.
        close_iter: Binary-closing iterations applied last. ``0`` skips
            the closing step.

    Returns:
        The cleaned boolean mask, same shape as ``mask``.

    Raises:
        ValueError: If any size/iteration argument is negative or
            ``mask`` has fewer than two dimensions.
    """
    out = remove_small_objects(mask, min_area_px)
    out = remove_small_holes(out, max_hole_area_px)
    return close_mask(out, close_iter)


def altitude_mask(
    dem: Float[np.ndarray, "h w"] | Float[np.ndarray, "1 h w"],
    *,
    min_elev: float | None = None,
    max_elev: float | None = None,
    keep: Keep = "inside",
) -> Bool[np.ndarray, "h w"]:
    """Mask DEM cells by an elevation interval (True = drop).

    Args:
        dem: Single-band elevation raster, ``(H, W)`` or ``(1, H, W)``.
        min_elev: Inclusive lower elevation bound (DEM units). ``None``
            leaves the interval open below.
        max_elev: Inclusive upper elevation bound. ``None`` leaves the
            interval open above.
        keep: ``"inside"`` (default) keeps cells inside
            ``[min_elev, max_elev]`` — the mask is True outside it;
            ``"outside"`` keeps cells outside the interval.

    Returns:
        Boolean ``(H, W)`` mask, True where the cell should be dropped.

    Raises:
        ValueError: If both bounds are ``None``, ``keep`` is unknown, or
            ``dem`` is not a single-band map.
    """
    if min_elev is None and max_elev is None:
        raise ValueError("altitude_mask: at least one elevation bound is required")
    arr = single_band(dem, name="altitude_mask")
    mask = np.ones(arr.shape, dtype=bool)
    if min_elev is not None:
        mask &= arr >= min_elev
    if max_elev is not None:
        mask &= arr <= max_elev
    return _drop_polarity(mask, keep, "altitude_mask")


def slope_degrees(
    dem: Float[np.ndarray, "h w"] | Float[np.ndarray, "1 h w"],
    pixel_size: tuple[float, float],
) -> Float[np.ndarray, "h w"]:
    """Compute slope in degrees from a single-band DEM.

    The gradient is estimated with central differences
    (:func:`numpy.gradient`) scaled by the pixel size; the slope is
    ``degrees(arctan(hypot(dz/dx, dz/dy)))``. Elevation and pixel size
    must be in the same linear units for the angles to be meaningful.

    Args:
        dem: Single-band elevation raster, ``(H, W)`` or ``(1, H, W)``.
        pixel_size: ``(row_height, col_width)`` pixel extent in the same
            units as the elevation values.

    Returns:
        Float ``(H, W)`` slope map in degrees, in ``[0, 90)``.

    Raises:
        ValueError: If ``dem`` is not a single-band map.
    """
    arr = single_band(dem, name="slope_degrees").astype(float, copy=False)
    yres, xres = pixel_size
    grad_y, grad_x = np.gradient(arr, yres, xres)
    return np.degrees(np.arctan(np.hypot(grad_x, grad_y)))


def slope_mask(
    dem: Float[np.ndarray, "h w"] | Float[np.ndarray, "1 h w"],
    pixel_size: tuple[float, float],
    *,
    min_slope_deg: float | None = None,
    max_slope_deg: float | None = None,
    keep: Keep = "inside",
) -> Bool[np.ndarray, "h w"]:
    """Mask DEM cells by a slope interval (True = drop).

    Slope is computed with :func:`slope_degrees`; see there for the
    pixel-size / unit contract.

    Args:
        dem: Single-band elevation raster, ``(H, W)`` or ``(1, H, W)``.
        pixel_size: ``(row_height, col_width)`` pixel extent in the same
            units as the elevation values.
        min_slope_deg: Inclusive lower slope bound in degrees. ``None``
            leaves the interval open below.
        max_slope_deg: Inclusive upper slope bound in degrees. ``None``
            leaves the interval open above.
        keep: ``"inside"`` (default) keeps cells whose slope lies inside
            ``[min_slope_deg, max_slope_deg]`` — the mask is True outside
            it; ``"outside"`` keeps cells outside the interval.

    Returns:
        Boolean ``(H, W)`` mask, True where the cell should be dropped.

    Raises:
        ValueError: If both bounds are ``None``, ``keep`` is unknown, or
            ``dem`` is not a single-band map.
    """
    if min_slope_deg is None and max_slope_deg is None:
        raise ValueError("slope_mask: at least one slope bound is required")
    slope = slope_degrees(dem, pixel_size)
    mask = np.ones(slope.shape, dtype=bool)
    if min_slope_deg is not None:
        mask &= slope >= min_slope_deg
    if max_slope_deg is not None:
        mask &= slope <= max_slope_deg
    return _drop_polarity(mask, keep, "slope_mask")


def distance_mask(
    geometry_mask: Bool[np.ndarray, "h w"],
    distance: float,
    *,
    keep: Keep = "inside",
    pixel_size: tuple[float, float] = (1.0, 1.0),
) -> Bool[np.ndarray, "h w"]:
    """Mask pixels by their distance to an already-rasterized geometry.

    The zone within ``distance`` of the geometry is
    :func:`buffer_mask` of ``geometry_mask`` in ``pixel_size`` units —
    leave the default ``(1.0, 1.0)`` for pixel units, or pass
    ``(abs(yres), abs(xres))`` for CRS units. The result follows the
    package polarity (True = drop).

    Args:
        geometry_mask: Boolean ``(H, W)`` mask, True on the geometry.
        distance: Maximum distance from the geometry. Pixels on the
            geometry itself are at distance ``0``.
        keep: ``"inside"`` (default) keeps pixels within ``distance`` —
            the mask is True beyond it; ``"outside"`` keeps pixels beyond
            ``distance`` and drops the zone around the geometry.
        pixel_size: ``(row_height, col_width)`` distance sampling.

    Returns:
        Boolean ``(H, W)`` mask, True where the pixel should be dropped.

    Raises:
        ValueError: If ``distance`` is negative or ``keep`` is unknown.
    """
    if distance < 0:
        raise ValueError("distance_mask: `distance` must be non-negative")
    _check_keep(keep, "distance_mask")
    within = buffer_mask(
        np.asarray(geometry_mask, dtype=bool),
        distance,
        unit="meters",
        pixel_size=pixel_size,
    )
    return _drop_polarity(within, keep, "distance_mask")


def _apply_binary_morphology(
    mask: Bool[np.ndarray, "*batch h w"],
    func: Callable[..., np.ndarray],
    *,
    iterations: int = 1,
    structure: Bool[np.ndarray, "sh sw"] | None = None,
) -> Bool[np.ndarray, "*batch h w"]:
    if iterations < 0:
        raise ValueError("morphology iterations must be non-negative")
    if structure is None:
        structure = np.ones((3, 3), dtype=bool)
    else:
        structure = np.asarray(structure, dtype=bool)
        if structure.ndim != 2:
            raise ValueError("morphology `structure` must be 2D")
    if iterations == 0:
        return np.asarray(mask, dtype=bool).copy()
    return _apply_spatial(mask, func, iterations=iterations, structure=structure)


def _apply_spatial(
    mask: Bool[np.ndarray, "*batch h w"],
    func: Callable[..., np.ndarray],
    **kwargs: object,
) -> Bool[np.ndarray, "*batch h w"]:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim < 2:
        raise ValueError("mask operations require at least two spatial dimensions")
    if arr.ndim == 2:
        return func(arr, **kwargs)
    out = np.empty_like(arr, dtype=bool)
    for idx in np.ndindex(arr.shape[:-2]):
        out[idx] = func(arr[idx], **kwargs)
    return out


def _buffer_2d(
    mask: Bool[np.ndarray, "h w"],
    *,
    radius: float,
    sampling: tuple[float, float],
) -> Bool[np.ndarray, "h w"]:
    dist = ndi.distance_transform_edt(~mask, sampling=sampling)
    return dist <= radius


def _remove_small_objects_2d(
    mask: Bool[np.ndarray, "h w"], *, min_area_px: int, connectivity: Connectivity
) -> Bool[np.ndarray, "h w"]:
    return (
        label_components(mask, connectivity=connectivity, min_area_px=min_area_px) > 0
    )


def apply_mask(
    arr: Shaped[np.ndarray, "*dims"],
    mask: Bool[np.ndarray, "*mask_dims"],
    fill_value: float = np.nan,
) -> Shaped[np.ndarray, "*dims"]:
    """Apply a boolean mask to a multi-band array, filling masked pixels.

    The mask follows the package polarity: True where pixels should be
    *masked out* (dropped) — see "Mask polarity" in ``docs/concepts.md``.
    The result is ``arr`` with ``fill_value`` substituted wherever
    ``mask`` is True. Flip a keep-polarity mask with :func:`invert_mask`
    first.

    The mask broadcasts against the spatial trailing axes of ``arr``,
    so a ``(H, W)`` mask applies to every band of a ``(C, H, W)``
    raster automatically.

    Args:
        arr: Input array, any shape.
        mask: Boolean mask. Must broadcast against the spatial
            (trailing two) axes of ``arr``.
        fill_value: Value substituted where the mask says "drop".
            Default ``np.nan`` (the right choice for float arrays;
            switch to a sentinel like ``0`` for integer inputs).

    Returns:
        Array of the same shape as ``arr``, masked pixels replaced
        with ``fill_value``.
    """
    bool_mask = np.asarray(mask, dtype=bool)
    # `np.where` upcasts to the wider dtype of (fill_value, arr). For
    # floating arrays we want to preserve `arr.dtype` — otherwise
    # `fill_value=np.nan` (float64) silently doubles memory on float32
    # inputs and undoes any upstream `ToFloat32()` stage. Cast the fill
    # to arr's dtype when arr is floating; integer arrays keep numpy's
    # native promotion (so `fill_value=np.nan` on int input still
    # upcasts, which is the only sensible behaviour).
    if np.issubdtype(arr.dtype, np.floating):
        fill_typed = np.asarray(fill_value, dtype=arr.dtype)
        return np.where(bool_mask, fill_typed, arr)
    return np.where(bool_mask, fill_value, arr)
