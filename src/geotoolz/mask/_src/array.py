"""Pure-numpy helpers for geometry, morphology, and algebra masks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.ndimage as ndi
from jaxtyping import Bool, Float

from geotoolz._src.shape import single_band


def combine_masks(
    masks: Sequence[Bool[np.ndarray, "*batch h w"]], op: str = "or"
) -> Bool[np.ndarray, "*batch h w"]:
    """Combine boolean masks element-wise with a logical operator.

    All masks must share (or broadcast to) a common shape. Non-boolean
    inputs are coerced with ``np.asarray(mask, dtype=bool)``.

    Args:
        masks: Non-empty sequence of boolean masks.
        op: One of ``"or"``, ``"and"``, ``"xor"`` (n-ary reductions over
            the sequence), or the unary ``"not"`` which expects exactly
            one mask and returns its complement. Case-insensitive.

    Returns:
        The combined boolean mask.

    Raises:
        ValueError: If ``masks`` is empty, ``op`` is unknown, or
            ``op='not'`` receives more than one mask.
    """
    if len(masks) == 0:
        raise ValueError("combine_masks: `masks` must not be empty")

    bool_masks = [np.asarray(mask, dtype=bool) for mask in masks]
    op_norm = op.lower()
    if op_norm == "or":
        return np.logical_or.reduce(bool_masks)
    if op_norm == "and":
        return np.logical_and.reduce(bool_masks)
    if op_norm == "xor":
        return np.logical_xor.reduce(bool_masks)
    if op_norm == "not":
        if len(bool_masks) != 1:
            raise ValueError("combine_masks: op='not' expects exactly one mask")
        return ~bool_masks[0]
    raise ValueError("combine_masks: `op` must be one of 'or', 'and', 'xor', 'not'")


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
    mask: Bool[np.ndarray, "*batch h w"], min_size: int
) -> Bool[np.ndarray, "*batch h w"]:
    """Remove connected True components smaller than ``min_size`` pixels.

    Connectivity is 4-neighbour (:func:`scipy.ndimage.label` default).
    Each leading (batch / band) slice is cleaned independently.

    Args:
        mask: Boolean mask, at least 2-D; trailing axes are ``(H, W)``.
        min_size: Minimum component area, in pixels, for a component to
            be kept. ``0`` keeps everything.

    Returns:
        The cleaned boolean mask, same shape as ``mask``.

    Raises:
        ValueError: If ``min_size`` is negative or ``mask`` has fewer
            than two dimensions.
    """
    if min_size < 0:
        raise ValueError("remove_small_objects: `min_size` must be non-negative")
    return _apply_spatial(mask, _remove_small_objects_2d, min_size=min_size)


def remove_small_holes(
    mask: Bool[np.ndarray, "*batch h w"], area_threshold: int
) -> Bool[np.ndarray, "*batch h w"]:
    """Fill enclosed False components up to ``area_threshold`` pixels.

    A hole is a False component that does not touch the image border.
    Each leading (batch / band) slice is processed independently.

    Args:
        mask: Boolean mask, at least 2-D; trailing axes are ``(H, W)``.
        area_threshold: Maximum hole area, in pixels, to fill. ``0``
            fills nothing.

    Returns:
        The filled boolean mask, same shape as ``mask``.

    Raises:
        ValueError: If ``area_threshold`` is negative or ``mask`` has
            fewer than two dimensions.
    """
    if area_threshold < 0:
        raise ValueError("remove_small_holes: `area_threshold` must be non-negative")
    return _apply_spatial(mask, _remove_small_holes_2d, area_threshold=area_threshold)


def clean_mask(
    mask: Bool[np.ndarray, "*batch h w"],
    *,
    min_object_size: int = 25,
    max_hole_size: int = 25,
    close_iter: int = 1,
) -> Bool[np.ndarray, "*batch h w"]:
    """Remove small objects, fill small holes, then close the mask.

    Convenience composition of :func:`remove_small_objects`,
    :func:`remove_small_holes`, and :func:`close_mask`, in that order.

    Args:
        mask: Boolean mask, at least 2-D; trailing axes are ``(H, W)``.
        min_object_size: Components smaller than this many pixels are
            removed.
        max_hole_size: Enclosed holes up to this many pixels are filled.
        close_iter: Binary-closing iterations applied last. ``0`` skips
            the closing step.

    Returns:
        The cleaned boolean mask, same shape as ``mask``.

    Raises:
        ValueError: If any size/iteration argument is negative or
            ``mask`` has fewer than two dimensions.
    """
    out = remove_small_objects(mask, min_object_size)
    out = remove_small_holes(out, max_hole_size)
    return close_mask(out, close_iter)


def altitude_mask(
    dem: Float[np.ndarray, "h w"] | Float[np.ndarray, "1 h w"],
    *,
    min_elev: float | None = None,
    max_elev: float | None = None,
) -> Bool[np.ndarray, "h w"]:
    """Mask DEM cells inside the requested elevation interval.

    Args:
        dem: Single-band elevation raster, ``(H, W)`` or ``(1, H, W)``.
        min_elev: Inclusive lower elevation bound (DEM units). ``None``
            leaves the interval open below.
        max_elev: Inclusive upper elevation bound. ``None`` leaves the
            interval open above.

    Returns:
        Boolean ``(H, W)`` mask, True where the elevation lies inside
        ``[min_elev, max_elev]``.

    Raises:
        ValueError: If both bounds are ``None`` or ``dem`` is not a
            single-band map.
    """
    if min_elev is None and max_elev is None:
        raise ValueError("altitude_mask: at least one elevation bound is required")
    arr = single_band(dem, name="altitude_mask")
    mask = np.ones(arr.shape, dtype=bool)
    if min_elev is not None:
        mask &= arr >= min_elev
    if max_elev is not None:
        mask &= arr <= max_elev
    return mask


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
) -> Bool[np.ndarray, "h w"]:
    """Mask DEM cells inside the requested slope interval.

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

    Returns:
        Boolean ``(H, W)`` mask, True where the slope lies inside
        ``[min_slope_deg, max_slope_deg]``.

    Raises:
        ValueError: If both bounds are ``None`` or ``dem`` is not a
            single-band map.
    """
    if min_slope_deg is None and max_slope_deg is None:
        raise ValueError("slope_mask: at least one slope bound is required")
    slope = slope_degrees(dem, pixel_size)
    mask = np.ones(slope.shape, dtype=bool)
    if min_slope_deg is not None:
        mask &= slope >= min_slope_deg
    if max_slope_deg is not None:
        mask &= slope <= max_slope_deg
    return mask


def distance_mask(
    geometry_mask: Bool[np.ndarray, "h w"],
    distance: float,
    *,
    inside: bool = True,
    pixel_size: tuple[float, float] = (1.0, 1.0),
) -> Bool[np.ndarray, "h w"]:
    """Mask pixels within ``distance`` of an already-rasterized geometry.

    Computes the Euclidean distance transform from the True pixels of
    ``geometry_mask`` and thresholds it at ``distance``. As with
    :func:`buffer_mask`, ``distance`` is measured in the units of
    ``pixel_size`` — leave the default ``(1.0, 1.0)`` for pixel units,
    or pass ``(abs(yres), abs(xres))`` for CRS units.

    Args:
        geometry_mask: Boolean ``(H, W)`` mask, True on the geometry.
        distance: Maximum distance from the geometry. Pixels on the
            geometry itself are at distance ``0``.
        inside: If True (default), mark pixels within ``distance``;
            if False, return the complement.
        pixel_size: ``(row_height, col_width)`` distance sampling.

    Returns:
        Boolean ``(H, W)`` mask.

    Raises:
        ValueError: If ``distance`` is negative.
    """
    if distance < 0:
        raise ValueError("distance_mask: `distance` must be non-negative")
    base = np.asarray(geometry_mask, dtype=bool)
    dist = ndi.distance_transform_edt(~base, sampling=pixel_size)
    out = dist <= distance
    return out if inside else ~out


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
    mask: Bool[np.ndarray, "h w"], *, min_size: int
) -> Bool[np.ndarray, "h w"]:
    labels, num = ndi.label(mask)
    if num == 0 or min_size == 0:
        return mask.copy()
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def _remove_small_holes_2d(
    mask: Bool[np.ndarray, "h w"], *, area_threshold: int
) -> Bool[np.ndarray, "h w"]:
    inv = ~mask
    labels, num = ndi.label(inv)
    if num == 0 or area_threshold == 0:
        return mask.copy()

    border_labels = set(
        np.unique(
            np.concatenate(
                [labels[0, :], labels[-1, :], labels[1:-1, 0], labels[1:-1, -1]]
            )
        )
    )

    sizes = np.bincount(labels.ravel())
    fill = np.zeros(num + 1, dtype=bool)
    for label in range(1, num + 1):
        fill[label] = label not in border_labels and sizes[label] <= area_threshold
    return mask | fill[labels]
