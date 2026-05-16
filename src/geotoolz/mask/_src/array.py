"""Pure-numpy helpers for geometry, morphology, and algebra masks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.ndimage as ndi


def combine_masks(masks: Sequence[np.ndarray], op: str = "or") -> np.ndarray:
    """Combine boolean masks with a small algebra."""
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


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Invert a boolean mask."""
    return ~np.asarray(mask, dtype=bool)


def dilate_mask(
    mask: np.ndarray,
    iterations: int = 1,
    structure: np.ndarray | None = None,
) -> np.ndarray:
    """Dilate a mask over its trailing spatial axes."""
    return _apply_binary_morphology(
        mask, ndi.binary_dilation, iterations=iterations, structure=structure
    )


def erode_mask(
    mask: np.ndarray,
    iterations: int = 1,
    structure: np.ndarray | None = None,
) -> np.ndarray:
    """Erode a mask over its trailing spatial axes."""
    return _apply_binary_morphology(
        mask, ndi.binary_erosion, iterations=iterations, structure=structure
    )


def open_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply binary opening to a mask over its trailing spatial axes."""
    return _apply_binary_morphology(mask, ndi.binary_opening, iterations=iterations)


def close_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply binary closing to a mask over its trailing spatial axes."""
    return _apply_binary_morphology(mask, ndi.binary_closing, iterations=iterations)


def buffer_mask(
    mask: np.ndarray,
    radius: float,
    *,
    unit: str = "pixels",
    pixel_size: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Radially expand True pixels by ``radius`` pixels or CRS units."""
    if radius < 0:
        raise ValueError("buffer_mask: `radius` must be non-negative")
    if unit not in {"pixels", "meters", "meter"}:
        raise ValueError("buffer_mask: `unit` must be 'pixels' or 'meters'")
    if radius == 0:
        return np.asarray(mask, dtype=bool).copy()

    sampling = (1.0, 1.0) if unit == "pixels" else pixel_size
    return _apply_spatial(mask, _buffer_2d, radius=radius, sampling=sampling)


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected True components smaller than ``min_size`` pixels."""
    if min_size < 0:
        raise ValueError("remove_small_objects: `min_size` must be non-negative")
    return _apply_spatial(mask, _remove_small_objects_2d, min_size=min_size)


def remove_small_holes(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    """Fill enclosed False components up to ``area_threshold`` pixels."""
    if area_threshold < 0:
        raise ValueError("remove_small_holes: `area_threshold` must be non-negative")
    return _apply_spatial(mask, _remove_small_holes_2d, area_threshold=area_threshold)


def clean_mask(
    mask: np.ndarray,
    *,
    min_object_size: int = 25,
    max_hole_size: int = 25,
    close_iter: int = 1,
) -> np.ndarray:
    """Remove small objects, fill small holes, then close the mask."""
    out = remove_small_objects(mask, min_object_size)
    out = remove_small_holes(out, max_hole_size)
    return close_mask(out, close_iter)


def altitude_mask(
    dem: np.ndarray,
    *,
    min_elev: float | None = None,
    max_elev: float | None = None,
) -> np.ndarray:
    """Mask DEM cells inside the requested elevation interval."""
    if min_elev is None and max_elev is None:
        raise ValueError("altitude_mask: at least one elevation bound is required")
    arr = _squeeze_single_band(dem)
    mask = np.ones(arr.shape, dtype=bool)
    if min_elev is not None:
        mask &= arr >= min_elev
    if max_elev is not None:
        mask &= arr <= max_elev
    return mask


def slope_degrees(dem: np.ndarray, pixel_size: tuple[float, float]) -> np.ndarray:
    """Compute slope in degrees from a single-band DEM."""
    arr = _squeeze_single_band(dem).astype(float, copy=False)
    yres, xres = pixel_size
    grad_y, grad_x = np.gradient(arr, yres, xres)
    return np.degrees(np.arctan(np.hypot(grad_x, grad_y)))


def slope_mask(
    dem: np.ndarray,
    pixel_size: tuple[float, float],
    *,
    min_slope_deg: float | None = None,
    max_slope_deg: float | None = None,
) -> np.ndarray:
    """Mask DEM cells inside the requested slope interval."""
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
    geometry_mask: np.ndarray,
    distance: float,
    *,
    inside: bool = True,
    pixel_size: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Mask pixels within ``distance`` of an already-rasterized geometry."""
    if distance < 0:
        raise ValueError("distance_mask: `distance` must be non-negative")
    base = np.asarray(geometry_mask, dtype=bool)
    dist = ndi.distance_transform_edt(~base, sampling=pixel_size)
    out = dist <= distance
    return out if inside else ~out


def _apply_binary_morphology(
    mask: np.ndarray,
    func: Callable[..., np.ndarray],
    *,
    iterations: int = 1,
    structure: np.ndarray | None = None,
) -> np.ndarray:
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
    mask: np.ndarray, func: Callable[..., np.ndarray], **kwargs: object
) -> np.ndarray:
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
    mask: np.ndarray,
    *,
    radius: float,
    sampling: tuple[float, float],
) -> np.ndarray:
    dist = ndi.distance_transform_edt(~mask, sampling=sampling)
    return dist <= radius


def _remove_small_objects_2d(mask: np.ndarray, *, min_size: int) -> np.ndarray:
    labels, num = ndi.label(mask)
    if num == 0 or min_size == 0:
        return mask.copy()
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def _remove_small_holes_2d(mask: np.ndarray, *, area_threshold: int) -> np.ndarray:
    inv = ~mask
    labels, num = ndi.label(inv)
    if num == 0 or area_threshold == 0:
        return mask.copy()

    border_labels = set(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[-1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, -1]))

    sizes = np.bincount(labels.ravel())
    fill = np.zeros(num + 1, dtype=bool)
    for label in range(1, num + 1):
        fill[label] = label not in border_labels and sizes[label] <= area_threshold
    return mask | fill[labels]


def _squeeze_single_band(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 2:
        return out
    if out.ndim == 3 and out.shape[0] == 1:
        return out[0]
    raise ValueError("DEM mask helpers expect a 2D array or single-band stack")
