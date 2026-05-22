"""Shared overlap-add blending primitives."""

from __future__ import annotations

import numpy as np


def triangular_weights(shape: tuple[int, ...], width: int) -> np.ndarray:
    r"""Linear-ramp edge taper for overlap-add blending.

    Builds an N-D weight array where each axis ramps linearly from the
    edge up to 1.0 over ``width`` pixels, then combines axes by product.
    ``width <= 0`` returns a boxcar window of ones.

    Args:
        shape: Spatial shape of the window.
        width: Ramp width in pixels.

    Returns:
        ``float32`` array of shape ``shape`` with values in ``[0, 1]``.
    """
    if width <= 0:
        return np.ones(shape, dtype=np.float32)
    axes = []
    for size in shape:
        distances = np.minimum(np.arange(size) + 1, np.arange(size, 0, -1))
        axes.append(np.clip(distances / width, 0.0, 1.0))
    weights = axes[0]
    for axis in axes[1:]:
        weights = np.multiply.outer(weights, axis)
    return weights.astype(np.float32)


def overlap_add(
    values: np.ndarray,
    weights: np.ndarray,
    tile: np.ndarray,
    out_slices: tuple[slice, slice],
    tile_slices: tuple[slice, slice],
    kernel: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> None:
    """Accumulate one weighted tile into overlap-add buffers.

    Args:
        values: Output value accumulator with shape ``(..., H, W)``.
            Modified in place.
        weights: Output weight accumulator with shape ``(H, W)``.
            Modified in place.
        tile: Tile values with shape ``(..., h, w)``.
        out_slices: Row/column slices selecting the tile footprint in
            ``values`` and ``weights``.
        tile_slices: Row/column slices selecting the overlapping region
            in ``tile``, ``kernel``, and ``valid_mask``.
        kernel: Per-tile spatial weights with shape ``(h, w)``.
        valid_mask: Optional boolean mask with shape ``(h, w)`` where
            ``False`` pixels contribute zero weight.
    """
    weight = kernel[tile_slices]
    if valid_mask is not None:
        weight = weight * valid_mask[tile_slices].astype(np.float32)
    values[..., out_slices[0], out_slices[1]] += (
        tile[..., tile_slices[0], tile_slices[1]] * weight
    )
    weights[out_slices] += weight


def normalize_overlap_add(
    values: np.ndarray,
    weights: np.ndarray,
    fill: float | int | None,
) -> None:
    """Normalize accumulated overlap-add buffers in place.

    Pixels with positive accumulated weight are divided by that weight.
    Pixels that never received a valid contribution are set to ``fill``.

    Args:
        values: Weighted-sum accumulator with shape ``(..., H, W)``.
            Modified in place.
        weights: Accumulated spatial weights with shape ``(H, W)``.
        fill: Value assigned to pixels where ``weights == 0``.
    """
    valid = weights > 0
    values[..., valid] /= weights[valid]
    values[..., ~valid] = fill
