"""Tier-A primitives for geometry operators — pure-numpy helpers.

The carrier-aware Operator wrappers in
:mod:`geotoolz.geom._src.operators` lift these into the ``GeoTensor``
pipeline. Each helper here is deliberately framework-agnostic: it
operates on ``numpy.ndarray`` (and tuples of plain ints / floats) so the
same primitive is callable from notebooks and tests without constructing
a ``GeoTensor``.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from rasterio.enums import Resampling


# Aliases between the user-friendly resampling names and the
# rasterio / skimage naming conventions used downstream.
_RASTERIO_ALIASES: dict[str, str] = {"bicubic": "cubic", "linear": "bilinear"}
_SKIMAGE_ALIASES: dict[str, str] = {
    "cubic": "bicubic",
    "cubic_spline": "bicubic",
    "average": "bilinear",
}


def resolve_resampling(name: str | Resampling) -> Resampling:
    """Translate a string or `Resampling` to the canonical `rasterio` enum.

    Accepts the user-friendly aliases ``"linear"`` (= ``"bilinear"``) and
    ``"bicubic"`` (= ``"cubic"``) in addition to all
    :class:`rasterio.enums.Resampling` members.

    Args:
        name: Either a `Resampling` enum value or a string alias.

    Returns:
        The matching :class:`rasterio.enums.Resampling` enum member.
    """
    if isinstance(name, Resampling):
        return name
    return Resampling[_RASTERIO_ALIASES.get(name, name)]


def resolve_interpolation(name: str) -> str:
    """Translate a resampling name into a ``skimage.transform.resize`` mode.

    ``GeoTensor.resize`` delegates to ``skimage.transform.resize`` whose
    ``interpolation`` parameter uses different names than rasterio
    (``"bicubic"`` instead of ``"cubic"``, etc.). This helper keeps the
    geom operators idiomatic from the user's side while passing the
    correct names downstream.

    Args:
        name: Resampling alias.

    Returns:
        The matching ``skimage`` interpolation name.
    """
    return _SKIMAGE_ALIASES.get(name, name)


def center_offsets(
    current: tuple[int, int], target: tuple[int, int]
) -> tuple[int, int]:
    """Pixel offsets to center-anchor a ``target`` shape inside ``current``.

    Returns ``((current_h - target_h) // 2, (current_w - target_w) // 2)``,
    the row/col offsets used by `CropTo(anchor="center")`.

    Args:
        current: Source spatial shape ``(H, W)``.
        target: Target spatial shape ``(H', W')``.

    Returns:
        ``(row_off, col_off)`` integer pair.
    """
    return ((current[0] - target[0]) // 2, (current[1] - target[1]) // 2)


def is_north_up(transform: Affine) -> bool:
    """Return ``True`` iff the affine transform is axis-aligned, north-up.

    Equivalent to ``b == 0 and d == 0`` on
    :class:`affine.Affine` — no rotation or shear. North-up transforms
    are the only ones that admit pixel-aligned mosaicking with the
    `Stitch` operator.

    Args:
        transform: An :class:`affine.Affine` transform.

    Returns:
        ``True`` if the transform is axis-aligned.
    """
    return transform.b == 0 and transform.d == 0


def feather_weights(shape: tuple[int, int], width: int) -> np.ndarray:
    r"""Edge-feathered weight kernel for tile blending.

    Builds a 2-D weight array where each pixel's weight is the minimum
    of:

    .. math::

        w(i, j) \;=\; \min\!\left(
            1,\,
            \frac{\min(i + 1,\,H - i)}{W_f},\,
            \frac{\min(j + 1,\,W - j)}{W_f}
        \right)

    so the centre of the tile is at full weight ``1`` and a band of
    ``width`` pixels along each edge ramps linearly down to
    ``1 / W_f``. Used by ``Stitch(blend="feather")``.

    Args:
        shape: Tile spatial shape ``(H, W)``.
        width: Feather band width in pixels. ``<= 0`` returns
            ``np.ones(shape)``.

    Returns:
        ``float32`` array of shape ``(H, W)`` with values in ``[0, 1]``.
    """
    height, width_px = shape
    if width <= 0:
        return np.ones(shape, dtype=np.float32)
    y = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1))
    x = np.minimum(np.arange(width_px) + 1, np.arange(width_px, 0, -1))
    y = np.clip(y / width, 0.0, 1.0)
    x = np.clip(x / width, 0.0, 1.0)
    return np.outer(y, x).astype(np.float32)


def target_slices(
    tile_transform: Affine,
    tile_shape: tuple[int, int],
    target_transform: Affine,
    target_shape: tuple[int, int],
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Project a tile onto a target grid as ``(out_slice, tile_slice)``.

    Computes the pixel-space slices into ``target`` and ``tile`` arrays
    that overlap. Returns empty slices when the tile lies fully outside
    the target. Both transforms are assumed north-up (call
    :func:`is_north_up` first).

    Args:
        tile_transform: Affine of the tile.
        tile_shape: Tile spatial shape ``(h, w)``.
        target_transform: Affine of the target mosaic.
        target_shape: Target spatial shape ``(H, W)``.

    Returns:
        ``((row_slice, col_slice), (tile_row_slice, tile_col_slice))``.
    """
    row = round((tile_transform.f - target_transform.f) / target_transform.e)
    col = round((tile_transform.c - target_transform.c) / target_transform.a)
    height, width = tile_shape
    row0 = max(row, 0)
    col0 = max(col, 0)
    row1 = min(row + height, target_shape[0])
    col1 = min(col + width, target_shape[1])
    tile_row0 = row0 - row
    tile_col0 = col0 - col
    tile_row1 = tile_row0 + (row1 - row0)
    tile_col1 = tile_col0 + (col1 - col0)
    return (
        (slice(row0, row1), slice(col0, col1)),
        (slice(tile_row0, tile_row1), slice(tile_col0, tile_col1)),
    )


def valid_pixel_mask(tile: np.ndarray, fill: float | int | None) -> np.ndarray:
    """Boolean mask of pixels not equal to ``fill``, collapsed over band axis.

    Tiles produced by ``Tile(include_incomplete=True, boundless=True)``
    pad the right/bottom edges with the carrier's ``fill_value_default``;
    blending must not aggregate those sentinel pixels into the output.

    Args:
        tile: Numpy view of the tile values, shape ``(..., H, W)``.
        fill: Sentinel value. ``None`` short-circuits to all-True.

    Returns:
        Boolean ``(H, W)`` mask where ``True`` means "real data".
    """
    if fill is None:
        return np.ones(tile.shape[-2:], dtype=bool)
    if np.issubdtype(tile.dtype, np.floating) and np.isnan(fill):
        per_band = ~np.isnan(tile)
    else:
        per_band = tile != fill
    if per_band.ndim == 2:
        return per_band
    return per_band.reshape(-1, *per_band.shape[-2:]).any(axis=0)
