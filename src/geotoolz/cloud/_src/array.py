"""Tier-A primitives — pure-numpy cloud / mask helpers.

Two families:

1. **Mask extraction** — derive a boolean ``(H, W)`` mask from a QA /
   classification band.
   - `mask_from_qa_bits` for Landsat-style bitmask QA layers
     (``QA_PIXEL``).
   - `mask_from_scl` for Sentinel-2 SCL class-label bands.
2. **Mask application** — apply a boolean mask to a multi-band raster.
   - `apply_mask` does the broadcast + fill in one call.

The wrappers in `geotoolz.cloud._src.operators` consume these
primitives and produce `GeoTensor`s with metadata preserved.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mask_from_qa_bits(
    qa: np.ndarray,
    bits: Sequence[int],
    *,
    invert: bool = False,
) -> np.ndarray:
    r"""Decode a Landsat-style bitmask QA band.

    Returns a boolean array where any of the supplied ``bits`` is set
    in the QA value. For Landsat-8 ``QA_PIXEL`` the bit assignments
    (Collection-2 Level-2) are:

    ===== ==========================
    Bit   Meaning
    ===== ==========================
    0     Fill / no-data
    1     Dilated Cloud
    2     Cirrus (high confidence)
    3     Cloud
    4     Cloud Shadow
    5     Snow
    6     Clear
    7     Water
    ===== ==========================

    So ``mask_from_qa_bits(qa, bits=[3, 4])`` returns True where the
    pixel is *either* cloudy *or* in cloud shadow.

    The math is a single ``(qa & bitmask) != 0`` per bit, OR-ed
    together — vectorised over the whole array.

    Args:
        qa: Integer QA array (any integer dtype). Typically
            ``(H, W)``; works for arbitrary shape.
        bits: Sequence of bit positions to test. The function returns
            True where ANY of these is set.
        invert: When True, flip the result. Useful for "clear" masks
            (``mask_from_qa_bits(qa, [3,4,5], invert=True)``).

    Returns:
        Boolean array of the same shape as ``qa``.

    References:
        USGS, "Landsat 8-9 Collection 2 Level-2 Science Product
        Guide", LSDS-1619, 2022.
    """
    qa_int = qa.astype(np.int64, copy=False)
    bitmask = 0
    for b in bits:
        if b < 0:
            raise ValueError(f"bit position must be non-negative; got {b}")
        bitmask |= 1 << int(b)
    out = (qa_int & bitmask) != 0
    return ~out if invert else out


def mask_from_scl(
    scl: np.ndarray,
    classes: Sequence[int],
    *,
    invert: bool = False,
) -> np.ndarray:
    """Decode a Sentinel-2 SCL band by class membership.

    Returns True where the SCL value equals any of the listed
    ``classes``. With ``invert=True``, returns True where the SCL value
    is *not* in the list — handy for "keep only these classes" masks.

    SCL class IDs come from `geotoolz.cloud._src.scl.SCL` (an IntEnum).
    Mixing raw ints and enum members is fine — they're cast to
    ``int`` here.

    Args:
        scl: SCL band, integer dtype, typically ``(H, W)`` or
            ``(1, H, W)``.
        classes: SCL class IDs to match (raw ints or `SCL` members).
        invert: When True, return True where SCL value is NOT in
            ``classes`` (keep-only-these mask).

    Returns:
        Boolean array of the same shape as ``scl``.
    """
    if len(classes) == 0:
        raise ValueError("mask_from_scl: `classes` must not be empty")
    # `isin` is the vectorised any-of comparison.
    out = np.isin(scl, np.asarray([int(c) for c in classes]))
    return ~out if invert else out


def apply_mask(
    arr: np.ndarray,
    mask: np.ndarray,
    fill_value: float = np.nan,
    *,
    invert: bool = False,
) -> np.ndarray:
    """Apply a boolean mask to a multi-band array, filling masked pixels.

    By convention here, the mask is True where pixels should be
    *masked out* — `mask_from_qa_bits` and `mask_from_scl` already
    follow that convention ("True = cloudy"). The result is
    ``arr`` with ``fill_value`` substituted wherever ``mask`` is True
    (or where ``~mask`` is True if ``invert=True``).

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
        invert: When True, fill where the mask is False instead of
            True — i.e. treat the mask as "keep-only" rather than
            "mask-out".

    Returns:
        Array of the same shape as ``arr``, masked pixels replaced
        with ``fill_value``.
    """
    bool_mask = np.asarray(mask, dtype=bool)
    if invert:
        bool_mask = ~bool_mask
    # `where(condition, x, y)` returns x where condition is True; we want
    # `fill` where the mask says "drop", so condition is `~mask`.
    return np.where(bool_mask, fill_value, arr)
