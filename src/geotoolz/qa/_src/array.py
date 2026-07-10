"""Tier-A primitives — pure-numpy QA / cloud-mask decoding.

The canonical home for QA-band decoding (these functions used to be
split between here and the retired ``geotoolz.cloud`` module):

1. **`mask_from_qa_bits`** — single-bit-flag decoding for Landsat-style
   bitmask QA layers (``QA_PIXEL``): True where ANY listed bit is set.
2. **`mask_from_scl`** — Sentinel-2 SCL class-membership decoding.
3. **`mask_from_bit_field`** — extract a multi-bit field (e.g. MODIS
   ``state_1km`` bits ``[0, 1]`` which encode a 2-bit cloud state) and
   test membership against a set of integer field-values. Different from
   ``mask_from_qa_bits`` because that helper OR-s individual bits, which
   is wrong when the bits form a single contiguous categorical field.
4. **`reduce_bit_masks`** — combine several bit-position groups (the
   shape used by the sensor presets in this module) into a single mask
   by OR-ing their `mask_from_qa_bits` results.

Mask *application* lives in `geotoolz.mask` (`apply_mask` /
`ApplyMask`).

References:
    USGS, "Landsat 8-9 Collection 2 Level-2 Science Product Guide",
    LSDS-1619, 2022.
    USGS, "Landsat 4-7 Collection 2 Level-2 Science Product Guide",
    LSDS-1618, 2022.
    MODIS Surface Reflectance User's Guide (MOD09 / MYD09),
    Vermote, 2015 — Table 12 (``state_1km``).
    ESA, Sentinel-2 MSI L1C Product Specification — QA60 bit table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise

import numpy as np
from jaxtyping import Bool, Int


def mask_from_qa_bits(
    qa: Int[np.ndarray, "*dims"],
    bits: Sequence[int],
    *,
    invert: bool = False,
) -> Bool[np.ndarray, "*dims"]:
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
    scl: Int[np.ndarray, "*dims"],
    classes: Sequence[int],
    *,
    invert: bool = False,
) -> Bool[np.ndarray, "*dims"]:
    """Decode a Sentinel-2 SCL band by class membership.

    Returns True where the SCL value equals any of the listed
    ``classes``. With ``invert=True``, returns True where the SCL value
    is *not* in the list — handy for "keep only these classes" masks.

    SCL class IDs come from `geotoolz.qa.SCL` (an IntEnum). Mixing raw
    ints and enum members is fine — they're cast to ``int`` here.

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


def mask_from_bit_field(
    qa: Int[np.ndarray, "*batch h w"],
    bits: Sequence[int],
    values: Sequence[int],
    *,
    invert: bool = False,
) -> Bool[np.ndarray, "*batch h w"]:
    """Decode a contiguous multi-bit QA field by value membership.

    Several sensor QA layers pack a small categorical field into two or
    more adjacent bits — e.g. MODIS ``state_1km`` bits ``[0, 1]`` are a
    2-bit "cloud state" field (``00`` clear, ``01`` cloudy, ``10``
    mixed, ``11`` not-set/assumed-clear). OR-ing the bits independently
    (the standard Landsat semantics) is wrong here: a "mixed" pixel has
    bit 1 set but is not cloudy.

    This helper extracts the bit field as an integer and returns True
    where its value is in ``values``.

    Args:
        qa: Integer QA array.
        bits: Bit positions, ordered LSB → MSB. The field width is
            ``len(bits)`` and the bits must be contiguous and in
            ascending order.
        values: Integer field-values to match. For MODIS ``cloud``,
            ``values=(1, 2)`` matches "cloudy" and "mixed".
        invert: When True, flip the result.

    Returns:
        Boolean array of the same shape as ``qa``.

    Raises:
        ValueError: If ``bits`` is empty, not contiguous, not
            ascending, or if ``values`` is empty.
    """
    bits_tuple = tuple(int(b) for b in bits)
    if not bits_tuple:
        raise ValueError("mask_from_bit_field: `bits` must not be empty")
    for prev, nxt in pairwise(bits_tuple):
        if nxt != prev + 1:
            raise ValueError(
                "mask_from_bit_field: `bits` must be contiguous and ascending; "
                f"got {bits_tuple}"
            )
    if any(b < 0 for b in bits_tuple):
        raise ValueError("mask_from_bit_field: bit positions must be non-negative")
    values_tuple = tuple(int(v) for v in values)
    if not values_tuple:
        raise ValueError("mask_from_bit_field: `values` must not be empty")

    qa_int = qa.astype(np.int64, copy=False)
    shift = bits_tuple[0]
    width = len(bits_tuple)
    field = (qa_int >> shift) & ((1 << width) - 1)
    out = np.isin(field, np.asarray(values_tuple))
    return ~out if invert else out


def reduce_bit_masks(
    qa: Int[np.ndarray, "*batch h w"],
    bit_groups: Mapping[str, Sequence[int]],
) -> Bool[np.ndarray, "*batch h w"]:
    """OR-reduce several named bit-groups into a single mask.

    Each entry in ``bit_groups`` is a sequence of bit positions; the
    helper calls ``mask_from_qa_bits`` per group and OR-s the results.
    Used by the sensor-preset Operators to materialise a single boolean
    mask from a registry slice.

    Args:
        qa: Integer QA array.
        bit_groups: Mapping from group name (purely for readability —
            keys are ignored at runtime) to bit positions.

    Returns:
        Boolean array of the same shape as ``qa``.

    Raises:
        ValueError: If ``bit_groups`` is empty.
    """
    if not bit_groups:
        raise ValueError("reduce_bit_masks: `bit_groups` must not be empty")
    out: np.ndarray | None = None
    for bits in bit_groups.values():
        layer = mask_from_qa_bits(qa, bits)
        out = layer if out is None else np.logical_or(out, layer)
    assert out is not None  # for type-checkers; we checked emptiness above
    return out
