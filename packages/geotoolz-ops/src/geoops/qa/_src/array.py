"""Tier-A kernels — element-wise QA / cloud-mask decoding on any backend.

The canonical home for QA-band decoding:

1. **`mask_from_qa_bits`** — single-bit-flag decoding for Landsat-style
   bitmask QA layers (``QA_PIXEL``): True where ANY listed bit is set.
2. **`mask_from_scl`** — Sentinel-2 SCL class-membership decoding.
3. **`mask_from_bit_field`** — extract a multi-bit field (e.g. MODIS
   ``state_1km`` bits ``[0, 1]`` which encode a 2-bit cloud state) and
   test membership against a set of integer field-values. Different from
   ``mask_from_qa_bits`` because that helper OR-s individual bits, which
   is wrong when the bits form a single contiguous categorical field.

Each decoder is element-wise and written against the array namespace of
its input, so the QA operators run it through pipekit-array's ``KernelOp``
on NumPy, on JAX (under ``jax.jit`` too, which needs ``jax_enable_x64``
for the ``int64`` bit arithmetic) and on the NumPy blocks of a dask cube.

Mask *application* lives in `geoops.mask` (`apply_mask` /
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

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from jaxtyping import Bool, Int
from pipekit_array import array_namespace


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
    bitmask = 0
    for b in bits:
        if b < 0:
            raise ValueError(f"bit position must be non-negative; got {b}")
        bitmask |= 1 << int(b)
    out = (_as_int64(qa) & bitmask) != 0
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

    SCL class IDs come from `geoops.qa.SCL` (an IntEnum). Mixing raw
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
    out = _isin(scl, [int(c) for c in classes])
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

    shift = bits_tuple[0]
    width = len(bits_tuple)
    field = (_as_int64(qa) >> shift) & ((1 << width) - 1)
    out = _isin(field, values_tuple)
    return ~out if invert else out


def _as_int64(qa: Int[np.ndarray, "*dims"]) -> Int[np.ndarray, "*dims"]:
    """``qa`` as ``int64`` on its own backend, for bit arithmetic.

    Example:
        >>> _as_int64(np.array([3], dtype=np.uint8)).dtype
        dtype('int64')
    """
    xp = array_namespace(qa)
    return xp.astype(qa, xp.int64)


def _isin(
    x: Int[np.ndarray, "*dims"], values: Sequence[int]
) -> Bool[np.ndarray, "*dims"]:
    """Element-wise membership in a short list of integers, on any backend.

    The Array API has no ``isin``; for the handful of QA classes or field
    values a mask tests, an OR of equalities is as fast and runs under
    ``jax.jit``.

    Example:
        >>> _isin(np.array([1, 2, 3]), [1, 3])
        array([ True, False,  True])
    """
    out = None
    for v in values:
        hit = x == int(v)
        out = hit if out is None else out | hit
    if out is None:
        raise ValueError("isin: `values` must not be empty")
    return out
