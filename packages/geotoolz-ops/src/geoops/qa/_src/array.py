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
4. **`flagged_band_count`** and **`mask_from_band_count`** — EMIT's
   packed per-band mask: the number of flagged bands per pixel, and the
   dilated saturation mask above each detector column's baseline. These
   two are NumPy-only (a lookup table and a SciPy distance transform).

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
    JPL, "EMIT L2A Data Product User Guide", V1.0, JPL D-107863, 2022 —
    Table 3-3 (``band_mask``).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import einx
import numpy as np
from jaxtyping import Bool, Int, UInt8
from pipekit_array import array_namespace
from scipy import ndimage


#: Bits set in each byte value 0 to 255.
_POPCOUNT: Int[np.ndarray, " value"] = einx.sum(
    "value [bit]",
    einx.rearrange(
        "(value bit) -> value bit",
        np.unpackbits(np.arange(256, dtype=np.uint8)),
        bit=8,
    ).astype(np.int64),
)


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


def flagged_band_count(
    bandmask: UInt8[np.ndarray, "row col byte"],
) -> Int[np.ndarray, "row col"]:
    """Number of bands flagged in each pixel's packed band mask.

    nᵢⱼ = ∑_k b_ijk, the bits set across the pixel's bytes.

    EMIT's ``band_mask`` packs one bit per spectral band into 36 bytes for
    its 285 bands (EMIT L2A User Guide, Table 3-3); a set bit marks a band
    interpolated over a bad detector element or saturation. The count does
    not depend on the bit order within a byte.

    Algorithm:
        1. Look up the bits set in every byte value in a 256-entry table.
        2. Gather it for each byte and sum over the bytes.

    The table avoids ``np.unpackbits``, which would allocate eight times
    the input.

    Args:
        bandmask: Packed band mask, the bytes on the last axis.

    Returns:
        The count nᵢⱼ, ``int64``.

    Examples:
        >>> bandmask = np.array([[[0b1010_0000, 0b0000_0001]]], dtype=np.uint8)
        >>> flagged_band_count(bandmask).tolist()
        [[3]]
    """
    packed = np.asarray(bandmask)
    if packed.dtype != np.uint8:
        raise TypeError(
            f"flagged_band_count: bandmask must be uint8, got {packed.dtype}"
        )
    # bits set in each byte of each pixel
    flagged = _POPCOUNT[packed]
    # nᵢⱼ = ∑_k b_ijk
    return einx.sum("row col [byte]", flagged)


def mask_from_band_count(
    count: Int[np.ndarray, "row col"],
    *,
    valid: Bool[np.ndarray, "row col"] | None = None,
    dilation: int = 10,
) -> Bool[np.ndarray, "row col"]:
    """Pixels with more flagged bands than their column's baseline, dilated.

    satᵢⱼ = [nᵢⱼ - m_j > 0] with m_j = minᵢ nᵢⱼ over valid rows, grown by
    ``dilation`` pixels in city-block distance.

    Algorithm:
        1. Take each column's minimum count over its valid rows.
        2. Flag valid pixels above their column's minimum.
        3. Threshold the city-block distance to the flagged pixels.

    EMIT is a push-broom instrument, so column j is one cross-track detector
    element in every row. A dead or hot element flags the same bands down
    its whole column, so the column minimum is the static part and what a
    pixel has above it is transient, in practice saturation. Compute the
    mask in swath geometry, before any orthorectification: on a map grid
    the columns no longer follow the detector. ``dilation=10`` is SPIIRE's
    default, 600 m on EMIT's 60 m pixels.

    Invalid pixels never set a baseline and are never flagged, not even by
    the dilation, and a column with no valid row is all ``False``.

    Args:
        count: Flagged-band count nᵢⱼ, from :func:`flagged_band_count`.
        valid: Pixels to use; ``None`` uses all.
        dilation: Growth radius r in pixels.

    Returns:
        The mask; True means mask out.

    Raises:
        ValueError: If ``dilation`` is negative, or ``valid`` does not match
            the shape of ``count``.

    Examples:
        A column with three bands flagged in every row, and one pixel with
        two more:

        >>> count = np.full((5, 5), 3)
        >>> count[2, 2] = 5
        >>> mask = mask_from_band_count(count, dilation=1)
        >>> mask.astype(int).tolist()[1:4]
        [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]]
    """
    if dilation < 0:
        raise ValueError(f"mask_from_band_count: dilation must be >= 0, got {dilation}")
    counts = np.asarray(count)
    if valid is None:
        valid_mask = np.ones(counts.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid, dtype=bool)
        if valid_mask.shape != counts.shape:
            raise ValueError(
                "mask_from_band_count: valid has shape "
                f"{valid_mask.shape}, count has {counts.shape}"
            )
    # invalid pixels never set a baseline
    largest = np.iinfo(np.int64).max
    masked = np.where(valid_mask, counts.astype(np.int64), largest)
    # m_j = minᵢ nᵢⱼ over valid rows
    baseline = einx.min("[row] col", masked)
    # satᵢⱼ = [nᵢⱼ - m_j > 0], valid pixels only
    excess = einx.subtract("row col, col -> row col", masked, baseline)
    saturated = valid_mask & (excess > 0)
    if not bool(saturated.any()):
        return saturated
    # city-block distance to the nearest saturated pixel
    distance = ndimage.distance_transform_cdt(~saturated, metric="taxicab")
    # maskᵢⱼ = [d₁((i, j), sat) ≤ r], on valid pixels only
    return (distance <= dilation) & valid_mask


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
