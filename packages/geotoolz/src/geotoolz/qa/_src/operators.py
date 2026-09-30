"""Tier-B Operators — sensor-specific QA decoders and mask presets.

This module is the *carrier-aware* layer on top of the QA primitives in
`geotoolz.qa._src.array` (single-bit-flag, SCL class-membership,
multi-bit-field, and bit-group-reduction decoding).

Two operator families live here:

- *Generic* mask extraction (`MaskClouds`, `MaskCloudShadow`,
  `MaskCirrus`, `MaskSnow`, `MaskWater`, `MaskNoData` — one decoder,
  several semantic names — plus `MaskInvalid`, `MaskSaturated` and the
  multi-layer `DecodeBitmask`) — pick these when you want to feed an
  explicit list of bits or class values. Pair with
  `geotoolz.mask.ApplyMask` to apply the result.
- *Sensor presets* whose defaults encode the published bit / class
  layouts (Landsat C2 ``QA_PIXEL``, Sentinel-2 ``QA60`` / ``SCL``,
  MODIS ``state_1km``). Pick these for "give me the standard cloud mask
  for sensor X" workflows.

Both layers return a boolean mask following the project convention:
**True = mask this pixel out**. Bit decoding is metadata-independent, so
every operator accepts either a ``GeoTensor`` or a plain ``np.ndarray``
and returns the same carrier kind (string ``qa_band`` selectors and the
fill-value path of `MaskInvalid` / `MaskNoData` are the
metadata-dependent exceptions — they require a GeoTensor or an explicit
``invalid_value``).

One vocabulary throughout: ``qa_band`` selects the QA band from a stack
(``None`` = the carrier itself *is* the QA band), ``bits`` lists
independent flag bits, ``values`` lists categorical class / field
values, and the sensor presets take ``targets`` — names from
`SENSOR_QA_REGISTRY` to mask out.

The band axis is ``-3`` by default. A ``(T, C, H, W)`` time stack is
decoded frame by frame into a ``(T, 1, H, W)`` mask
(``(T, n_layers, H, W)`` for `DecodeBitmask`).

References:
    USGS, "Landsat 8-9 Collection 2 Level-2 Science Product Guide",
    LSDS-1619, 2022.
    USGS, "Landsat 4-7 Collection 2 Level-2 Science Product Guide",
    LSDS-1618, 2022.
    MODIS Surface Reflectance User's Guide (MOD09 / MYD09),
    Vermote, 2015 — Table 12 (``state_1km``).
    ESA, Sentinel-2 MSI Level-1C / Level-2A product specifications.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pipekit import Operator

from geotoolz._src.bands import resolve_band, strip_band_attrs
from geotoolz._src.config import mapping_from_pairs, mapping_to_pairs
from geotoolz._src.shape import over_frames
from geotoolz._src.valid import invalid_values, valid_pixels, wrap_filled
from geotoolz._src.wrap import wrap_like
from geotoolz.qa._src.array import (
    mask_from_bit_field,
    mask_from_qa_bits,
    mask_from_scl,
)
from geotoolz.qa._src.scl import SCL_TARGETS


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


BandSelector = int | str | None


# ---------------------------------------------------------------------------
# Sensor → target registry
# ---------------------------------------------------------------------------
#
# The presets below pull bit / class definitions from this registry. Each
# inner entry is one of:
#
# - ``{"bits": (b, ...)}``      — independent single-bit flags;
#                                 OR-reduced via ``mask_from_qa_bits``.
# - ``{"field": (b0, b1, ...),  — a contiguous multi-bit field decoded by
#    "values": (v, ...)}``        ``mask_from_bit_field``.
# - ``{"values": (v, ...)}``    — categorical class IDs decoded via
#                                 ``mask_from_scl``.
#
# Bit citations:
# - Landsat 8/9 C2 ``QA_PIXEL`` — USGS LSDS-1619 Table 6-3 (bits
#   0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow,
#   6 clear, 7 water). Bits 8-15 hold the confidence sub-fields, which
#   we do not expose here.
# - Landsat 4-7 C2 ``QA_PIXEL`` — USGS LSDS-1618 Table 6-3. Bits match
#   the C2 layout above EXCEPT bit 2 ("cirrus") is unused on TM/ETM+
#   (no SWIR-cirrus channel); we omit it from the L7 preset.
# - MODIS ``state_1km`` — MOD09 user-guide Table 12. Bits [0,1] are a
#   2-bit cloud-state field (0=clear, 1=cloudy, 2=mixed, 3=not-set);
#   bit 2 is cloud shadow; bits [8,9] are a 2-bit cirrus field
#   (0=none, 1=small, 2=average, 3=high).
SENSOR_QA_REGISTRY: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {
    "s2_qa60": {
        "cloud": {"bits": (10,)},
        "cirrus": {"bits": (11,)},
    },
    # Derived from `geotoolz.qa._src.scl.SCL_TARGETS` so the registry and
    # the SCL_* convenience sets cannot disagree.
    "s2_scl": {
        name: {"values": tuple(int(c) for c in classes)}
        for name, classes in SCL_TARGETS.items()
    },
    # Landsat 8/9 Collection-2 Level-2 QA_PIXEL (LSDS-1619 Table 6-3).
    "landsat_qa_pixel": {
        "fill": {"bits": (0,)},
        "dilated_cloud": {"bits": (1,)},
        "cirrus": {"bits": (2,)},
        "cloud": {"bits": (3,)},
        "cloud_shadow": {"bits": (4,)},
        "snow": {"bits": (5,)},
        "clear": {"bits": (6,)},
        "water": {"bits": (7,)},
    },
    # Landsat 4-7 Collection-2 QA_PIXEL (LSDS-1618 Table 6-3). Same bit
    # layout EXCEPT bit 2 ("cirrus") is unused on TM/ETM+ sensors.
    "landsat_qa_pixel_l7": {
        "fill": {"bits": (0,)},
        "dilated_cloud": {"bits": (1,)},
        "cloud": {"bits": (3,)},
        "cloud_shadow": {"bits": (4,)},
        "snow": {"bits": (5,)},
        "clear": {"bits": (6,)},
        "water": {"bits": (7,)},
    },
    # MODIS state_1km (MOD09 user guide, Table 12).
    "modis_state_qa": {
        # cloud state: bits[0,1] field — 0=clear, 1=cloudy, 2=mixed,
        # 3=not-set. Default "cloud" target flags cloudy + mixed.
        "cloud": {"field": (0, 1), "values": (1, 2)},
        "cloud_shadow": {"bits": (2,)},
        # cirrus field: bits[8,9] — 0=none, 1=small, 2=average, 3=high.
        # Default "cirrus" flags small/average/high.
        "cirrus": {"field": (8, 9), "values": (1, 2, 3)},
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_int_sequence(
    values: Sequence[int] | None, name: str
) -> tuple[int, ...] | None:
    """Cast to tuple of ints; return ``None`` unchanged; reject empty."""
    if values is None:
        return None
    normalized = tuple(int(v) for v in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _select_qa(
    gt: GeoTensor | np.ndarray, qa_band: BandSelector, axis: int
) -> np.ndarray:
    """Select the QA band from a stack, or return the carrier as-is."""
    arr = np.asarray(gt)
    if qa_band is None:
        return arr
    return np.take(arr, resolve_band(gt, qa_band), axis=axis)


def _decode_bits_with_mode(
    qa: np.ndarray, bits: Sequence[int], mode: str
) -> np.ndarray:
    """Decode independent flag bits with ``any`` (OR) or ``all`` (AND) reduction.

    ``mode="any"`` is the standard `mask_from_qa_bits` semantics; we
    forward to that helper. ``mode="all"`` is the rare "every listed
    bit must be set" case (e.g. confidence-pair fields) — implemented
    inline since the cloud primitive only offers OR.
    """
    bits_tuple = tuple(int(bit) for bit in bits)
    for bit in bits_tuple:
        if bit < 0:
            raise ValueError(f"bit position must be non-negative; got {bit}")
    if mode == "any":
        return mask_from_qa_bits(qa, bits_tuple)
    if mode == "all":
        qa_int = qa.astype(np.int64, copy=False)
        bitmask = 0
        for bit in bits_tuple:
            bitmask |= 1 << bit
        return (qa_int & bitmask) == bitmask
    raise ValueError("mode must be 'any' or 'all'")


def _mask_from_definition(
    qa: np.ndarray,
    *,
    bits: Sequence[int] | None,
    values: Sequence[int] | None,
    mode: str,
    invert: bool = False,
) -> np.ndarray:
    """Build a mask from either bit positions or categorical values."""
    if bits is None and values is None:
        raise ValueError("provide either bits or values")
    if bits is not None and values is not None:
        raise ValueError("provide only one of bits or values")
    if bits is not None:
        mask = _decode_bits_with_mode(qa, bits, mode)
    else:
        # Delegate value-membership decoding to the shared cloud primitive.
        mask = mask_from_scl(qa, values or ())
    return ~mask if invert else mask


def _decode_registry_entry(
    qa: np.ndarray, entry: Mapping[str, Sequence[int]]
) -> np.ndarray:
    """Decode a single registry entry to a boolean mask.

    Dispatches on the entry shape:
    - ``{"bits": (...)}``  → `mask_from_qa_bits` (OR of single bits).
    - ``{"field": (...), "values": (...)}`` → `mask_from_bit_field`.
    - ``{"values": (...)}`` → `mask_from_scl` (categorical).
    """
    if "field" in entry:
        return mask_from_bit_field(qa, entry["field"], entry["values"])
    if "bits" in entry:
        return mask_from_qa_bits(qa, entry["bits"])
    if "values" in entry:
        return mask_from_scl(qa, entry["values"])
    raise ValueError(f"unrecognised registry entry: {dict(entry)!r}")


def _registry_bit_groups(
    sensor: str, targets: Sequence[str]
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Look up multiple targets in the registry and return their definitions."""
    if not targets:
        raise ValueError(f"{sensor} QA targets must not be empty")
    sensor_def = SENSOR_QA_REGISTRY[sensor]
    out: dict[str, dict[str, tuple[int, ...]]] = {}
    for target in targets:
        try:
            out[target] = sensor_def[target]
        except KeyError as exc:
            raise ValueError(f"unknown {sensor} QA target: {target!r}") from exc
    return out


def _decode_targets_to_mask(
    qa: np.ndarray,
    sensor: str,
    targets: Sequence[str],
) -> np.ndarray:
    """OR-reduce decoded masks for each named target into one mask."""
    entries = _registry_bit_groups(sensor, targets)
    out: np.ndarray | None = None
    for entry in entries.values():
        layer = _decode_registry_entry(qa, entry)
        out = layer if out is None else np.logical_or(out, layer)
    assert out is not None  # _registry_bit_groups rejects empty targets
    return out


# ---------------------------------------------------------------------------
# Generic decoders
# ---------------------------------------------------------------------------


class DecodeBitmask(Operator):
    """Unpack a QA bitmask into named boolean mask layers.

    A more general sibling of `MaskClouds` (``bits=`` mode): instead
    of returning a single OR-ed mask, this operator returns a *stacked*
    multi-band boolean carrier with one layer per named entry in
    ``bits``.

    Args:
        bits: Mapping from output-layer name to bit positions, or the
            equivalent ``[[name, bits], ...]`` pairs that ``get_config``
            emits.
        mode: ``"any"`` marks a pixel when any listed bit is set;
            ``"all"`` requires every listed bit to be set (rare —
            useful for confidence-pair sub-fields).
        qa_band: Optional integer or named band selector. When omitted,
            the input carrier itself is treated as the QA band.
        axis: Position of the band axis when ``qa_band`` selects from a
            stack.

    Returns:
        GeoTensor | numpy.ndarray: A multi-band boolean carrier with
            shape ``(n_layers, height, width)``, matching the input
            carrier kind. For GeoTensor input the output ``band_names``
            attr is set from the keys of ``bits``; plain ``np.ndarray``
            input returns a plain stacked boolean array (no metadata to
            carry the names).

    Examples:
        >>> from geotoolz.qa import DecodeBitmask
        >>> # Landsat-8 QA_PIXEL — one band per flag.
        >>> op = DecodeBitmask(
        ...     bits={"cloud": [3], "cirrus": [2], "shadow": [4]},
        ...     qa_band="QA_PIXEL",
        ... )
        >>> layers = op(landsat_stack)  # (3, H, W) bool GeoTensor
    """

    def __init__(
        self,
        *,
        bits: Mapping[str, Sequence[int]] | Sequence[Sequence[Any]],
        mode: str = "any",
        qa_band: BandSelector = None,
        axis: int = -3,
    ) -> None:
        bits_map = mapping_from_pairs(bits)
        if not bits_map:
            raise ValueError("DecodeBitmask: `bits` must not be empty")
        if mode not in {"any", "all"}:
            raise ValueError("mode must be 'any' or 'all'")
        self.bits = {
            name: tuple(int(bit) for bit in bit_list)
            for name, bit_list in bits_map.items()
        }
        for name, bit_list in self.bits.items():
            if not bit_list:
                raise ValueError(f"bits[{name!r}] must not be empty")
        self.mode = mode
        self.qa_band = qa_band
        self.axis = axis

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        qa = _select_qa(gt, self.qa_band, self.axis)
        names = list(self.bits)
        layers = [
            _decode_bits_with_mode(qa, self.bits[name], self.mode) for name in names
        ]
        mask = np.stack(layers, axis=0)
        if not hasattr(gt, "transform"):
            return mask
        return wrap_like(
            gt,
            mask,
            fill_value_default=False,
            attrs=strip_band_attrs(gt.attrs),
            band_names=names,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "bits": mapping_to_pairs(self.bits),
            "mode": self.mode,
            "qa_band": self.qa_band,
            "axis": self.axis,
        }


def _invalid_mask(
    gt: GeoTensor | np.ndarray,
    invalid_value: float | None,
    axis: int,
    name: str,
) -> np.ndarray:
    """Per-pixel invalid mask shared by `MaskInvalid` and `MaskNoData`.

    An element is invalid when it equals the fill (a ``NaN`` fill matches
    ``NaN``) or is non-finite (see :mod:`geotoolz._src.valid`); a pixel is
    invalid when any band is.
    """
    fill = (
        invalid_value
        if invalid_value is not None
        else getattr(gt, "fill_value_default", None)
    )
    if fill is None:
        raise ValueError(
            f"{name}: no invalid_value provided and the carrier has no "
            "fill_value_default. Pass `invalid_value=...` explicitly."
        )
    invalid = invalid_values(gt, fill_value=fill)
    return invalid if invalid.ndim <= 2 else np.any(invalid, axis=axis)


class MaskInvalid(Operator):
    """Mark invalid (nodata) pixels: True = invalid, drop it.

    The package-polarity counterpart of georeader's
    ``GeoTensor.invalidmask()`` reduced over bands: a pixel is invalid
    when *any* band equals the fill value or is non-finite. Unlike
    georeader's comparison, a ``NaN`` fill (``fill_value_default=nan``)
    matches ``NaN`` pixels (see :mod:`geotoolz._src.valid`).

    Uses the carrier's ``fill_value_default`` by default; pass
    ``invalid_value`` explicitly to mark a different sentinel (required
    for plain-ndarray carriers, which have no fill value).

    Args:
        invalid_value: Sentinel value treated as "invalid". ``None``
            uses the carrier's ``fill_value_default``.
        axis: Position of the band axis (used for the per-pixel
            ANY-band reduction). Default ``-3``.

    Examples:
        >>> from geotoolz.mask import ApplyMask
        >>> from geotoolz.qa import MaskInvalid
        >>> mask = MaskInvalid()(geotensor)  # True = invalid
        >>> clean = ApplyMask(mask=MaskInvalid(), fill_value=np.nan)(geotensor)
    """

    def __init__(
        self, *, invalid_value: float | int | None = None, axis: int = -3
    ) -> None:
        self.invalid_value = invalid_value
        self.axis = axis

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        mask = _invalid_mask(gt, self.invalid_value, self.axis, "MaskInvalid")
        return wrap_like(gt, mask, fill_value_default=False)


class _QAMask(Operator):
    """The one generic QA decoder behind the ``Mask*`` shortcuts.

    Subclasses differ only by semantic name (``MaskClouds``,
    ``MaskCirrus``, ...) — the runtime behaviour is identical and
    delegates to the `geotoolz.qa` primitives. Accepts a ``GeoTensor``
    or a plain ``np.ndarray`` and returns the same carrier kind
    (string ``qa_band`` selectors require a GeoTensor with band-name
    attrs).

    Args:
        qa_band: Integer or named band selector for the QA band. ``None``
            (default) treats the carrier itself as the QA band.
        bits: Independent flag bits; a pixel is marked when any
            (``mode="any"``) or every (``mode="all"``) bit is set.
        values: Categorical values (e.g. SCL classes) to mark. Pass
            exactly one of ``bits`` / ``values``.
        mode: ``"any"`` (default) or ``"all"``; only used with ``bits``.
        axis: Position of the band axis. Default ``-3``.
        invert: Mark pixels that do *not* match instead — a keep-list
            (e.g. ``values=[SCL.VEGETATION], invert=True`` drops every
            other class).

    Returns:
        GeoTensor | numpy.ndarray: Boolean mask, True = drop.
    """

    _target_name: ClassVar[str] = "qa"

    def __init__(
        self,
        *,
        qa_band: BandSelector = None,
        bits: Sequence[int] | None = None,
        values: Sequence[int] | None = None,
        mode: str = "any",
        axis: int = -3,
        invert: bool = False,
    ) -> None:
        self.qa_band = qa_band
        self.bits = _normalize_int_sequence(bits, "bits")
        self.values = _normalize_int_sequence(values, "values")
        if mode not in {"any", "all"}:
            raise ValueError("mode must be 'any' or 'all'")
        self.mode = mode
        self.axis = axis
        self.invert = invert

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        qa = _select_qa(gt, self.qa_band, self.axis)
        mask = _mask_from_definition(
            qa,
            bits=self.bits,
            values=self.values,
            mode=self.mode,
            invert=self.invert,
        )
        return wrap_like(gt, mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "qa_band": self.qa_band,
            "bits": None if self.bits is None else list(self.bits),
            "values": None if self.values is None else list(self.values),
            "mode": self.mode,
            "axis": self.axis,
            "invert": self.invert,
        }


class MaskClouds(_QAMask):
    """Return True where a QA band marks cloud-contaminated pixels.

    Examples:
        >>> from geotoolz.qa import MaskClouds
        >>> # Sentinel-2 QA60: bit 10 is "opaque clouds".
        >>> mask = MaskClouds(qa_band="QA60", bits=[10])(s2_geotensor)
        >>> # Sentinel-2 SCL: the cloud-like classes.
        >>> from geotoolz.qa import SCL_CLOUDS
        >>> mask = MaskClouds(qa_band="SCL", values=sorted(SCL_CLOUDS))(s2_l2a)
    """


class MaskCloudShadow(_QAMask):
    """Return True where a QA band marks cloud-shadow pixels.

    Examples:
        >>> from geotoolz.qa import MaskCloudShadow
        >>> # Landsat-8 QA_PIXEL: bit 4 is "cloud shadow".
        >>> mask = MaskCloudShadow(qa_band="QA_PIXEL", bits=[4])(landsat_stack)
    """


class MaskCirrus(_QAMask):
    """Return True where a QA band marks cirrus pixels.

    Examples:
        >>> from geotoolz.qa import MaskCirrus
        >>> # Sentinel-2 QA60: bit 11 is "cirrus".
        >>> mask = MaskCirrus(qa_band="QA60", bits=[11])(s2_geotensor)
    """


class MaskSnow(_QAMask):
    """Return True where a QA band marks snow or ice pixels.

    Examples:
        >>> from geotoolz.qa import MaskSnow
        >>> # Landsat-8 QA_PIXEL: bit 5 is "snow / ice".
        >>> mask = MaskSnow(qa_band="QA_PIXEL", bits=[5])(landsat_stack)
    """


class MaskWater(_QAMask):
    """Return True where a QA band marks water pixels.

    Examples:
        >>> from geotoolz.qa import MaskWater
        >>> # Landsat-8 QA_PIXEL: bit 7 is "water".
        >>> mask = MaskWater(qa_band="QA_PIXEL", bits=[7])(landsat_stack)
    """


class MaskNoData(_QAMask):
    """Return True where pixels are no-data by QA value or carrier fill value.

    Two operating modes:

    1. **QA-driven**: pass ``bits`` / ``values`` (plus ``qa_band`` when
       the QA band is part of a stack) to decode no-data from a
       dedicated QA band, exactly like the other `_QAMask` shortcuts.
    2. **Fill-driven** (default): without ``bits`` / ``values`` it is
       `MaskInvalid` — pixels invalid in *any* band (equal to the
       carrier's ``fill_value_default``, a ``NaN`` fill matching
       ``NaN``, or non-finite) are marked (see
       :mod:`geotoolz._src.valid`).

    The QA-driven mode accepts a ``GeoTensor`` or a plain ``np.ndarray``
    and returns the same carrier kind; the fill-driven mode reads
    carrier metadata, so it requires a GeoTensor with a
    ``fill_value_default``.

    Args:
        qa_band: Optional QA band selector (QA-driven mode only).
        bits: Bit positions that mark no-data.
        values: Categorical values that mark no-data (e.g. SCL=0).
        mode: ``"any"`` (default) or ``"all"``; only used with ``bits``.
        axis: Position of the band axis. Default ``-3``.
        invert: Flip the QA-driven mask (see `MaskClouds`).

    Returns:
        GeoTensor: Boolean no-data mask.

    Examples:
        >>> from geotoolz.qa import MaskNoData
        >>> # Sentinel-2 SCL: class 0 is NO_DATA.
        >>> nodata = MaskNoData(qa_band="SCL", values=[0])(s2_l2a)
        >>> # Fill-value fallback when no QA band is available.
        >>> nodata = MaskNoData()(carrier_with_fill_value)
    """

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        if self.bits is not None or self.values is not None:
            return super()._apply(gt)
        if self.qa_band is not None:
            raise ValueError(
                "MaskNoData: qa_band needs bits or values to decode; drop "
                "qa_band to fall back to the carrier fill value."
            )
        mask = _invalid_mask(gt, None, self.axis, "MaskNoData")
        return wrap_like(gt, mask, fill_value_default=False)


class MaskSaturated(Operator):
    """Return True where pixels are at or above a saturation value.

    The one saturation mask in the package. Nodata pixels (fill value or
    non-finite, see :mod:`geotoolz._src.valid`) are never reported as
    saturated — so a ``uint16`` fill at the dtype maximum is not flagged
    — and the output declares ``fill_value_default=False``.

    Args:
        qa_band: Optional band selector. When omitted, every band is
            tested.
        saturation_value: Saturation threshold; pixels ``>=`` it are
            flagged. If omitted for integer arrays, the dtype maximum is
            used; float arrays require an explicit value.
        reduce_bands: When True (default) and ``qa_band`` is None,
            OR-reduce the per-band flags over ``axis`` into one
            ``(H, W)`` mask; when False, return one flag per band
            (same shape as the input).
        axis: Position of the band axis (used for the cross-band
            reduction).

    Returns:
        GeoTensor | numpy.ndarray: Boolean mask with saturated pixels
            marked True, matching the input carrier kind (``GeoTensor``
            in → ``GeoTensor`` out, plain ``np.ndarray`` in → plain
            array out).

    Examples:
        >>> from geotoolz.qa import MaskSaturated
        >>> # uint16 Sentinel-2 — saturation_value defaults to 65535.
        >>> sat = MaskSaturated()(s2_uint16_stack)
        >>> # Per-band flags at a reflectance ceiling.
        >>> sat = MaskSaturated(saturation_value=1.0, reduce_bands=False)(refl)
    """

    def __init__(
        self,
        *,
        qa_band: BandSelector = None,
        saturation_value: float | int | None = None,
        reduce_bands: bool = True,
        axis: int = -3,
    ) -> None:
        self.qa_band = qa_band
        self.saturation_value = saturation_value
        self.reduce_bands = reduce_bands
        self.axis = axis

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = _select_qa(gt, self.qa_band, self.axis)
        saturation_value = self.saturation_value
        if saturation_value is None:
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(
                    "MaskSaturated: pass saturation_value for non-integer inputs."
                )
            saturation_value = np.iinfo(arr.dtype).max
        mask = arr >= saturation_value
        if self.reduce_bands and self.qa_band is None and mask.ndim > 2:
            mask = np.any(mask, axis=self.axis)
        valid = valid_pixels(gt) if np.ndim(gt) >= 2 else ~invalid_values(gt)
        return wrap_filled(gt, mask, fill_value_default=False, valid=valid)


# ---------------------------------------------------------------------------
# Sensor presets
# ---------------------------------------------------------------------------


class S2QA60(Operator):
    """Sentinel-2 L1C QA60 cloud + cirrus mask preset.

    QA60 (per ESA's S2 L1C product specification) encodes opaque clouds
    in bit 10 and cirrus in bit 11 (``SENSOR_QA_REGISTRY["s2_qa60"]``).
    Returns True where any requested ``targets`` bit is set.

    Note: QA60 is unreliable / zeroed-out on newer processing baselines
    (≥ 04.00). Prefer the L2A SCL band (`S2SCL`) or an ML-based
    detector when available.

    Accepts a ``GeoTensor`` or a plain ``np.ndarray`` and returns the
    same carrier kind; plain arrays need an integer ``qa_band`` (the
    default is a band *name*, which requires GeoTensor attrs).

    Args:
        qa_band: Band selector for QA60 within the input stack.
        targets: Target names to OR together (``"cloud"``,
            ``"cirrus"``). Default both.
        axis: Band axis position. Default ``-3``.

    Examples:
        >>> from geotoolz.qa import S2QA60
        >>> mask = S2QA60()(s2_l1c_stack_with_qa60_appended)
    """

    def __init__(
        self,
        *,
        qa_band: int | str = "QA60",
        targets: Sequence[str] = ("cloud", "cirrus"),
        axis: int = -3,
    ) -> None:
        self.qa_band = qa_band
        self.targets = tuple(str(target) for target in targets)
        self.axis = axis

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        qa = _select_qa(gt, self.qa_band, self.axis)
        mask = _decode_targets_to_mask(qa, "s2_qa60", self.targets)
        return wrap_like(gt, mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "qa_band": self.qa_band,
            "targets": list(self.targets),
            "axis": self.axis,
        }


class S2SCL(Operator):
    """Sentinel-2 L2A SCL preset that masks the requested ``targets`` classes.

    Returns True for pixels whose SCL class belongs to any of the named
    ``targets`` (see ``SENSOR_QA_REGISTRY["s2_scl"]``, derived from
    `geotoolz.qa.SCL`). The default targets are every class except
    vegetation, soil and water — i.e. vegetation, soil, and water are
    kept.

    Accepts a ``GeoTensor`` or a plain ``np.ndarray`` and returns the
    same carrier kind; plain arrays need an integer ``qa_band``.

    Args:
        qa_band: Band selector for the SCL band.
        targets: Class names to mask out. See
            ``SENSOR_QA_REGISTRY["s2_scl"]`` for the full vocabulary.
        axis: Band axis position. Default ``-3``.

    Examples:
        >>> from geotoolz.qa import S2SCL
        >>> # Default: mask everything that isn't vegetation/soil/water.
        >>> mask = S2SCL()(s2_l2a_with_scl)
        >>> # Custom: mask clouds, cirrus and cloud shadow only.
        >>> mask = S2SCL(targets=["cloud", "cirrus", "cloud_shadow"])(s2_l2a)
    """

    _DEFAULT_TARGETS: ClassVar[tuple[str, ...]] = (
        "no_data",
        "saturated",
        "dark",
        "cloud_shadow",
        "unclassified",
        "cloud",
        "cirrus",
        "snow",
    )

    def __init__(
        self,
        *,
        qa_band: int | str = "SCL",
        targets: Sequence[str] = _DEFAULT_TARGETS,
        axis: int = -3,
    ) -> None:
        self.qa_band = qa_band
        self.targets = tuple(str(name) for name in targets)
        self.axis = axis

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        qa = _select_qa(gt, self.qa_band, self.axis)
        mask = _decode_targets_to_mask(qa, "s2_scl", self.targets)
        return wrap_like(gt, mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "qa_band": self.qa_band,
            "targets": list(self.targets),
            "axis": self.axis,
        }


class LandsatQA_PIXEL(Operator):
    """Landsat Collection-2 QA_PIXEL mask preset.

    Returns True where any requested ``targets`` flag is set in
    ``QA_PIXEL``. Defaults to cloud, cloud shadow, and cirrus targets
    (the standard "drop cloudy" mask for L8/L9).

    Sensor selection: pass ``sensor="l7"`` for Landsat 4-7
    Collection-2 — same bit layout as L8/L9 except bit 2 ("cirrus") is
    unused on TM/ETM+. For ``sensor="l89"`` (default) the full L8/L9
    layout is used.

    Accepts a ``GeoTensor`` or a plain ``np.ndarray`` and returns the
    same carrier kind; plain arrays need an integer ``qa_band``.

    Args:
        qa_band: Band selector for QA_PIXEL.
        targets: Target flag names to OR together. See
            ``SENSOR_QA_REGISTRY["landsat_qa_pixel"]`` for available
            target names.
        sensor: ``"l89"`` (Landsat 8/9, default) or ``"l7"`` (Landsat
            4-7).
        axis: Band axis position. Default ``-3``.

    Examples:
        >>> from geotoolz.qa import LandsatQA_PIXEL
        >>> # L8/L9 default "drop everything not clear".
        >>> mask = LandsatQA_PIXEL()(landsat8_stack)
        >>> # L7 — same but no cirrus bit.
        >>> mask = LandsatQA_PIXEL(
        ...     sensor="l7", targets=["cloud", "cloud_shadow"]
        ... )(landsat7_stack)

    References:
        USGS, "Landsat 8-9 Collection 2 Level-2 Science Product Guide",
        LSDS-1619, 2022.
        USGS, "Landsat 4-7 Collection 2 Level-2 Science Product Guide",
        LSDS-1618, 2022.
    """

    _SENSOR_KEYS: ClassVar[dict[str, str]] = {
        "l89": "landsat_qa_pixel",
        "l7": "landsat_qa_pixel_l7",
    }
    _DEFAULT_TARGETS_L89: ClassVar[tuple[str, ...]] = (
        "cloud",
        "cloud_shadow",
        "cirrus",
    )
    _DEFAULT_TARGETS_L7: ClassVar[tuple[str, ...]] = ("cloud", "cloud_shadow")

    def __init__(
        self,
        *,
        qa_band: int | str = "QA_PIXEL",
        targets: Sequence[str] | None = None,
        sensor: str = "l89",
        axis: int = -3,
    ) -> None:
        if sensor not in self._SENSOR_KEYS:
            raise ValueError(
                f"sensor must be one of {sorted(self._SENSOR_KEYS)}; got {sensor!r}"
            )
        if targets is None:
            targets = (
                self._DEFAULT_TARGETS_L7
                if sensor == "l7"
                else self._DEFAULT_TARGETS_L89
            )
        self.qa_band = qa_band
        self.targets = tuple(str(target) for target in targets)
        self.sensor = sensor
        self.axis = axis

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        registry_key = self._SENSOR_KEYS[self.sensor]
        qa = _select_qa(gt, self.qa_band, self.axis)
        mask = _decode_targets_to_mask(qa, registry_key, self.targets)
        return wrap_like(gt, mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "qa_band": self.qa_band,
            "targets": list(self.targets),
            "sensor": self.sensor,
            "axis": self.axis,
        }


class MODISStateQA(Operator):
    """MODIS ``state_1km`` (or ``state_500m``) QA mask preset.

    Returns True where any requested ``targets`` is set in the MODIS
    State QA band. Defaults to cloud and cloud-shadow targets.

    The cloud and cirrus targets are decoded as *2-bit fields*, not
    independent bit flags: bits ``[0, 1]`` are the cloud state
    (0=clear, 1=cloudy, 2=mixed, 3=not-set) and bits ``[8, 9]`` are
    cirrus level (0=none, 1=small, 2=average, 3=high). The default
    "cloud" target matches cloudy + mixed; "cirrus" matches
    small/average/high.

    Accepts a ``GeoTensor`` or a plain ``np.ndarray`` and returns the
    same carrier kind; plain arrays need an integer ``qa_band``.

    Args:
        qa_band: Band selector for the state QA band.
        targets: Target flag names to OR together. See
            ``SENSOR_QA_REGISTRY["modis_state_qa"]``.
        axis: Band axis position. Default ``-3``.

    Examples:
        >>> from geotoolz.qa import MODISStateQA
        >>> # Default cloud + cloud-shadow mask.
        >>> mask = MODISStateQA()(modis_state_band)
        >>> # Include cirrus too.
        >>> mask = MODISStateQA(targets=["cloud", "cloud_shadow", "cirrus"])(state)

    References:
        Vermote, E. F., "MODIS Surface Reflectance User's Guide",
        2015, Table 12.
    """

    def __init__(
        self,
        *,
        qa_band: int | str = "state_1km",
        targets: Sequence[str] = ("cloud", "cloud_shadow"),
        axis: int = -3,
    ) -> None:
        self.qa_band = qa_band
        self.targets = tuple(str(target) for target in targets)
        self.axis = axis

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        qa = _select_qa(gt, self.qa_band, self.axis)
        mask = _decode_targets_to_mask(qa, "modis_state_qa", self.targets)
        return wrap_like(gt, mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "qa_band": self.qa_band,
            "targets": list(self.targets),
            "axis": self.axis,
        }


# ---------------------------------------------------------------------------
# ML-based mask placeholders (require optional extra)
# ---------------------------------------------------------------------------


class S2Cloudless(Operator):
    """Placeholder for the optional ML-based s2cloudless mask.

    Raises ``ImportError`` on call. Reserved for the future
    ``[cloud-ml]`` extra so that pipelines can be configured today and
    light up once the dependency is installed.

    Examples:
        >>> from geotoolz.qa import S2Cloudless
        >>> S2Cloudless(threshold=0.4).get_config()
        {'threshold': 0.4}
    """

    # Calling this Operator raises ImportError — its `get_config` is
    # still serialisable, but actually instantiating the underlying
    # model is closure-like. Mark non-YAML so callers don't accidentally
    # depend on it round-tripping through configs in production.
    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, threshold: float = 0.4) -> None:
        self.threshold = threshold

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        raise ImportError(
            "S2Cloudless requires the optional ML mask extra, which is not "
            "packaged in this release."
        )


class OmniCloudMask(Operator):
    """Placeholder for the optional ML-based OmniCloudMask detector.

    Raises ``ImportError`` on call.

    Examples:
        >>> from geotoolz.qa import OmniCloudMask
        >>> OmniCloudMask(checkpoint="default").get_config()
        {'checkpoint': 'default'}
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, checkpoint: str = "default") -> None:
        self.checkpoint = checkpoint

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        raise ImportError(
            "OmniCloudMask requires the optional ML mask extra, which is not "
            "packaged in this release."
        )


class CloudSEN12(Operator):
    """Placeholder for the optional ML-based CloudSEN12 detector.

    Raises ``ImportError`` on call.

    Examples:
        >>> from geotoolz.qa import CloudSEN12
        >>> CloudSEN12(checkpoint="default").get_config()
        {'checkpoint': 'default'}
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, checkpoint: str = "default") -> None:
        self.checkpoint = checkpoint

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        raise ImportError(
            "CloudSEN12 requires the optional ML mask extra, which is not "
            "packaged in this release."
        )
