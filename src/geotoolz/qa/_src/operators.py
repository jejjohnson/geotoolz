"""QA-band decoding operators and sensor presets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from geotoolz.cloud import SCL
from geotoolz.core import Operator


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


BandSelector = int | str | None


SENSOR_QA_REGISTRY: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {
    "s2_qa60": {
        "cloud": {"bits": (10,)},
        "cirrus": {"bits": (11,)},
    },
    "s2_scl": {
        "no_data": {"values": (int(SCL.NO_DATA),)},
        "saturated": {"values": (int(SCL.SATURATED_OR_DEFECTIVE),)},
        "dark": {"values": (int(SCL.DARK_AREA_PIXELS),)},
        "cloud_shadow": {"values": (int(SCL.CLOUD_SHADOWS),)},
        "vegetation": {"values": (int(SCL.VEGETATION),)},
        "soil": {"values": (int(SCL.NOT_VEGETATED),)},
        "water": {"values": (int(SCL.WATER),)},
        "unclassified": {"values": (int(SCL.UNCLASSIFIED),)},
        "cloud": {
            "values": (
                int(SCL.CLOUD_MEDIUM_PROBABILITY),
                int(SCL.CLOUD_HIGH_PROBABILITY),
            )
        },
        "cirrus": {"values": (int(SCL.THIN_CIRRUS),)},
        "snow": {"values": (int(SCL.SNOW),)},
    },
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
    "modis_state_qa": {
        "cloud": {"bits": (0, 1)},
        "cloud_shadow": {"bits": (2,)},
        "cirrus": {"bits": (8, 9)},
    },
}


def _as_tuple(values: Sequence[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(int(v) for v in values)


def _require_non_empty(values: tuple[int, ...] | None, name: str) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _band_names(attrs: Mapping[str, Any]) -> Sequence[str] | Mapping[str, int] | None:
    for key in ("band_names", "bands", "band_descriptions", "descriptions"):
        names = attrs.get(key)
        if names is not None:
            return names
    return None


def _resolve_band_index(gt: GeoTensor, qa_band: BandSelector, axis: int) -> int | None:
    if qa_band is None:
        return None
    if isinstance(qa_band, int):
        return qa_band

    names = _band_names(gt.attrs)
    if isinstance(names, Mapping):
        try:
            return int(names[qa_band])
        except KeyError as exc:
            raise ValueError(
                f"qa_band {qa_band!r} is not present in GeoTensor attrs"
            ) from exc

    if names is not None:
        names_list = [str(name) for name in names]
        try:
            return names_list.index(qa_band)
        except ValueError as exc:
            raise ValueError(
                f"qa_band {qa_band!r} is not present in GeoTensor attrs"
            ) from exc

    raise ValueError(
        "String qa_band selectors require GeoTensor attrs with `band_names` "
        "or a similar band-name sequence."
    )


def _select_qa(gt: GeoTensor, qa_band: BandSelector, axis: int) -> np.ndarray:
    arr = np.asarray(gt)
    band_idx = _resolve_band_index(gt, qa_band, axis)
    if band_idx is None:
        return arr
    return np.take(arr, band_idx, axis=axis)


def _decode_bits(qa: np.ndarray, bits: Sequence[int], mode: str) -> np.ndarray:
    bits_tuple = _require_non_empty(tuple(int(bit) for bit in bits), "bits")
    if mode not in {"any", "all"}:
        raise ValueError("mode must be 'any' or 'all'")
    qa_int = qa.astype(np.int64, copy=False)
    masks = []
    for bit in bits_tuple:
        if bit < 0:
            raise ValueError(f"bit position must be non-negative; got {bit}")
        masks.append((qa_int & (1 << bit)) != 0)
    reducer = np.logical_or if mode == "any" else np.logical_and
    return reducer.reduce(masks)


def _decode_values(qa: np.ndarray, values: Sequence[int]) -> np.ndarray:
    values_tuple = _require_non_empty(tuple(int(value) for value in values), "values")
    return np.isin(qa, np.asarray(values_tuple))


def _mask_from_definition(
    qa: np.ndarray,
    *,
    bits: Sequence[int] | None,
    values: Sequence[int] | None,
    mode: str,
    invert: bool = False,
) -> np.ndarray:
    if bits is None and values is None:
        raise ValueError("provide either bits or values")
    if bits is not None and values is not None:
        raise ValueError("provide only one of bits or values")
    mask = (
        _decode_bits(qa, bits, mode)
        if bits is not None
        else _decode_values(qa, values or ())
    )
    return ~mask if invert else mask


def _attrs_with_band_names(gt: GeoTensor, names: Sequence[str]) -> dict[str, Any]:
    attrs = dict(cast(Mapping[str, Any], gt.attrs))
    attrs["band_names"] = list(names)
    return attrs


class DecodeBitmask(Operator):
    """Unpack a QA bitmask into named boolean mask layers.

    Args:
        bits: Mapping from output layer name to bit positions.
        mode: ``"any"`` marks a pixel when any listed bit is set; ``"all"``
            requires every listed bit to be set.
        qa_band: Optional integer or named band selector. When omitted, the
            input carrier itself is treated as the QA band.
        axis: Position of the band axis when ``qa_band`` selects from a stack.
    """

    def __init__(
        self,
        *,
        bits: Mapping[str, Sequence[int]],
        mode: str = "any",
        qa_band: BandSelector = None,
        axis: int = 0,
    ) -> None:
        if len(bits) == 0:
            raise ValueError("DecodeBitmask: `bits` must not be empty")
        if mode not in {"any", "all"}:
            raise ValueError("mode must be 'any' or 'all'")
        self.bits = {
            name: tuple(int(bit) for bit in bit_list) for name, bit_list in bits.items()
        }
        for name, bit_list in self.bits.items():
            _require_non_empty(bit_list, f"bits[{name!r}]")
        self.mode = mode
        self.qa_band = qa_band
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        qa = _select_qa(gt, self.qa_band, self.axis)
        names = list(self.bits)
        layers = [_decode_bits(qa, self.bits[name], self.mode) for name in names]
        mask = np.stack(layers, axis=0)
        return type(gt)(
            mask,
            transform=gt.transform,
            crs=gt.crs,
            fill_value_default=False,
            attrs=_attrs_with_band_names(gt, names),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "bits": {name: list(bits) for name, bits in self.bits.items()},
            "mode": self.mode,
            "qa_band": self.qa_band,
            "axis": self.axis,
        }


class _QAMask(Operator):
    """Base class for single QA mask shortcuts."""

    _target_name: ClassVar[str] = "qa"

    def __init__(
        self,
        *,
        qa_band: BandSelector,
        bits: Sequence[int] | None = None,
        values: Sequence[int] | None = None,
        mode: str = "any",
        axis: int = 0,
        invert: bool = False,
    ) -> None:
        self.qa_band = qa_band
        self.bits = _as_tuple(bits)
        self.values = _as_tuple(values)
        self.mode = mode
        self.axis = axis
        self.invert = invert

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        qa = _select_qa(gt, self.qa_band, self.axis)
        mask = _mask_from_definition(
            qa,
            bits=self.bits,
            values=self.values,
            mode=self.mode,
            invert=self.invert,
        )
        return gt.array_as_geotensor(mask, fill_value_default=False)

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
    """Return True where a QA band marks cloud-contaminated pixels."""


class MaskCloudShadow(_QAMask):
    """Return True where a QA band marks cloud-shadow pixels."""


class MaskCirrus(_QAMask):
    """Return True where a QA band marks cirrus pixels."""


class MaskSnow(_QAMask):
    """Return True where a QA band marks snow or ice pixels."""


class MaskWater(_QAMask):
    """Return True where a QA band marks water pixels."""


class MaskNoData(Operator):
    """Return True where pixels are no-data by QA value or carrier fill value."""

    def __init__(
        self,
        *,
        qa_band: BandSelector = None,
        bits: Sequence[int] | None = None,
        values: Sequence[int] | None = None,
        axis: int = 0,
    ) -> None:
        self.qa_band = qa_band
        self.bits = _as_tuple(bits)
        self.values = _as_tuple(values)
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        if self.bits is not None or self.values is not None or self.qa_band is not None:
            qa = _select_qa(gt, self.qa_band, self.axis)
            mask = _mask_from_definition(
                qa, bits=self.bits, values=self.values, mode="any"
            )
        else:
            if gt.fill_value_default is None:
                raise ValueError(
                    "MaskNoData: no qa_band/bits/values provided and the carrier "
                    "has no fill_value_default."
                )
            arr = np.asarray(gt)
            if arr.ndim <= 2:
                mask = arr == gt.fill_value_default
            else:
                mask = np.any(arr == gt.fill_value_default, axis=self.axis)
        return gt.array_as_geotensor(mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "qa_band": self.qa_band,
            "bits": None if self.bits is None else list(self.bits),
            "values": None if self.values is None else list(self.values),
            "axis": self.axis,
        }


class MaskSaturated(Operator):
    """Return True where pixels equal a saturation value."""

    def __init__(
        self,
        *,
        qa_band: BandSelector = None,
        saturation_value: float | int | None = None,
        axis: int = 0,
    ) -> None:
        self.qa_band = qa_band
        self.saturation_value = saturation_value
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = _select_qa(gt, self.qa_band, self.axis)
        saturation_value = self.saturation_value
        if saturation_value is None:
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(
                    "MaskSaturated: pass saturation_value for non-integer inputs."
                )
            saturation_value = np.iinfo(arr.dtype).max
        mask = arr == saturation_value
        if self.qa_band is None and np.asarray(gt).ndim > 2:
            mask = np.any(mask, axis=self.axis)
        return gt.array_as_geotensor(mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "qa_band": self.qa_band,
            "saturation_value": self.saturation_value,
            "axis": self.axis,
        }


def _registry_bits(sensor: str, targets: Sequence[str]) -> dict[str, tuple[int, ...]]:
    sensor_def = SENSOR_QA_REGISTRY[sensor]
    out = {}
    for target in targets:
        try:
            out[target] = sensor_def[target]["bits"]
        except KeyError as exc:
            raise ValueError(f"unknown {sensor} QA target: {target!r}") from exc
    return out


def _registry_values(sensor: str, targets: Sequence[str]) -> tuple[int, ...]:
    sensor_def = SENSOR_QA_REGISTRY[sensor]
    values: list[int] = []
    for target in targets:
        try:
            values.extend(sensor_def[target]["values"])
        except KeyError as exc:
            raise ValueError(f"unknown {sensor} QA target: {target!r}") from exc
    return tuple(values)


class S2QA60(Operator):
    """Sentinel-2 L1C QA60 cloud and cirrus mask preset."""

    def __init__(self, *, qa_band: int | str = "QA60", axis: int = 0) -> None:
        self.qa_band = qa_band
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return MaskClouds(
            qa_band=self.qa_band,
            bits=(10, 11),
            axis=self.axis,
        )(gt)

    def get_config(self) -> dict[str, Any]:
        return {"qa_band": self.qa_band, "axis": self.axis}


class S2SCL(Operator):
    """Sentinel-2 L2A SCL preset that masks pixels outside kept classes."""

    def __init__(
        self,
        *,
        qa_band: int | str = "SCL",
        keep: Sequence[str] = ("vegetation", "soil", "water"),
        axis: int = 0,
    ) -> None:
        self.qa_band = qa_band
        self.keep = tuple(str(name) for name in keep)
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        keep_values = _registry_values("s2_scl", self.keep)
        return MaskClouds(
            qa_band=self.qa_band,
            values=keep_values,
            axis=self.axis,
            invert=True,
        )(gt)

    def get_config(self) -> dict[str, Any]:
        return {"qa_band": self.qa_band, "keep": list(self.keep), "axis": self.axis}


class LandsatQA_PIXEL(Operator):
    """Landsat Collection-2 QA_PIXEL mask preset."""

    def __init__(
        self,
        *,
        qa_band: int | str = "QA_PIXEL",
        targets: Sequence[str] = ("cloud", "cloud_shadow", "cirrus"),
        axis: int = 0,
    ) -> None:
        self.qa_band = qa_band
        self.targets = tuple(str(target) for target in targets)
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        bits = _registry_bits("landsat_qa_pixel", self.targets)
        qa = _select_qa(gt, self.qa_band, self.axis)
        masks = [_decode_bits(qa, bit_list, "any") for bit_list in bits.values()]
        mask = np.logical_or.reduce(masks)
        return gt.array_as_geotensor(mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "qa_band": self.qa_band,
            "targets": list(self.targets),
            "axis": self.axis,
        }


class MODISStateQA(Operator):
    """MODIS State QA mask preset."""

    def __init__(
        self,
        *,
        qa_band: int | str = "state_1km",
        targets: Sequence[str] = ("cloud", "cloud_shadow"),
        axis: int = 0,
    ) -> None:
        self.qa_band = qa_band
        self.targets = tuple(str(target) for target in targets)
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        bits = _registry_bits("modis_state_qa", self.targets)
        qa = _select_qa(gt, self.qa_band, self.axis)
        masks = [_decode_bits(qa, bit_list, "any") for bit_list in bits.values()]
        mask = np.logical_or.reduce(masks)
        return gt.array_as_geotensor(mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "qa_band": self.qa_band,
            "targets": list(self.targets),
            "axis": self.axis,
        }


class S2Cloudless(Operator):
    """Placeholder for the optional ML-based s2cloudless mask."""

    def __init__(self, *, threshold: float = 0.4) -> None:
        self.threshold = threshold

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        raise ImportError("S2Cloudless requires the optional ML mask extra.")

    def get_config(self) -> dict[str, Any]:
        return {"threshold": self.threshold}


class OmniCloudMask(Operator):
    """Placeholder for the optional ML-based OmniCloudMask detector."""

    def __init__(self, *, checkpoint: str = "default") -> None:
        self.checkpoint = checkpoint

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        raise ImportError("OmniCloudMask requires the optional ML mask extra.")

    def get_config(self) -> dict[str, Any]:
        return {"checkpoint": self.checkpoint}


class CloudSEN12(Operator):
    """Placeholder for the optional ML-based CloudSEN12 detector."""

    def __init__(self, *, checkpoint: str = "default") -> None:
        self.checkpoint = checkpoint

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        raise ImportError("CloudSEN12 requires the optional ML mask extra.")

    def get_config(self) -> dict[str, Any]:
        return {"checkpoint": self.checkpoint}
