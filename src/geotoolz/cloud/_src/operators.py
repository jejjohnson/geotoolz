"""Tier-B Operators — carrier-aware cloud / mask helpers.

Mask-extraction operators (`MaskFromQABits`, `MaskFromSCL`) return a
*single-band* boolean `GeoTensor` with the same spatial transform / CRS
as the input. `ApplyMask` consumes either a precomputed mask or a mask
*Operator* (which it runs on its input before applying) and produces
the masked carrier.

`MaskValid` is the tiniest member of the family — True where the input
carrier doesn't equal its `fill_value_default` — and exists so users
can write ``ApplyMask(mask=MaskValid())`` to drop no-data pixels
without writing the condition by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from geotoolz.cloud._src.array import apply_mask, mask_from_qa_bits, mask_from_scl
from geotoolz.core import Operator


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


class MaskFromQABits(Operator):
    """Extract a boolean mask from a Landsat-style bitmask QA band.

    Pulls the band at index ``band_idx`` from the carrier, then returns
    True where ANY of the supplied ``bits`` is set.

    For Landsat-8 Collection-2 ``QA_PIXEL``: cloud is bit 3, cirrus is
    bit 2, cloud-shadow is bit 4. So
    ``MaskFromQABits(band_idx=-1, bits=[2, 3, 4])`` builds the standard
    "everything non-clear" mask.

    Args:
        band_idx: Index of the QA band along the carrier's channel
            axis. ``-1`` for "last band" is the typical convention.
        bits: Bit positions to test.
        axis: Position of the band axis. Default ``0``.
        invert: Return True for *unset* bits instead.

    Examples:
        >>> from geotoolz.cloud import MaskFromQABits
        >>> # Landsat-8 cloud + cirrus + shadow.
        >>> qa_op = MaskFromQABits(band_idx=-1, bits=[2, 3, 4])
        >>> cloudy = qa_op(landsat_stack_geotensor)  # (H, W) bool GeoTensor
    """

    def __init__(
        self,
        *,
        band_idx: int,
        bits: Sequence[int],
        axis: int = 0,
        invert: bool = False,
    ) -> None:
        self.band_idx = band_idx
        self.bits = tuple(bits)
        self.axis = axis
        self.invert = invert

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        qa = np.take(np.asarray(gt), self.band_idx, axis=self.axis)
        mask = mask_from_qa_bits(qa, self.bits, invert=self.invert)
        return gt.array_as_geotensor(mask)

    def get_config(self) -> dict[str, Any]:
        return {
            "band_idx": self.band_idx,
            "bits": list(self.bits),
            "axis": self.axis,
            "invert": self.invert,
        }


class MaskFromSCL(Operator):
    """Extract a boolean mask from a Sentinel-2 SCL band by class membership.

    Pulls the band at index ``band_idx`` from the carrier, returns True
    where the SCL value equals any of the listed ``classes``.

    Pair with `geotoolz.cloud.SCL_CLOUDS` (the canonical cloud-class
    bundle) for "everything cloudy" or pass an explicit list of `SCL`
    enum members for finer control.

    Args:
        band_idx: Index of the SCL band along the carrier's channel
            axis. For Sentinel-2 stacks where SCL is appended last,
            ``-1`` works; for the per-band-resolution L2A products
            where SCL is a separate file, the band-stack convention
            depends on how you loaded it.
        classes: SCL class IDs to match. Accepts raw ints or `SCL`
            enum members.
        axis: Position of the band axis. Default ``0``.
        invert: When True, return True where the SCL value is NOT in
            ``classes`` (keep-only-these mask).

    Examples:
        >>> from geotoolz.cloud import MaskFromSCL, SCL_CLOUDS
        >>> # Standard "drop everything cloudy" mask.
        >>> cloud_op = MaskFromSCL(band_idx=-1, classes=SCL_CLOUDS)
        >>> cloudy = cloud_op(s2_l2a_geotensor)
        >>>
        >>> # Or: "keep only vegetation and water"
        >>> from geotoolz.cloud import SCL
        >>> keep = MaskFromSCL(
        ...     band_idx=-1,
        ...     classes=[SCL.VEGETATION, SCL.WATER],
        ...     invert=True,  # True = mask out everything NOT in classes
        ... )
    """

    def __init__(
        self,
        *,
        band_idx: int,
        classes: Sequence[int],
        axis: int = 0,
        invert: bool = False,
    ) -> None:
        if len(classes) == 0:
            raise ValueError("MaskFromSCL: `classes` must not be empty")
        self.band_idx = band_idx
        # Cast through int so enum members + raw ints both serialize.
        self.classes = tuple(int(c) for c in classes)
        self.axis = axis
        self.invert = invert

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        scl = np.take(np.asarray(gt), self.band_idx, axis=self.axis)
        mask = mask_from_scl(scl, self.classes, invert=self.invert)
        return gt.array_as_geotensor(mask)

    def get_config(self) -> dict[str, Any]:
        return {
            "band_idx": self.band_idx,
            "classes": list(self.classes),
            "axis": self.axis,
            "invert": self.invert,
        }


class MaskValid(Operator):
    """Mark pixels equal to a sentinel "invalid" value.

    Returns a boolean carrier where True means "this pixel is invalid
    and should probably be dropped." Uses the carrier's
    ``fill_value_default`` by default; pass ``invalid_value`` explicitly
    to mark a different sentinel.

    The mask broadcasts across all bands — a pixel is invalid if it
    matches the sentinel in ANY band. (Use a per-band check if you'd
    rather mask only where every band is invalid.)

    Args:
        invalid_value: Sentinel value treated as "invalid". ``None``
            uses the carrier's ``fill_value_default``.
        axis: Position of the band axis (used for the per-pixel
            ANY-band reduction). Default ``0``.

    Examples:
        >>> from geotoolz.cloud import MaskValid, ApplyMask
        >>> mask = MaskValid()(geotensor)        # True = invalid
        >>> clean = ApplyMask(mask=MaskValid())(geotensor)  # NaN-fills invalid
    """

    def __init__(
        self, *, invalid_value: float | int | None = None, axis: int = 0
    ) -> None:
        self.invalid_value = invalid_value
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        sentinel = (
            self.invalid_value
            if self.invalid_value is not None
            else gt.fill_value_default
        )
        if sentinel is None:
            raise ValueError(
                "MaskValid: no invalid_value provided and the carrier has no "
                "fill_value_default. Pass `invalid_value=...` explicitly."
            )
        arr = np.asarray(gt)
        if arr.ndim <= 2:
            mask = arr == sentinel
        else:
            mask = np.any(arr == sentinel, axis=self.axis)
        return gt.array_as_geotensor(mask)

    def get_config(self) -> dict[str, Any]:
        return {"invalid_value": self.invalid_value, "axis": self.axis}


class ApplyMask(Operator):
    """Apply a mask (carrier or Operator) to the input, filling masked pixels.

    Convention: the mask is True where pixels should be *masked out*
    (the convention `MaskFromQABits` / `MaskFromSCL` / `MaskValid` all
    follow). Use ``invert=True`` to flip it.

    The ``mask`` argument can be:

    - A precomputed boolean array (or `GeoTensor`), useful when the
      mask comes from somewhere outside the operator pipeline.
    - An `Operator` instance — `ApplyMask` will *run it on the same
      input* first, then apply the result. This is the composition
      pattern: ``ApplyMask(mask=MaskFromSCL(band_idx=-1, classes=...))``.

    Args:
        mask: Boolean array, `GeoTensor`, or `Operator` that produces
            one when called on the input.
        fill_value: Value substituted where the mask says "drop".
            Default ``np.nan``.
        invert: When True, mask everything where the mask is False
            instead of where it is True (keep-only-these mode).

    Examples:
        >>> import geotoolz as gz
        >>> # The canonical S2 L2A cloud-mask + NDVI pipeline:
        >>> clean_ndvi = (
        ...     gz.radiometry.DNToReflectance(scale=1e-4)
        ...     | gz.cloud.ApplyMask(
        ...         mask=gz.cloud.MaskFromSCL(
        ...             band_idx=-1, classes=gz.cloud.SCL_CLOUDS,
        ...         ),
        ...     )
        ...     | gz.indices.NDVI(nir_idx=7, red_idx=3)
        ... )
        >>> v = clean_ndvi(s2_dn_geotensor)
    """

    def __init__(
        self,
        *,
        mask: Operator | np.ndarray | Any,
        fill_value: float = float("nan"),
        invert: bool = False,
    ) -> None:
        self.mask = mask
        self.fill_value = fill_value
        self.invert = invert

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        # Resolve the mask: if it's an Operator, run it on the input.
        if isinstance(self.mask, Operator):
            mask_arr = np.asarray(self.mask(gt))
        else:
            mask_arr = np.asarray(self.mask)
        out = apply_mask(
            np.asarray(gt), mask_arr, fill_value=self.fill_value, invert=self.invert
        )
        return gt.array_as_geotensor(out)

    # Operator-valued masks round-trip cleanly (nested {class, config}),
    # but raw-array masks only emit a shape/dtype summary in `get_config`.
    # Mark the class non-YAML-safe so callers know to handle the
    # array-mask case explicitly (set to True under either codepath for
    # simplicity, per the design contract for closure-carrying ops).
    forbid_in_yaml: ClassVar[bool] = True

    def get_config(self) -> dict[str, Any]:
        if isinstance(self.mask, Operator):
            mask_config: Any = {
                "class": type(self.mask).__name__,
                "config": self.mask.get_config(),
            }
        else:
            arr = np.asarray(self.mask)
            mask_config = {
                "type": "ndarray",
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
        return {
            "mask": mask_config,
            "fill_value": self.fill_value,
            "invert": self.invert,
        }
