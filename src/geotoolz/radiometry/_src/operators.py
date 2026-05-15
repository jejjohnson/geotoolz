"""Tier-B Operators — carrier-aware radiometric transforms.

Each Operator wraps a primitive in
:mod:`geotoolz.radiometry._src.array`. Most are ufunc-pure (arithmetic,
power, clipping) so the carrier's ``__array_ufunc__`` round-trips
``transform`` / ``crs`` / ``fill_value_default`` for free; the
non-ufunc cases (``PercentileClip``) explicitly wrap via
``array_as_geotensor``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from geotoolz.core import Operator
from geotoolz.radiometry._src.array import (
    _broadcast_to_band_axis,
    dn_to_radiance,
    dn_to_reflectance,
    gamma_correct,
    min_max_normalize,
    percentile_clip,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


class ToFloat32(Operator):
    """Cast the carrier's values to ``float32``.

    The first stop on most radiometric pipelines: sensor DN are usually
    ``uint16``, and downstream arithmetic (division for indices, gain
    application, etc.) wants float. ``float32`` is the right default for
    imagery — half the memory of ``float64`` with plenty of dynamic
    range for reflectance.

    Pure ufunc — `GeoTensor.__array_ufunc__` preserves transform / CRS /
    fill-value automatically.

    Examples:
        >>> import geotoolz as gz
        >>> # Standard first stage of a TOA-reflectance pipeline:
        >>> pipe = (
        ...     gz.radiometry.ToFloat32()
        ...     | gz.radiometry.DNToReflectance(scale=1e-4)
        ... )
        >>> reflectance = pipe(uint16_dn_geotensor)
    """

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        # `astype` on a GeoTensor returns a GeoTensor with metadata
        # preserved via __array_finalize__.
        return gt.astype(np.float32)

    def get_config(self) -> dict[str, Any]:
        return {}


class DNToRadiance(Operator):
    r"""Convert DN to at-sensor radiance.

    .. math::

        L \;=\; \text{gain} \cdot DN + \text{offset}

    Pure linear decode; gain and offset come from sensor metadata. Pass
    scalars for uniform per-pixel coefficients, or 1-D sequences with
    one entry per band to apply different coefficients along the band
    axis.

    See :func:`~geotoolz.radiometry._src.array.dn_to_radiance` for the
    physics. The Operator additionally reshapes 1-D coefficients so
    they broadcast along the configured ``axis``.

    Args:
        gain: Slope of the DN→L decode. Scalar or per-band sequence.
        offset: Intercept. Default ``0.0``.
        axis: Position of the band axis when ``gain`` / ``offset`` are
            per-band sequences. Default ``0``.

    Examples:
        >>> import numpy as np
        >>> from geotoolz.radiometry import DNToRadiance
        >>> # Landsat-8 OLI per-band radiance multiplicative + additive
        >>> # constants from the MTL file (just illustrative numbers):
        >>> gains   = np.array([0.012, 0.013, 0.011, 0.009])
        >>> offsets = np.array([-60.0, -61.0, -55.0, -45.0])
        >>> dn_to_rad = DNToRadiance(gain=gains, offset=offsets)
        >>> radiance = dn_to_rad(dn_geotensor)  # shape preserved
    """

    def __init__(
        self,
        *,
        gain: float | np.ndarray | list,
        offset: float | np.ndarray | list = 0.0,
        axis: int = 0,
    ) -> None:
        self.gain = gain
        self.offset = offset
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt)
        n_bands = arr.shape[self.axis] if arr.ndim > 2 else 1
        gain = _broadcast_to_band_axis(self.gain, n_bands, self.axis, arr.ndim)
        offset = _broadcast_to_band_axis(self.offset, n_bands, self.axis, arr.ndim)
        out = dn_to_radiance(arr, gain, offset)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"gain": self.gain, "offset": self.offset, "axis": self.axis}


class DNToReflectance(Operator):
    r"""Convert DN to TOA reflectance via a quantification scale.

    .. math::

        \rho \;=\; \text{scale} \cdot (DN + \text{offset})

    The Sentinel-2 L1C shortcut: ESA's processor has already absorbed
    solar geometry into the DN, so a single scale recovers reflectance.
    For L1C scenes after 2022-01-25 the per-band ``RADIO_ADD_OFFSET``
    (``-1000``) must be applied before scaling.

    For sensors without a pre-scaled-reflectance product (raw radiance
    only), use `DNToRadiance` then call
    `georeader.reflectance.radiance_to_reflectance` (which handles
    solar geometry properly).

    Args:
        scale: Quantification scale. S2 L1C: ``1e-4``. Landsat-9 C2 SR:
            ``2.75e-5``. Scalar or per-band 1-D sequence.
        offset: Pre-scale offset. S2 L1C post-2022-01-25: ``-1000``.
            Default ``0.0``.
        axis: Band axis when ``scale`` / ``offset`` are per-band.

    Examples:
        >>> from geotoolz.radiometry import DNToReflectance
        >>> # Sentinel-2 L1C: a single global scale.
        >>> op = DNToReflectance(scale=1e-4)
        >>> reflectance = op(s2_l1c_dn_geotensor)
        >>>
        >>> # Post-2022 S2 L1C with the radiometric offset baked in:
        >>> op_v2 = DNToReflectance(scale=1e-4, offset=-1000.0)
        >>> reflectance = op_v2(s2_l1c_modern_dn_geotensor)
    """

    def __init__(
        self,
        *,
        scale: float | np.ndarray | list,
        offset: float | np.ndarray | list = 0.0,
        axis: int = 0,
    ) -> None:
        self.scale = scale
        self.offset = offset
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt)
        n_bands = arr.shape[self.axis] if arr.ndim > 2 else 1
        scale = _broadcast_to_band_axis(self.scale, n_bands, self.axis, arr.ndim)
        offset = _broadcast_to_band_axis(self.offset, n_bands, self.axis, arr.ndim)
        out = dn_to_reflectance(arr, scale, offset)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"scale": self.scale, "offset": self.offset, "axis": self.axis}


class MinMax(Operator):
    r"""Linear contrast stretch into ``[0, 1]``.

    .. math::

        y \;=\; \frac{x - v_{\min}}{v_{\max} - v_{\min}}

    Display-prep — no physics; just a fixed-bound rescale. See
    `PercentileClip` for the more robust per-scene variant.

    Args:
        vmin: Lower bound (maps to 0).
        vmax: Upper bound (maps to 1). Must be strictly greater than
            ``vmin``.
        clip: Whether to clamp output to ``[0, 1]``. Default ``True``.

    Examples:
        >>> from geotoolz.radiometry import MinMax
        >>> # Stretch reflectance in [0, 0.3] to display range.
        >>> op = MinMax(vmin=0.0, vmax=0.3)
        >>> display_ready = op(reflectance_geotensor)
    """

    def __init__(self, *, vmin: float, vmax: float, clip: bool = True) -> None:
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = min_max_normalize(np.asarray(gt), self.vmin, self.vmax, clip=self.clip)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"vmin": self.vmin, "vmax": self.vmax, "clip": self.clip}


class PercentileClip(Operator):
    r"""Per-band robust contrast stretch using percentile thresholds.

    Computes :math:`v_{lo} = P_{p_{\min}}(\text{arr})` and
    :math:`v_{hi} = P_{p_{\max}}(\text{arr})` over the configured
    ``axis`` and rescales each slice into ``[0, 1]``.

    Robust against bright outliers (cumulus, specular reflection, sensor
    saturation): a tiny sub-percent population of bright pixels won't
    crush the rest of the histogram the way a true min/max would.

    Args:
        p_min: Lower percentile. Default ``2.0``.
        p_max: Upper percentile. Default ``98.0``.
        axis: Axis (or tuple) to compute percentiles over.
            ``(-2, -1)`` is per-band/-time. ``None`` is global.

    Examples:
        >>> from geotoolz.radiometry import PercentileClip
        >>> # Standard "satellite RGB" stretch -- per-band 2-98 %.
        >>> op = PercentileClip(p_min=2.0, p_max=98.0)
        >>> rgb = op(reflectance_geotensor)
    """

    def __init__(
        self,
        *,
        p_min: float = 2.0,
        p_max: float = 98.0,
        axis: int | tuple[int, ...] | None = (-2, -1),
    ) -> None:
        self.p_min = p_min
        self.p_max = p_max
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = percentile_clip(
            np.asarray(gt), p_min=self.p_min, p_max=self.p_max, axis=self.axis
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"p_min": self.p_min, "p_max": self.p_max, "axis": self.axis}


class Gamma(Operator):
    r"""Power-law gamma correction.

    .. math::

        y \;=\; x^{1/\gamma}

    Display-prep — ``g > 1`` brightens midtones, ``g < 1`` darkens
    them. A gentle ``g = 1.2`` is the workhorse "satellite RGB pop"
    default; sRGB encoding uses ``g ≈ 2.2``.

    Args:
        g: Gamma factor (must be strictly positive). Default ``1.2``.

    Examples:
        >>> import geotoolz as gz
        >>> # Classic display pipeline: stretch, then gamma-brighten.
        >>> pipe = (
        ...     gz.radiometry.PercentileClip()
        ...     | gz.radiometry.Gamma(g=1.4)
        ... )
        >>> rgb = pipe(reflectance_geotensor)
    """

    def __init__(self, *, g: float = 1.2) -> None:
        self.g = g

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = gamma_correct(np.asarray(gt), g=self.g)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"g": self.g}
