"""Tier-A primitives — pure-numpy radiometric transforms.

These are the deliberately *generic* radiometry primitives that work
regardless of sensor: they take user-supplied gain / offset / scale
constants rather than reading them from sensor metadata. Sensor-specific
TOA / BOA pipelines (e.g. Sentinel-2 L1C with the metadata-driven
``solar_irradiance`` and ``radio_add_offsets`` tables) live in the
v0.4 ``geotoolz.presets.*`` modules that will wrap
`georeader.reflectance.radiance_to_reflectance`.

Quick refresher on the physical quantities, so the function signatures
make sense:

- **Digital number (DN)** — the raw integer value the sensor reports
  (often uint16). No physical units; a sensor-specific encoding of how
  many photons hit the detector.
- **Radiance (L)** — energy per unit area per unit solid angle per unit
  wavelength reaching the sensor, typically
  :math:`W \\cdot m^{-2} \\cdot sr^{-1} \\cdot \\mu m^{-1}`. The raw
  physical quantity the sensor measures, after a linear gain/offset
  decode from DN.
- **Reflectance (ρ)** — dimensionless ratio of upwelling to downwelling
  radiance, in :math:`[0, 1]`. Top-of-atmosphere (TOA) reflectance
  corrects for solar geometry only; surface (BOA) reflectance also
  corrects for atmospheric scattering / absorption.

The DN→radiance step is a per-band linear decode the sensor specifies
in metadata: :math:`L = \\text{gain} \\cdot DN + \\text{offset}`.
The radiance→reflectance step requires solar geometry and is the
sensor-preset responsibility (or the user calls
`georeader.reflectance.radiance_to_reflectance` directly).

Sentinel-2 L1C is the convenient special case where ESA's processor
has already done the work: DN are scaled TOA reflectance, and
:math:`\\rho = (DN + \\text{offset}) / \\text{quantification\\_value}`
with a quantification value of 10000 — i.e. ``scale = 1e-4``. That's
what `dn_to_reflectance` encodes.

The remaining primitives (``min_max_normalize``, ``percentile_clip``,
``gamma_correct``) are display-prep helpers: they bring float
reflectance into the ``[0, 1]`` display range with various contrast
strategies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Float, Num, Shaped

from geotoolz._src.stretch import percentile_stretch


def dn_to_radiance(
    dn: Num[np.ndarray, "*dims"],
    gain: float | Float[np.ndarray, "*coef"],
    offset: float | Float[np.ndarray, "*coef"] = 0.0,
    scale: float | Float[np.ndarray, "*coef"] = 1.0,
) -> Float[np.ndarray, "*dims"]:
    r"""Convert raw DN to at-sensor radiance via per-band gain/offset.

    .. math::

        L \;=\; \text{gain} \cdot DN / \text{scale} + \text{offset}

    The gain and offset come from the sensor metadata file (Landsat MTL,
    Sentinel-2 MTD, EMIT L1B, etc.). Pass scalars for single-band /
    uniform gain, or 1-D arrays shaped like the band axis to apply
    different coefficients per band — broadcasting handles the
    arithmetic.

    Units of ``L`` are whatever the gain/offset are calibrated in,
    typically :math:`W \cdot m^{-2} \cdot sr^{-1} \cdot \mu m^{-1}`. The
    primitive itself is unit-agnostic.

    Args:
        dn: Raw DN array. Any shape; the gain/offset broadcast across
            it. A common pattern is ``(C, H, W)`` with per-band
            coefficients ``shape == (C, 1, 1)``.
        gain: Slope of the linear decode. Scalar or array.
        offset: Intercept. Scalar or array. Default ``0``.
        scale: Optional DN scale divisor. Default ``1``.

    Returns:
        Radiance array of the same shape as ``dn``, in float64 by
        default (cast outside if you want float32 to save memory).
    """
    return gain * dn / scale + offset


def radiance_to_dn(
    radiance: Float[np.ndarray, "*dims"],
    gain: float | Float[np.ndarray, "*coef"],
    offset: float | Float[np.ndarray, "*coef"] = 0.0,
    scale: float | Float[np.ndarray, "*coef"] = 1.0,
) -> Float[np.ndarray, "*dims"]:
    r"""Convert at-sensor radiance back to DN via the affine inverse.

    .. math::

        DN \;=\; (L - \text{offset}) \cdot \text{scale} / \text{gain}

    Args:
        radiance: At-sensor radiance array.
        gain: Slope used by `dn_to_radiance`.
        offset: Intercept used by `dn_to_radiance`. Default ``0``.
        scale: DN scale divisor used by `dn_to_radiance`. Default ``1``.

    Returns:
        DN array in floating point. Cast or round outside if integer DN are
        required.
    """
    return (radiance - offset) * scale / gain


def dn_to_reflectance(
    dn: Num[np.ndarray, "*dims"],
    scale: float | Float[np.ndarray, "*coef"],
    offset: float | Float[np.ndarray, "*coef"] = 0.0,
) -> Float[np.ndarray, "*dims"]:
    r"""Convert DN to TOA / surface reflectance via a linear affine decode.

    .. math::

        \rho \;=\; \text{scale} \cdot DN + \text{offset}

    The canonical pre-scaled-reflectance decode: ``scale`` is the slope
    (reflectance per DN unit) and ``offset`` is the *reflectance-units*
    intercept. This convention matches Landsat Collection-2 surface
    reflectance verbatim and absorbs the Sentinel-2 L1C ``RADIO_ADD_OFFSET``
    after multiplying it out (see notes below).

    Sensor-specific coefficients (illustrative — always consult the
    product MTL / metadata):

    - **Sentinel-2 L1C** (pre-2022-01-25): ``scale=1e-4, offset=0``.
    - **Sentinel-2 L1C** (post-2022-01-25): the per-band
      ``RADIO_ADD_OFFSET=-1000`` is in DN units; multiplied through the
      scale it becomes ``offset = -1000 * 1e-4 = -0.1``, so use
      ``scale=1e-4, offset=-0.1``.
    - **Landsat-8/9 Collection-2 SR**: ``scale=2.75e-5, offset=-0.2``.

    For sensors where the metadata gives gain/offset to radiance but
    *not* a direct DN→ρ shortcut, do the two-step decode via
    `dn_to_radiance` then call
    `georeader.reflectance.radiance_to_reflectance` for the proper
    solar-geometry correction.

    Args:
        dn: Raw DN array.
        scale: Quantification slope (reflectance per DN unit). Scalar
            or per-band 1-D array.
        offset: Reflectance-units intercept. Scalar or per-band 1-D
            array. Default ``0.0``.

    Returns:
        Reflectance array; values should fall in :math:`[0, 1]` for
        well-calibrated inputs.
    """
    return scale * dn + offset


def min_max_normalize(
    arr: Num[np.ndarray, "*dims"],
    vmin: float,
    vmax: float,
    *,
    clip: bool = True,
) -> Float[np.ndarray, "*dims"]:
    r"""Linearly map ``[vmin, vmax]`` to ``[0, 1]``.

    .. math::

        y \;=\; \frac{x - v_{\min}}{v_{\max} - v_{\min}}

    A pure display-prep primitive: no physics, just a contrast stretch
    that gets reflectance into the byte / display range. ``clip=True``
    forces out-of-range pixels to the endpoints; ``clip=False`` leaves
    them as-is (useful when feeding a downstream model that wants
    informative tails).

    Args:
        arr: Input array.
        vmin: Lower bound (mapped to 0). Must be strictly less than
            ``vmax``.
        vmax: Upper bound (mapped to 1).
        clip: Whether to clamp output to ``[0, 1]``. Default ``True``.

    Returns:
        Float array. Shape and broadcasting preserved.
    """
    if vmax <= vmin:
        raise ValueError(
            f"min_max_normalize requires vmax > vmin; got {vmin=}, {vmax=}"
        )
    out = (arr - vmin) / (vmax - vmin)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def percentile_clip(
    arr: Shaped[np.ndarray, "*dims"],
    p_min: float = 2.0,
    p_max: float = 98.0,
    *,
    axis: int | tuple[int, ...] | None = (-2, -1),
) -> Float[np.ndarray, "*dims"]:
    r"""Robust contrast stretch using percentile thresholds.

    Computes :math:`v_{lo} = P_{p_{\min}}(\text{arr})` and
    :math:`v_{hi} = P_{p_{\max}}(\text{arr})` over the configured
    ``axis``, then min-max normalises ``arr`` between them with
    clipping. The percentile thresholds are far more robust than fixed
    ``vmin / vmax`` against bright outliers (cumulus clouds, specular
    water glint, sensor saturation).

    Default ``axis=(-2, -1)`` computes percentiles per leading band /
    time slice — the typical "RGB display per band" mode. Pass
    ``axis=None`` for a single global percentile across the whole
    array.

    Args:
        arr: Input float array.
        p_min: Lower percentile (in ``[0, 100]``). Default ``2.0``.
        p_max: Upper percentile. Default ``98.0``. Must be strictly
            greater than ``p_min``.
        axis: Axis (or tuple of axes) to compute percentiles over.
            ``(-2, -1)`` -> per-band/-time stretch. ``None`` -> global.

    Returns:
        Float array of the same shape, values in ``[0, 1]``.
    """
    if p_max <= p_min:
        raise ValueError(
            f"percentile_clip requires p_max > p_min; got {p_min=}, {p_max=}"
        )
    return percentile_stretch(arr, p_min, p_max, axis=axis)


def gamma_correct(
    arr: Float[np.ndarray, "*dims"], g: float = 1.2
) -> Float[np.ndarray, "*dims"]:
    r"""Apply a gamma (power-law) correction.

    .. math::

        y \;=\; x^{1/\gamma}

    Display-prep helper: human visual perception of brightness is
    nonlinear, so a power-law tweak after a min-max stretch brightens
    midtones (``g > 1``) or darkens them (``g < 1``). Standard sRGB
    encoding uses ``g ≈ 2.2``; a gentle ``g = 1.2`` is a common
    "make satellite RGBs pop" default.

    Negative inputs are clipped to zero before the power to avoid
    complex-number warnings; reflectance should be non-negative anyway.

    Args:
        arr: Input float array, ideally already in ``[0, 1]``.
        g: Gamma factor. ``> 1`` brightens; ``< 1`` darkens.
            Default ``1.2``.

    Returns:
        Float array of the same shape, gamma-corrected.
    """
    if g <= 0:
        raise ValueError(f"gamma_correct requires g > 0; got {g}")
    return np.maximum(arr, 0.0) ** (1.0 / g)


def bt_from_radiance(
    radiance: Float[np.ndarray, "*dims"],
    k1: float | Float[np.ndarray, "*coef"],
    k2: float | Float[np.ndarray, "*coef"],
) -> Float[np.ndarray, "*dims"]:
    r"""Convert thermal radiance to brightness temperature.

    .. math::

        T = K_2 / \ln(K_1 / L + 1)

    Args:
        radiance: Thermal at-sensor radiance ``L``.
        k1: Planck ``K1`` constant. Scalar or per-band array.
        k2: Planck ``K2`` constant. Scalar or per-band array.

    Returns:
        Brightness temperature in Kelvin.
    """
    return k2 / np.log((k1 / radiance) + 1.0)


def dos1(
    reflectance: Float[np.ndarray, "*dims"],
    dark_percentile: float = 1.0,
    *,
    axis: int | tuple[int, ...] | None = (-2, -1),
) -> Float[np.ndarray, "*dims"]:
    """Apply a simple DOS1 dark-object subtraction to reflectance.

    Args:
        reflectance: TOA reflectance array.
        dark_percentile: Percentile used as the dark-object estimate.
        axis: Axes over which to estimate the dark-object value.

    Returns:
        Reflectance with per-band dark-object values subtracted and clipped
        to zero.
    """
    if not 0.0 <= dark_percentile <= 100.0:
        raise ValueError(
            f"dos1 requires dark_percentile in [0, 100]; got {dark_percentile}"
        )
    dark = np.nanpercentile(reflectance, dark_percentile, axis=axis, keepdims=True)
    return np.maximum(reflectance - dark, 0.0)


def _broadcast_to_band_axis(
    coef: Any, n_bands: int, axis: int, target_ndim: int
) -> np.ndarray | float:
    """Reshape a 1-D per-band coefficient so it broadcasts along ``axis``.

    Used by the DN→radiance / DN→reflectance Operators when the user
    passes a sequence-shaped gain or offset matching the band axis.
    Scalars pass through unchanged.

    Returns an array shaped like ``(1, ..., 1, n_bands, 1, ..., 1)`` —
    all singletons except at the band axis — so it broadcasts against
    an input of rank ``target_ndim``.
    """
    arr = np.asarray(coef)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"per-band coefficient must be a scalar or 1-D array; got shape {arr.shape}"
        )
    if arr.shape[0] != n_bands:
        raise ValueError(
            f"per-band coefficient length {arr.shape[0]} doesn't match "
            f"band count {n_bands}"
        )
    norm_axis = axis if axis >= 0 else target_ndim + axis
    if not 0 <= norm_axis < target_ndim:
        raise ValueError(f"axis {axis} out of range for ndim {target_ndim}")
    shape = [1] * target_ndim
    shape[norm_axis] = n_bands
    return arr.reshape(shape)
