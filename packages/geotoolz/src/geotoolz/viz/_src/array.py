"""Tier-A primitives for display-ready remote-sensing visualizations.

These pure-numpy helpers cover the steps that ``radiometry`` doesn't:
casting to ``uint8`` for display, mapping a single-band array through
a matplotlib colormap, hillshading, and alpha blending. The float
contrast stretch and gamma live in :mod:`geotoolz.radiometry`
(``percentile_clip`` / ``gamma_correct``, both NaN-safe); this module
composes them with a rounded byte cast rather than re-implementing the
math. Every float -> ``uint8`` cast here rounds to the nearest byte
(``np.rint``) instead of truncating.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import einx
import numpy as np
from jaxtyping import Bool, Float, Int, Num, UInt8

from geotoolz._src.shape import single_band
from geotoolz.radiometry._src.array import gamma_correct, percentile_clip


Color = tuple[float, float, float, float]


def _unit_to_uint8(arr: Float[np.ndarray, "*dims"]) -> UInt8[np.ndarray, "*dims"]:
    """Map unit-interval floats to bytes, rounding to nearest; NaN -> ``0``."""
    scaled = np.clip(np.nan_to_num(np.asarray(arr) * 255.0, nan=0.0), 0.0, 255.0)
    return np.rint(scaled).astype(np.uint8)


def stretch_to_uint8(
    arr: Num[np.ndarray, "*batch y x"],
    *,
    lower: float = 2.0,
    upper: float = 98.0,
    reduce_axes: int | tuple[int, ...] | None = (-2, -1),
) -> UInt8[np.ndarray, "*batch y x"]:
    """Percentile stretch an array into display-ready ``uint8`` values.

    Exactly :func:`geotoolz.radiometry.percentile_clip` (NaN-safe:
    percentiles via ``np.nanpercentile``) followed by a rounded byte
    cast; NaN pixels map to ``0``. Use ``percentile_clip`` directly if
    you need the intermediate ``[0, 1]`` floats for further math.

    Args:
        arr: Input array, typically ``(C, H, W)``.
        lower: Lower percentile. Default ``2.0``.
        upper: Upper percentile. Default ``98.0``; must exceed ``lower``.
        reduce_axes: Axes the percentiles are computed over. Default
            ``(-2, -1)`` stretches each band (and frame) independently;
            ``None`` uses one global pair of thresholds.

    Returns:
        ``uint8`` array of the same shape.

    Raises:
        ValueError: If ``upper <= lower``.
    """
    return _unit_to_uint8(percentile_clip(arr, lower, upper, reduce_axes=reduce_axes))


def gamma_correct_display(
    arr: Num[np.ndarray, "*dims"], *, gamma: float = 1.0, inplace_norm: bool = True
) -> Num[np.ndarray, "*dims"]:
    """Apply power-law gamma correction to display-range arrays.

    Gamma is a unit-interval operation: the math
    ``out = clip(arr, 0, 1) ** (1 / gamma)`` (via
    :func:`geotoolz.radiometry.gamma_correct`) is only meaningful when
    the input is normalised to ``[0, 1]``. Display arrays, however,
    arrive in two flavours — float in ``[0, 1]`` *or* integer in
    ``[0, 255]`` (uint8) / ``[0, 65535]`` (uint16). Without
    normalisation, an integer input is raised to ``1 / gamma`` directly,
    giving e.g. ``256 ** 0.5 = 16`` — not a display-correct gamma
    transform.

    With ``inplace_norm=True`` (the default), integer inputs are scaled
    by their dtype maximum into ``[0, 1]``, the gamma exponent is
    applied, and the result is rounded back to the original integer
    dtype's full range. Floating-point inputs are clipped to ``[0, 1]``
    (the display range) before the exponent; NaN stays NaN.

    Args:
        arr: Display array. Integer (``uint8`` / ``uint16``) or float.
        gamma: Strictly positive gamma factor. ``> 1`` brightens
            midtones; ``< 1`` darkens them.
        inplace_norm: When ``True`` (default), normalise integer inputs
            to ``[0, 1]`` before the exponent and scale back. When
            ``False``, integer inputs get ``arr ** (1 / gamma)`` directly
            (a float result) — only set this if you've already
            normalised upstream. Float inputs are unaffected.

    Returns:
        Gamma-corrected array of the same shape; same dtype as ``arr``
        for float input and for integer input with ``inplace_norm``.

    Raises:
        ValueError: If ``gamma <= 0``.
    """
    if not np.issubdtype(arr.dtype, np.integer):
        return gamma_correct(np.clip(arr, 0.0, 1.0), gamma=gamma)
    if not inplace_norm:
        return gamma_correct(arr, gamma=gamma)
    dtype_max = float(np.iinfo(arr.dtype).max)
    normed = np.clip(arr.astype(np.float64) / dtype_max, 0.0, 1.0)
    corrected = gamma_correct(normed, gamma=gamma)
    return np.rint(np.clip(corrected * dtype_max, 0.0, dtype_max)).astype(arr.dtype)


def rgb_recipe(
    channels: Float[np.ndarray, "3 h w"],
    *,
    vmin: float | tuple[float, float, float] = 0.0,
    vmax: float | tuple[float, float, float] = 1.0,
    gamma: float | tuple[float, float, float] = 1.0,
) -> Float[np.ndarray, "3 h w"]:
    r"""Stretch and gamma-correct three channels into a display RGB.

    Each channel ``c`` is mapped through a fixed linear stretch and a power
    law — the "RGB recipe" form of the operational satellite RGB guides:

    .. math::

        y_c \;=\; \operatorname{clip}\!\Bigl(
            \frac{x_c - v_{\min,c}}{v_{\max,c} - v_{\min,c}}, 0, 1
        \Bigr)^{1/\gamma_c}

    A ``vmin`` above ``vmax`` inverts the channel (colder brightness
    temperatures brighter, for instance). ``NaN`` inputs stay ``NaN``.

    Args:
        channels: The red, green and blue inputs stacked on the first axis.
        vmin: Value mapped to 0, per channel or one for all.
        vmax: Value mapped to 1, per channel or one for all.
        gamma: Gamma per channel or one for all (``> 0``; ``> 1``
            brightens midtones).

    Returns:
        ``float32`` array of the same shape, in ``[0, 1]`` (``NaN`` kept).

    Raises:
        ValueError: ``channels`` does not have 3 entries on its first axis,
            a ``vmin`` equals its ``vmax``, or a gamma is not positive.
    """
    arr = np.asarray(channels, dtype=np.float32)
    if arr.shape[0] != 3:
        raise ValueError(f"rgb_recipe needs 3 channels; got shape {arr.shape}.")
    lo, hi, g = (
        np.broadcast_to(np.asarray(v, dtype=np.float64), (3,))
        for v in (vmin, vmax, gamma)
    )
    if np.any(lo == hi):
        raise ValueError(f"vmin and vmax must differ per channel; got {lo}, {hi}.")
    if np.any(g <= 0):
        raise ValueError(f"gamma must be positive; got {g}.")
    out = np.empty_like(arr)
    for c in range(3):
        unit = np.clip((arr[c] - lo[c]) / (hi[c] - lo[c]), 0.0, 1.0)
        out[c] = unit ** (1.0 / g[c])
    return out


def rgba_from_scalar(
    arr: Num[np.ndarray, "h w"] | Num[np.ndarray, "1 h w"],
    cmap: Callable[[np.ndarray], np.ndarray],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    nan_color: Color = (0.0, 0.0, 0.0, 0.0),
) -> UInt8[np.ndarray, "4 h w"]:
    """Map a single-band array to a four-band uint8 RGBA image.

    Values are linearly normalised into ``[0, 1]`` between ``vmin`` and
    ``vmax`` (auto-detected from the finite values when omitted), then
    looked up through ``cmap``. Non-finite pixels are painted with
    ``nan_color`` rather than whatever the colormap does at 0.

    Args:
        arr: Single-band map, ``(H, W)`` or ``(1, H, W)``.
        cmap: Matplotlib-style callable mapping unit-interval floats to
            ``(..., 4)`` RGBA floats in ``[0, 1]``.
        vmin: Lower normalisation bound. ``None`` uses the finite
            minimum of ``arr``.
        vmax: Upper normalisation bound. ``None`` uses the finite
            maximum of ``arr``.
        nan_color: RGBA tuple in ``[0, 1]`` painted over non-finite
            pixels. Default fully transparent.

    Returns:
        Channel-first ``(4, H, W)`` ``uint8`` RGBA image.

    Raises:
        ValueError: If ``arr`` is not a single-band map.
    """
    band = single_band(arr, name="rgba_from_scalar")
    valid = np.isfinite(band)
    if vmin is None:
        vmin = float(np.nanmin(band)) if valid.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(band)) if valid.any() else 1.0
    denom = vmax - vmin
    if denom <= 0:
        denom = 1.0
    normed = np.clip((band - vmin) / denom, 0.0, 1.0)
    rgba = np.asarray(cmap(normed), dtype=np.float32)
    rgba[~valid] = np.asarray(nan_color, dtype=np.float32)
    return _float_rgba_to_uint8(rgba)


def rgba_from_categories(
    arr: Int[np.ndarray, "h w"] | Int[np.ndarray, "1 h w"],
    mapping: Mapping[int, Color],
    *,
    default: Color = (0.0, 0.0, 0.0, 0.0),
) -> UInt8[np.ndarray, "4 h w"]:
    """Map integer classes to a four-band uint8 RGBA image.

    Categorical-label rendering: each class ID in ``mapping`` paints its
    pixels with the associated colour; pixels whose value is not in the
    mapping fall back to ``default``.

    Args:
        arr: Single-band label map, ``(H, W)`` or ``(1, H, W)``.
        mapping: ``{class_id: (r, g, b, a)}`` lookup table with float
            components in ``[0, 1]``.
        default: RGBA colour for unmapped classes. Default fully
            transparent.

    Returns:
        Channel-first ``(4, H, W)`` ``uint8`` RGBA image.

    Raises:
        ValueError: If ``arr`` is not a single-band map.
    """
    band = single_band(arr, name="rgba_from_categories")
    rgba = np.zeros((*band.shape, 4), dtype=np.float32)
    rgba[...] = np.asarray(default, dtype=np.float32)
    for value, color in mapping.items():
        rgba[band == value] = np.asarray(color, dtype=np.float32)
    return _float_rgba_to_uint8(rgba)


def hillshade(
    dem: Num[np.ndarray, "h w"] | Num[np.ndarray, "1 h w"],
    *,
    x_resolution: float = 1.0,
    y_resolution: float = 1.0,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
    valid: Bool[np.ndarray, "h w"] | None = None,
) -> UInt8[np.ndarray, "h w"]:
    """Compute GDAL-style hillshade as a single ``uint8`` band.

    Slope and aspect are derived from the DEM gradient (in map units,
    via ``x_resolution`` / ``y_resolution``) and combined with a sun
    position to give the classic terrain-shading effect. Illumination
    is clipped to ``[0, 1]`` and scaled to byte range. The DEM is
    assumed north-up (row 0 is the northern edge, as in a standard
    geotransform), so for a planar DEM the result matches
    ``matplotlib.colors.LightSource(azimuth_deg, altitude_deg).hillshade``
    (up to byte quantization; matplotlib additionally contrast-stretches
    non-constant outputs).

    Args:
        dem: Elevation map, ``(H, W)`` or ``(1, H, W)``, in the same
            linear units as the resolutions (after ``z_factor``).
        x_resolution: Pixel width in map units. Default ``1.0``.
        y_resolution: Pixel height in map units. Default ``1.0``.
        azimuth_deg: Sun azimuth in degrees clockwise from north.
            Default ``315`` (NW — the cartographic convention).
        altitude_deg: Sun elevation in degrees above the horizon.
            Default ``45``. Values ``>= 90`` short-circuit to a flat
            fully lit (255) image.
        z_factor: Vertical exaggeration applied to the DEM before the
            gradient. Default ``1.0``.
        valid: Optional ``(H, W)`` mask of valid DEM pixels. Invalid
            pixels never enter a neighbour's gradient (a one-sided
            difference is used instead, or a flat ``0`` slope when both
            neighbours along an axis are invalid) and are written as
            ``0`` in the output. ``None`` (default) treats every pixel
            as valid.

    Returns:
        ``(H, W)`` ``uint8`` shading band (0 = fully shaded, 255 =
        fully lit).

    Raises:
        ValueError: If ``dem`` is not a single-band map.
    """
    band = single_band(dem, name="hillshade").astype(np.float64, copy=False)
    invalid = None if valid is None else ~np.asarray(valid, dtype=bool)
    if invalid is not None and not invalid.any():
        invalid = None
    if altitude_deg >= 90.0:
        out = np.full(band.shape, 255, dtype=np.uint8)
        if invalid is not None:
            out[invalid] = 0
        return out

    if invalid is None:
        dy, dx = np.gradient(band * z_factor, abs(y_resolution), abs(x_resolution))
    else:
        masked = np.where(invalid, np.nan, band * z_factor)
        dy = _nan_gradient(masked, abs(y_resolution), axis=0)
        dx = _nan_gradient(masked, abs(x_resolution), axis=1)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    # Aspect = math angle (CCW from east) of the surface normal's
    # horizontal part, (-dz/deast, -dz/dnorth). `dy` is the gradient
    # along rows, i.e. towards the *south* (row 0 is north), so
    # dz/dnorth = -dy and the normal points along (-dx, dy).
    aspect = np.arctan2(dy, -dx)
    # Convert geographic azimuth (clockwise from north) to the mathematical
    # angle expected by the aspect term (counter-clockwise from east).
    azimuth = np.deg2rad(360.0 - azimuth_deg + 90.0)
    altitude = np.deg2rad(altitude_deg)
    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(
        slope
    ) * np.cos(azimuth - aspect)
    out = _unit_to_uint8(np.clip(shaded, 0.0, 1.0))
    if invalid is not None:
        out[invalid] = 0
    return out


def _nan_gradient(
    values: Float[np.ndarray, "h w"], spacing: float, *, axis: int
) -> Float[np.ndarray, "h w"]:
    """First-order gradient along ``axis`` that never reads a NaN neighbour.

    Central difference where both neighbours are finite, one-sided where
    only one is (matching :func:`numpy.gradient` at the edges), ``0``
    where neither is.
    """
    moved = np.moveaxis(values, axis, 0)
    diff = np.diff(moved, axis=0) / spacing  # diff[i] = z[i + 1] - z[i]
    pad = np.full((1, *moved.shape[1:]), np.nan)
    forward = np.concatenate([diff, pad], axis=0)
    backward = np.concatenate([pad, diff], axis=0)
    both = np.stack([forward, backward])
    count = np.isfinite(both).sum(axis=0)
    total = np.nansum(both, axis=0)
    grad = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return np.moveaxis(grad, 0, axis)


def blend_rgba(
    background: Num[np.ndarray, "h w"] | Num[np.ndarray, "c h w"],
    foreground: Num[np.ndarray, "h w"] | Num[np.ndarray, "c h w"],
    *,
    alpha: float = 0.6,
    mode: str = "alpha",
) -> UInt8[np.ndarray, "4 h w"]:
    """Blend two display images and return uint8 RGBA.

    Both inputs are first promoted through :func:`ensure_rgba` (2-D
    grayscale, 3-band RGB, or 4-band RGBA in). The foreground's alpha
    channel is scaled by ``alpha`` and composed over the background
    using source-over alpha composition, so partially transparent
    layers accumulate opacity correctly.

    Args:
        background: Bottom layer — ``(H, W)``, ``(3, H, W)``, or
            ``(4, H, W)``.
        foreground: Top layer on the same pixel grid.
        alpha: Global foreground opacity in ``[0, 1]``. ``0`` returns
            the background (as RGBA) untouched. Default ``0.6``.
        mode: Blend mode for the RGB channels — ``"alpha"`` (normal
            source-over), ``"multiply"``, or ``"screen"``. Default
            ``"alpha"``.

    Returns:
        ``(4, H, W)`` ``uint8`` RGBA composite.

    Raises:
        ValueError: If ``alpha`` is outside ``[0, 1]``, the spatial
            grids differ, or ``mode`` is not a supported blend mode.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"blend_rgba requires alpha in [0, 1]; got {alpha}")
    bg = ensure_rgba(background)
    fg = ensure_rgba(foreground)
    if bg.shape[-2:] != fg.shape[-2:]:
        raise ValueError(
            f"background and foreground grids must match; got {bg.shape=} {fg.shape=}"
        )
    if alpha == 0.0:
        return bg

    bg_f = bg.astype(np.float32) / 255.0
    fg_f = fg.astype(np.float32) / 255.0
    fg_alpha = np.clip(fg_f[3:4] * alpha, 0.0, 1.0)
    if mode == "alpha":
        rgb = fg_f[:3] * fg_alpha + bg_f[:3] * (1.0 - fg_alpha)
    elif mode == "multiply":
        rgb = (bg_f[:3] * fg_f[:3]) * fg_alpha + bg_f[:3] * (1.0 - fg_alpha)
    elif mode == "screen":
        screened = 1.0 - (1.0 - bg_f[:3]) * (1.0 - fg_f[:3])
        rgb = screened * fg_alpha + bg_f[:3] * (1.0 - fg_alpha)
    else:
        expected = "'alpha', 'multiply', or 'screen'"
        raise ValueError(f"unsupported overlay mode {mode!r}; expected {expected}")
    # Source-over alpha composition (foreground on top of background):
    # out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha). Equivalent to
    # max(fg_alpha, bg_alpha) only when one of them is 0 or 1; mixing
    # two partially transparent layers must accumulate opacity.
    out_alpha = fg_alpha + bg_f[3:4] * (1.0 - fg_alpha)
    return _unit_to_uint8(np.concatenate([rgb, out_alpha], axis=0))


def ensure_rgba(
    arr: Num[np.ndarray, "h w"] | Num[np.ndarray, "c h w"],
    *,
    name: str = "ensure_rgba",
) -> UInt8[np.ndarray, "4 h w"]:
    """Return ``arr`` as a four-band uint8 RGBA image.

    Accepts ``(H, W)`` grayscale (replicated to RGB), ``(3, H, W)`` RGB,
    or ``(4, H, W)`` RGBA. Float (or boolean) inputs whose finite values
    are all ``<= 1`` are treated as ``[0, 1]`` fractions and scaled to
    byte range; other numeric inputs are treated as already
    display-scaled and clipped into ``[0, 255]``. The decision is made on
    the input values alone; RGB / grayscale inputs then get an opaque
    (255) alpha plane. NaNs map to 0.

    Args:
        arr: The display array.
        name: Caller name that prefixes the error message (an operator
            passes its own).

    Raises:
        ValueError: If ``arr`` is not 2-D or a 3-/4-band cube.
    """
    values = np.asarray(arr)
    if values.ndim == 2:
        values = np.repeat(values[None, ...], 3, axis=0)
    if values.ndim != 3:
        raise ValueError(
            f"{name}: display arrays must be 2D, RGB, or RGBA; got shape {values.shape}"
        )
    if values.shape[0] not in (3, 4):
        raise ValueError(
            f"{name}: display arrays must have 3 or 4 bands; got {values.shape[0]}"
        )
    # Decide byte-vs-fraction scaling from the *input* values, before any
    # opaque alpha plane is added (a 255 alpha would otherwise make every
    # float [0, 1] RGB / grayscale input look byte-scaled -> black).
    if values.dtype == np.uint8:
        rgb_bytes = values
    else:
        is_fractional = values.dtype == np.bool_ or np.issubdtype(
            values.dtype, np.floating
        )
        finite = np.isfinite(values)
        max_value = float(np.max(values[finite])) if finite.any() else 0.0
        scaled = (
            values.astype(np.float64) * 255.0
            if is_fractional and max_value <= 1.0
            else values.astype(np.float64)
        )
        rgb_bytes = np.rint(np.nan_to_num(np.clip(scaled, 0.0, 255.0), nan=0.0)).astype(
            np.uint8
        )
    if rgb_bytes.shape[0] == 4:
        return rgb_bytes.copy()
    alpha = np.full((1, *rgb_bytes.shape[-2:]), 255, dtype=np.uint8)
    return np.concatenate([rgb_bytes, alpha], axis=0)


def _float_rgba_to_uint8(
    rgba: Float[np.ndarray, "y x 4"],
) -> UInt8[np.ndarray, "4 y x"]:
    return _unit_to_uint8(einx.id("y x c -> c y x", rgba))
