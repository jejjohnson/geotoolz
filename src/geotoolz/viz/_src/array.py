"""Tier-A primitives for display-ready remote-sensing visualizations.

These pure-numpy helpers cover the steps that ``radiometry`` doesn't:
casting to ``uint8`` for display, mapping a single-band array through
a matplotlib colormap, hillshading, and alpha blending. The float-only
contrast stretches (`MinMax`, `PercentileClip`) live in
:mod:`geotoolz.radiometry`; this module composes them with a byte
cast rather than re-implementing the percentile math.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from jaxtyping import Float, Int, Num, Shaped, UInt8

from geotoolz._src.shape import single_band
from geotoolz._src.stretch import percentile_stretch


Color = tuple[float, float, float, float]


def composite(
    arr: Shaped[np.ndarray, "*dims"],
    bands: Sequence[int],
    *,
    axis: int = 0,
) -> Shaped[np.ndarray, "*dims"]:
    """Select display bands from ``arr`` while preserving spatial axes.

    Thin wrapper around :func:`numpy.take` that accepts any integer
    iterable so band tuples and lists round-trip through hydra-zen.

    Args:
        arr: Input cube, typically ``(C, H, W)``. Any shape works as
            long as ``axis`` indexes the band dimension.
        bands: Integer band positions to select, in output order.
            Repeats are allowed (e.g. grayscale-to-RGB).
        axis: Band axis to take along. Default ``0``.

    Returns:
        Array with ``len(bands)`` slices along ``axis``; all other axes
        are unchanged. Same dtype as ``arr``.
    """
    return np.take(arr, list(bands), axis=axis)


def stretch_to_uint8(
    arr: Num[np.ndarray, "*batch h w"],
    *,
    lower: float = 2.0,
    upper: float = 98.0,
    per_band: bool = True,
) -> UInt8[np.ndarray, "*batch h w"]:
    """Percentile stretch an array into display-ready ``uint8`` values.

    A NaN-safe variant of :func:`geotoolz.radiometry.percentile_clip`
    that additionally rescales the unit-interval output to byte range.
    Distinct from radiometry's primitive in two ways:

    1. Uses ``np.nanpercentile`` so cloud / nodata pixels don't pull
       the bounds toward the extremes.
    2. Returns ``uint8`` rather than ``float`` for direct use with
       PIL / matplotlib display sinks.

    Use radiometry's ``PercentileClip`` + ``MinMax`` if you need the
    intermediate floats for further math.
    """
    if upper <= lower:
        raise ValueError(
            f"stretch_to_uint8 requires upper > lower; got {lower=}, {upper=}"
        )
    axis = (-2, -1) if per_band and arr.ndim > 2 else None
    scaled = percentile_stretch(arr, lower, upper, axis=axis)
    return np.nan_to_num(scaled * 255.0, nan=0.0).astype(np.uint8)


def gamma_correct_display(
    arr: Num[np.ndarray, "*dims"], *, gamma: float = 1.0, inplace_norm: bool = True
) -> Num[np.ndarray, "*dims"]:
    """Apply power-law gamma correction to display-range arrays.

    Gamma is a unit-interval operation: the math
    ``out = clip(arr, 0, 1) ** (1 / gamma)`` is only meaningful when the
    input is normalised to ``[0, 1]``. Display arrays, however, arrive
    in two flavours — float in ``[0, 1]`` *or* integer in ``[0, 255]``
    (uint8) / ``[0, 65535]`` (uint16). Without normalisation, an integer
    input is raised to ``1 / gamma`` directly, giving e.g.
    ``256 ** 0.5 = 16`` — not a display-correct gamma transform.

    With ``inplace_norm=True`` (the default), integer inputs are scaled
    by their dtype maximum into ``[0, 1]``, the gamma exponent is
    applied, and the result is scaled back to the original integer
    dtype's full range. Floating-point inputs are assumed to already be
    in ``[0, 1]`` and are left unscaled.

    Args:
        arr: Display array. Integer (``uint8`` / ``uint16``) or float.
        gamma: Strictly positive gamma factor. ``> 1`` brightens
            midtones; ``< 1`` darkens them.
        inplace_norm: When ``True`` (default), normalise integer inputs
            to ``[0, 1]`` before the exponent and scale back. When
            ``False``, apply ``arr ** (1 / gamma)`` directly — only set
            this if you've already normalised upstream.

    Returns:
        Gamma-corrected array of the same shape and dtype as ``arr``.
    """
    if gamma <= 0:
        raise ValueError(f"gamma_correct_display requires gamma > 0; got {gamma}")
    exponent = 1.0 / gamma
    if not inplace_norm or not np.issubdtype(arr.dtype, np.integer):
        return np.maximum(arr, 0.0) ** exponent
    dtype_max = float(np.iinfo(arr.dtype).max)
    normed = np.clip(arr.astype(np.float64) / dtype_max, 0.0, 1.0)
    corrected = normed**exponent
    return np.clip(corrected * dtype_max, 0.0, dtype_max).astype(arr.dtype)


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
) -> UInt8[np.ndarray, "h w"]:
    """Compute GDAL-style hillshade as a single ``uint8`` band.

    Slope and aspect are derived from the DEM gradient (in map units,
    via ``x_resolution`` / ``y_resolution``) and combined with a sun
    position to give the classic terrain-shading effect. Illumination
    is clipped to ``[0, 1]`` and scaled to byte range.

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

    Returns:
        ``(H, W)`` ``uint8`` shading band (0 = fully shaded, 255 =
        fully lit).

    Raises:
        ValueError: If ``dem`` is not a single-band map.
    """
    band = single_band(dem, name="hillshade").astype(np.float64, copy=False)
    if altitude_deg >= 90.0:
        return np.full(band.shape, 255, dtype=np.uint8)

    dy, dx = np.gradient(band * z_factor, abs(y_resolution), abs(x_resolution))
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    # Convert geographic azimuth (clockwise from north) to the mathematical
    # angle expected by the aspect term (counter-clockwise from east).
    azimuth = np.deg2rad(360.0 - azimuth_deg + 90.0)
    altitude = np.deg2rad(altitude_deg)
    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(
        slope
    ) * np.cos(azimuth - aspect)
    return (np.clip(shaded, 0.0, 1.0) * 255.0).astype(np.uint8)


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
    return np.clip(np.concatenate([rgb, out_alpha], axis=0) * 255.0, 0.0, 255.0).astype(
        np.uint8
    )


def ensure_rgba(
    arr: Num[np.ndarray, "h w"] | Num[np.ndarray, "c h w"],
) -> UInt8[np.ndarray, "4 h w"]:
    """Return ``arr`` as a four-band uint8 RGBA image.

    Float inputs with all finite values in ``[0, 1]`` are scaled to byte
    range; other numeric inputs are treated as already display-scaled and
    clipped into ``[0, 255]``.
    """
    values = np.asarray(arr)
    if values.ndim == 2:
        values = np.repeat(values[None, ...], 3, axis=0)
    if values.ndim != 3:
        raise ValueError(
            f"display arrays must be 2D, RGB, or RGBA; got shape {values.shape}"
        )
    if values.shape[0] == 4:
        rgba = values
    elif values.shape[0] == 3:
        alpha = np.full((1, *values.shape[-2:]), 255, dtype=values.dtype)
        rgba = np.concatenate([values, alpha], axis=0)
    else:
        raise ValueError(
            f"display arrays must have 3 or 4 bands; got {values.shape[0]}"
        )
    if rgba.dtype == np.uint8:
        return rgba.copy()
    max_value = np.nanmax(rgba) if np.isfinite(rgba).any() else 0.0
    scaled = rgba if max_value > 1.0 else rgba * 255.0
    return np.nan_to_num(np.clip(scaled, 0.0, 255.0), nan=0.0).astype(np.uint8)


def _float_rgba_to_uint8(
    rgba: Float[np.ndarray, "h w 4"],
) -> UInt8[np.ndarray, "4 h w"]:
    return np.clip(np.moveaxis(rgba, -1, 0) * 255.0, 0.0, 255.0).astype(np.uint8)
