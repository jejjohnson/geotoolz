"""Tier-A primitives for display-ready remote-sensing visualizations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


Color = tuple[float, float, float, float]


def composite(
    arr: np.ndarray,
    bands: Sequence[int],
    *,
    axis: int = 0,
) -> np.ndarray:
    """Select display bands from ``arr`` while preserving spatial axes."""
    return np.take(arr, list(bands), axis=axis)


def stretch_to_uint8(
    arr: np.ndarray,
    *,
    lower: float = 2.0,
    upper: float = 98.0,
    per_band: bool = True,
) -> np.ndarray:
    """Percentile stretch an array into display-ready ``uint8`` values."""
    if upper <= lower:
        raise ValueError(
            f"stretch_to_uint8 requires upper > lower; got {lower=}, {upper=}"
        )
    axis = (-2, -1) if per_band and arr.ndim > 2 else None
    lo = np.nanpercentile(arr, lower, axis=axis, keepdims=True)
    hi = np.nanpercentile(arr, upper, axis=axis, keepdims=True)
    denom = np.where(hi > lo, hi - lo, 1.0)
    scaled = np.clip((arr - lo) / denom, 0.0, 1.0)
    return np.nan_to_num(scaled * 255.0, nan=0.0).astype(np.uint8)


def gamma_correct_display(arr: np.ndarray, *, gamma: float = 1.0) -> np.ndarray:
    """Apply power-law gamma correction to display-range arrays."""
    if gamma <= 0:
        raise ValueError(f"gamma_correct_display requires gamma > 0; got {gamma}")
    return np.maximum(arr, 0.0) ** (1.0 / gamma)


def rgba_from_scalar(
    arr: np.ndarray,
    cmap: Callable[[np.ndarray], np.ndarray],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    nan_color: Color = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Map a single-band array to a four-band uint8 RGBA image."""
    band = _single_band(arr)
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
    arr: np.ndarray,
    mapping: Mapping[int, Color],
    *,
    default: Color = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Map integer classes to a four-band uint8 RGBA image."""
    band = _single_band(arr)
    rgba = np.zeros((*band.shape, 4), dtype=np.float32)
    rgba[...] = np.asarray(default, dtype=np.float32)
    for value, color in mapping.items():
        rgba[band == value] = np.asarray(color, dtype=np.float32)
    return _float_rgba_to_uint8(rgba)


def hillshade(
    dem: np.ndarray,
    *,
    x_resolution: float = 1.0,
    y_resolution: float = 1.0,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Compute GDAL-style hillshade as a single ``uint8`` band."""
    band = _single_band(dem).astype(np.float64, copy=False)
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
    background: np.ndarray,
    foreground: np.ndarray,
    *,
    alpha: float = 0.6,
    mode: str = "alpha",
) -> np.ndarray:
    """Blend two display images and return uint8 RGBA."""
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
    out_alpha = np.maximum(bg_f[3:4], fg_alpha)
    return np.clip(np.concatenate([rgb, out_alpha], axis=0) * 255.0, 0.0, 255.0).astype(
        np.uint8
    )


def ensure_rgba(arr: np.ndarray) -> np.ndarray:
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


def _single_band(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr)
    if values.ndim == 2:
        return values
    if values.ndim == 3 and values.shape[0] == 1:
        return values[0]
    raise ValueError(f"expected a single-band array; got shape {values.shape}")


def _float_rgba_to_uint8(rgba: np.ndarray) -> np.ndarray:
    return np.clip(np.moveaxis(rgba, -1, 0) * 255.0, 0.0, 255.0).astype(np.uint8)
