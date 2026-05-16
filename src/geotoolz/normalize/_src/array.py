"""Pure-numpy normalization primitives."""

from __future__ import annotations

import numpy as np


def stat_axes(arr: np.ndarray, *, per_band: bool = True) -> tuple[int, ...] | None:
    """Return spatial reduction axes for per-band remote-sensing arrays."""
    if per_band and arr.ndim >= 3:
        return (-2, -1)
    return None


def reshape_stat(
    stat: np.ndarray | float, arr: np.ndarray, axis: tuple[int, ...] | None
) -> np.ndarray:
    """Reshape a reduced statistic so it broadcasts over ``arr``."""
    stat_arr = np.asarray(stat, dtype=float)
    if stat_arr.ndim == 0 or axis is None:
        return stat_arr

    axes = tuple(a % arr.ndim for a in axis)
    kept_axes = tuple(i for i in range(arr.ndim) if i not in axes)
    kept_shape = tuple(arr.shape[i] for i in kept_axes)
    if stat_arr.shape != kept_shape:
        return stat_arr

    shape = [1] * arr.ndim
    for stat_axis, arr_axis in enumerate(kept_axes):
        shape[arr_axis] = stat_arr.shape[stat_axis]
    return stat_arr.reshape(shape)


def per_band_stats(
    arr: np.ndarray,
    *,
    percentiles: list[float] | tuple[float, ...] = (1.0, 99.0),
    axis: tuple[int, ...] | None = (-2, -1),
) -> dict[str, np.ndarray]:
    """Compute NaN-aware statistics over spatial axes."""
    return {
        "mean": np.nanmean(arr, axis=axis),
        "std": np.nanstd(arr, axis=axis),
        "min": np.nanmin(arr, axis=axis),
        "max": np.nanmax(arr, axis=axis),
        "percentiles": np.nanpercentile(arr, percentiles, axis=axis),
    }


def standard_scale(
    arr: np.ndarray,
    mean: np.ndarray | float,
    std: np.ndarray | float,
    *,
    axis: tuple[int, ...] | None = (-2, -1),
) -> np.ndarray:
    """Apply z-score scaling while preserving NaN pixels."""
    mean_b = reshape_stat(mean, arr, axis)
    std_b = reshape_stat(std, arr, axis)
    denom = np.where(std_b != 0, std_b, 1.0)
    return (arr - mean_b) / denom


def robust_scale(
    arr: np.ndarray,
    median: np.ndarray | float,
    iqr: np.ndarray | float,
    *,
    axis: tuple[int, ...] | None = (-2, -1),
) -> np.ndarray:
    """Apply median/IQR scaling while preserving NaN pixels."""
    median_b = reshape_stat(median, arr, axis)
    iqr_b = reshape_stat(iqr, arr, axis)
    denom = np.where(iqr_b != 0, iqr_b, 1.0)
    return (arr - median_b) / denom


def minmax_scale(
    arr: np.ndarray,
    vmin: np.ndarray | float,
    vmax: np.ndarray | float,
    *,
    out_range: tuple[float, float] = (0.0, 1.0),
    axis: tuple[int, ...] | None = (-2, -1),
) -> np.ndarray:
    """Linearly map ``[vmin, vmax]`` into ``out_range``."""
    out_min, out_max = out_range
    if out_max <= out_min:
        raise ValueError(f"out_range must be increasing; got {out_range}")
    vmin_b = reshape_stat(vmin, arr, axis)
    vmax_b = reshape_stat(vmax, arr, axis)
    denom = np.where(vmax_b > vmin_b, vmax_b - vmin_b, 1.0)
    return (arr - vmin_b) / denom * (out_max - out_min) + out_min


def percentile_clip(
    arr: np.ndarray,
    *,
    lower: float = 1.0,
    upper: float = 99.0,
    axis: tuple[int, ...] | None = (-2, -1),
) -> np.ndarray:
    """Clip to percentile bounds and stretch the result into ``[0, 1]``."""
    if upper <= lower:
        raise ValueError(f"upper must be greater than lower; got {lower=}, {upper=}")
    lo = np.nanpercentile(arr, lower, axis=axis, keepdims=True)
    hi = np.nanpercentile(arr, upper, axis=axis, keepdims=True)
    denom = np.where(hi > lo, hi - lo, 1.0)
    return np.clip((arr - lo) / denom, 0.0, 1.0)


def histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match source values to the empirical CDF of ``reference``."""
    out = np.array(source, dtype=float, copy=True)
    if (
        source.ndim >= 3
        and reference.ndim >= 3
        and source.shape[0] == reference.shape[0]
    ):
        for band in range(source.shape[0]):
            out[band] = _match_slice(source[band], reference[band])
        return out
    if source.ndim >= 3 and reference.ndim == 2:
        for band in range(source.shape[0]):
            out[band] = _match_slice(source[band], reference)
        return out
    return _match_slice(source, reference)


def _match_slice(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    out = np.array(source, dtype=float, copy=True)
    valid = np.isfinite(source)
    source_valid = source[valid]
    reference_valid = reference[np.isfinite(reference)]
    if source_valid.size == 0 or reference_valid.size == 0:
        return out

    _src_values, bin_idx, src_counts = np.unique(
        source_valid, return_inverse=True, return_counts=True
    )
    ref_values, ref_counts = np.unique(reference_valid, return_counts=True)
    src_quantiles = np.cumsum(src_counts).astype(float) / source_valid.size
    ref_quantiles = np.cumsum(ref_counts).astype(float) / reference_valid.size
    interp_values = np.interp(src_quantiles, ref_quantiles, ref_values)
    out[valid] = interp_values[bin_idx]
    return out


def log_scale(arr: np.ndarray, *, base: float = 10.0, eps: float = 1e-6) -> np.ndarray:
    """Apply log scaling with a small offset for zero-valued pixels."""
    if base <= 0 or base == 1.0:
        raise ValueError(f"base must be positive and not equal to 1; got {base}")
    if eps <= 0:
        raise ValueError(f"eps must be positive; got {eps}")
    return np.log(np.maximum(arr, 0.0) + eps) / np.log(base)


def asinh_scale(arr: np.ndarray, *, a: float = 1.0) -> np.ndarray:
    """Apply inverse-hyperbolic-sine scaling."""
    if a <= 0:
        raise ValueError(f"a must be positive; got {a}")
    return np.arcsinh(arr / a)


def power_scale(arr: np.ndarray, *, gamma: float = 0.5) -> np.ndarray:
    """Apply non-negative power scaling."""
    if gamma <= 0:
        raise ValueError(f"gamma must be positive; got {gamma}")
    return np.maximum(arr, 0.0) ** gamma
