"""Tier-A primitives for image restoration."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


_EPSILON = 1e-12
_IDW_POWER_THRESHOLD = 64
_MAD_TO_STD_SCALE = 1.4826


def _nanmean_filter(arr: np.ndarray, size: int | tuple[int, ...]) -> np.ndarray:
    values = np.asarray(arr, dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    count = ndimage.uniform_filter(valid.astype(float), size=size, mode="nearest")
    total = ndimage.uniform_filter(filled, size=size, mode="nearest")
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def _spatial_size(arr: np.ndarray, size: int) -> tuple[int, ...]:
    if size <= 0:
        raise ValueError("window/size must be positive")
    return (1,) * max(arr.ndim - 2, 0) + (int(size), int(size))


def _preserve_nan(original: np.ndarray, restored: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(original), np.nan, restored)


def despeckle_lee(arr: np.ndarray, *, window: int = 7, cu: float = 0.523) -> np.ndarray:
    """Apply the classical Lee local-statistics speckle filter."""
    values = np.asarray(arr, dtype=float)
    size = _spatial_size(values, window)
    mean = _nanmean_filter(values, size)
    mean_sq = _nanmean_filter(values * values, size)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    noise_var = (cu * mean) ** 2
    weight = 0.5 * np.divide(
        var, var + noise_var, out=np.zeros_like(var), where=var > 0
    )
    return _preserve_nan(values, mean + weight * (values - mean))


def despeckle_frost(
    arr: np.ndarray, *, window: int = 7, damping: float = 2.0
) -> np.ndarray:
    """Apply a compact Frost-style adaptive speckle smoother."""
    values = np.asarray(arr, dtype=float)
    size = _spatial_size(values, window)
    mean = _nanmean_filter(values, size)
    mean_sq = _nanmean_filter(values * values, size)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    coeff = np.divide(
        np.sqrt(var), np.abs(mean), out=np.zeros_like(var), where=mean != 0
    )
    alpha = np.exp(-float(damping) * coeff)
    return _preserve_nan(values, alpha * values + (1.0 - alpha) * mean)


def despeckle_refined_lee(arr: np.ndarray, *, window: int = 7) -> np.ndarray:
    """Apply a small refined-Lee approximation using the Lee core."""
    return despeckle_lee(arr, window=window, cu=0.523)


def destripe_column(
    arr: np.ndarray,
    *,
    method: Literal["mean", "median", "moment_matching"] = "mean",
    axis: Literal["column", "row"] = "column",
    window: int = 21,
) -> np.ndarray:
    """Remove row or column striping by matching cross-track statistics."""
    values = np.asarray(arr, dtype=float)
    if values.ndim < 2:
        raise ValueError("destripe_column expects at least two spatial dimensions")
    spatial_axis = -1 if axis == "column" else -2
    reduce_axis = -2 if axis == "column" else -1
    reducer = np.nanmedian if method == "median" else np.nanmean
    profile = reducer(values, axis=reduce_axis, keepdims=True)
    target = reducer(profile, axis=spatial_axis, keepdims=True)
    out = values - (profile - target)
    if method == "moment_matching":
        local = _nanmean_filter(out, _spatial_size(out, window))
        out = out - (_nanmean_filter(out, (1,) * out.ndim) - local)
    return _preserve_nan(values, out)


def gaussian_denoise(arr: np.ndarray, *, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smooth over the trailing spatial axes."""
    values = np.asarray(arr, dtype=float)
    sigma_tuple = (0.0,) * max(values.ndim - 2, 0) + (float(sigma), float(sigma))
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    weights = ndimage.gaussian_filter(valid.astype(float), sigma_tuple, mode="nearest")
    smooth = ndimage.gaussian_filter(filled, sigma_tuple, mode="nearest")
    out = np.divide(
        smooth, weights, out=np.full_like(smooth, np.nan), where=weights > 0
    )
    return _preserve_nan(values, out)


def median_denoise(arr: np.ndarray, *, size: int = 3) -> np.ndarray:
    """Median smooth over the trailing spatial axes."""
    values = np.asarray(arr, dtype=float)
    filled = np.where(np.isfinite(values), values, np.nanmedian(values))
    out = ndimage.median_filter(
        filled, size=_spatial_size(values, size), mode="nearest"
    )
    return _preserve_nan(values, out)


def bilateral_denoise(
    arr: np.ndarray, *, sigma_color: float = 0.1, sigma_space: float = 5.0
) -> np.ndarray:
    """Edge-aware denoise using a range-weighted Gaussian approximation."""
    values = np.asarray(arr, dtype=float)
    smooth = gaussian_denoise(values, sigma=sigma_space)
    weights = np.exp(-0.5 * ((values - smooth) / sigma_color) ** 2)
    return _preserve_nan(values, weights * values + (1.0 - weights) * smooth)


def nl_means(
    arr: np.ndarray, *, patch_size: int = 5, patch_distance: int = 6, h: float = 0.1
) -> np.ndarray:
    """Small non-local-means approximation for dependency-light denoising."""
    sigma = max((float(patch_distance) + float(patch_size)) / 6.0, 0.1)
    smooth = gaussian_denoise(arr, sigma=sigma)
    values = np.asarray(arr, dtype=float)
    weights = np.exp(-0.5 * ((values - smooth) / max(float(h), 1e-12)) ** 2)
    return _preserve_nan(values, weights * values + (1.0 - weights) * smooth)


def pca_denoise(arr: np.ndarray, *, n_components: int, axis: int = 0) -> np.ndarray:
    """Reconstruct an array from its top PCA components along ``axis``."""
    model = fit_pca(arr, n_components=n_components, axis=axis)
    return inverse_pca(model["scores"], model)


def fit_pca(
    arr: np.ndarray, *, n_components: int | None = None, axis: int = 0
) -> dict[str, np.ndarray | int | tuple[int, ...]]:
    """Fit PCA over a band axis and return scores plus reconstruction state."""
    values = np.asarray(arr, dtype=float)
    moved = np.moveaxis(values, axis, 0)
    bands = moved.shape[0]
    keep = bands if n_components is None else int(n_components)
    if not 1 <= keep <= bands:
        raise ValueError("n_components must be between 1 and the number of bands")
    flat = moved.reshape(bands, -1)
    nan_mask = ~np.isfinite(flat)
    means = np.nanmean(flat, axis=1, keepdims=True)
    filled = np.where(nan_mask, means, flat)
    centered = filled - means
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    components = u[:, :keep]
    scores = components.T @ centered
    return {
        "scores": scores.reshape((keep, *moved.shape[1:])),
        "components": components,
        "mean": means[:, 0],
        "axis": axis,
        "shape": values.shape,
        "nan_mask": nan_mask,
        "snr": s[:keep] ** 2,
    }


def inverse_pca(
    scores: np.ndarray, state: dict[str, np.ndarray | int | tuple[int, ...]]
) -> np.ndarray:
    """Reconstruct an array from PCA scores and state."""
    components = np.asarray(state["components"])
    mean = np.asarray(state["mean"])[:, None]
    shape = cast(tuple[int, ...], state["shape"])
    axis = int(state["axis"])
    flat_scores = np.asarray(scores, dtype=float).reshape(components.shape[1], -1)
    restored = components @ flat_scores + mean
    nan_mask = np.asarray(state["nan_mask"])
    restored = np.where(nan_mask, np.nan, restored)
    moved_shape = np.moveaxis(np.empty(shape), axis, 0).shape
    return np.moveaxis(restored.reshape(moved_shape), 0, axis)


def gap_fill_nearest(arr: np.ndarray, *, max_distance: int | None = None) -> np.ndarray:
    """Fill NaNs from the nearest finite neighbour."""
    values = np.asarray(arr, dtype=float)
    out = values.copy()
    for idx in np.ndindex(values.shape[:-2] or ()):
        plane = values[idx] if values.ndim > 2 else values
        mask = ~np.isfinite(plane)
        if not mask.any() or mask.all():
            continue
        distances, nearest = ndimage.distance_transform_edt(
            mask, return_distances=True, return_indices=True
        )
        filled = plane[tuple(nearest)]
        if max_distance is not None:
            filled = np.where(distances <= max_distance, filled, np.nan)
        if values.ndim > 2:
            out[idx] = np.where(mask, filled, plane)
        else:
            out = np.where(mask, filled, plane)
    return out


def gap_fill_idw(arr: np.ndarray, *, power: float = 2.0, radius: int = 5) -> np.ndarray:
    """Fill NaNs with inverse-distance weighted finite neighbours.

    Power values at or above ``_IDW_POWER_THRESHOLD`` automatically fall
    back to nearest-neighbour filling because very large powers converge
    to nearest-neighbour weights while risking overflow.
    """
    # Very large IDW powers converge to nearest-neighbour but risk overflow,
    # so delegate to the nearest-neighbour implementation.
    if power >= _IDW_POWER_THRESHOLD:
        return gap_fill_nearest(arr, max_distance=radius)
    values = np.asarray(arr, dtype=float)
    out = values.copy()
    for idx in np.ndindex(values.shape[:-2] or ()):
        plane = values[idx] if values.ndim > 2 else values
        missing = ~np.isfinite(plane)
        if not missing.any() or missing.all():
            continue
        valid = np.argwhere(np.isfinite(plane))
        tree = cKDTree(valid)
        for row, col in np.argwhere(missing):
            neighbours = tree.query_ball_point([row, col], r=radius)
            if not neighbours:
                continue
            coords = valid[neighbours]
            dist = np.linalg.norm(coords - np.array([row, col]), axis=1)
            weights = 1.0 / np.maximum(dist, _EPSILON) ** power
            out_value = np.sum(weights * plane[coords[:, 0], coords[:, 1]]) / np.sum(
                weights
            )
            if values.ndim > 2:
                out[*idx, row, col] = out_value
            else:
                out[row, col] = out_value
    return out


def gap_fill_laplacian(arr: np.ndarray, *, iterations: int = 200) -> np.ndarray:
    """Fill NaNs by iteratively solving a discrete Laplace equation."""
    values = np.asarray(arr, dtype=float)
    out = gap_fill_nearest(values)
    missing = ~np.isfinite(values)
    for _ in range(iterations):
        avg = (
            np.roll(out, 1, axis=-2)
            + np.roll(out, -1, axis=-2)
            + np.roll(out, 1, axis=-1)
            + np.roll(out, -1, axis=-1)
        ) / 4.0
        out = np.where(missing, avg, values)
    return out


def gap_fill_biharmonic(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs with a smooth biharmonic-style two-pass Laplacian fill."""
    values = np.asarray(arr, dtype=float)
    smooth = gaussian_denoise(gap_fill_laplacian(values), sigma=1.0)
    return np.where(np.isfinite(values), values, smooth)


def outlier_mask(
    arr: np.ndarray, *, method: Literal["mad", "zscore"] = "mad", k: float = 3.0
) -> np.ndarray:
    """Flag robust global outliers."""
    values = np.asarray(arr, dtype=float)
    if method == "mad":
        center = np.nanmedian(values)
        # _MAD_TO_STD_SCALE converts MAD to std: 1 / inverse_normal_cdf(0.75).
        scale = _MAD_TO_STD_SCALE * np.nanmedian(np.abs(values - center))
    elif method == "zscore":
        center = np.nanmean(values)
        scale = np.nanstd(values)
    else:
        raise ValueError("method must be 'mad' or 'zscore'")
    if scale == 0:
        return np.isfinite(values) & (values != center)
    if not np.isfinite(scale):
        return np.zeros(values.shape, dtype=bool)
    return np.abs(values - center) > k * scale


def replace_outliers(
    arr: np.ndarray,
    *,
    method: Literal["mad", "zscore"] = "mad",
    k: float = 3.0,
    fill: Literal["median", "nan", "interp"] = "median",
) -> np.ndarray:
    """Replace detected outliers with a scalar or nearest-neighbour fill."""
    values = np.asarray(arr, dtype=float)
    mask = outlier_mask(values, method=method, k=k)
    if fill == "median":
        return np.where(mask, np.nanmedian(values[~mask]), values)
    if fill == "nan":
        return np.where(mask, np.nan, values)
    if fill == "interp":
        return gap_fill_nearest(np.where(mask, np.nan, values))
    raise ValueError("fill must be 'median', 'nan', or 'interp'")


def saturation_flag(arr: np.ndarray, *, threshold: float | None = None) -> np.ndarray:
    """Flag pixels at or above a saturation threshold."""
    values = np.asarray(arr)
    if threshold is None:
        threshold = (
            float(np.iinfo(values.dtype).max)
            if np.issubdtype(values.dtype, np.integer)
            else 1.0
        )
    return values >= threshold
