"""Tier-A primitives for band-space spectral operations."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage, signal

# Re-use the canonical normalized-difference primitive instead of
# duplicating the maths here. Same algebra as NDVI / NDWI / NDBI / NBR.
from geotoolz.indices._src.array import normalized_difference as normalized_difference


def select_bands(arr: np.ndarray, indexes: list[int], *, axis: int = 0) -> np.ndarray:
    """Select bands by integer index along the configured band axis."""
    return np.take(arr, indexes, axis=axis)


def reorder_bands(arr: np.ndarray, order: list[int], *, axis: int = 0) -> np.ndarray:
    """Reorder bands by integer index along the configured band axis."""
    return select_bands(arr, order, axis=axis)


def band_ratio(
    arr: np.ndarray,
    numerator_idx: int,
    denominator_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute ``numerator / (denominator + eps)`` with the band axis collapsed."""
    numerator = np.take(arr, numerator_idx, axis=axis)
    denominator = np.take(arr, denominator_idx, axis=axis)
    return numerator / (denominator + eps)


_ALLOWED_FUNCS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "where": np.where,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
}


def evaluate_band_math(
    expression: str, variables: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Evaluate a restricted arithmetic expression over named band arrays."""
    tree = ast.parse(expression, mode="eval")
    return np.asarray(_eval_node(tree.body, variables))


def _eval_node(node: ast.AST, variables: Mapping[str, np.ndarray]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("BandMath constants must be numeric")
    if isinstance(node, ast.Name):
        try:
            return variables[node.id]
        except KeyError as exc:
            raise ValueError(f"Unknown band name in expression: {node.id!r}") from exc
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        try:
            func = _ALLOWED_FUNCS[node.func.id]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported function in BandMath: {node.func.id!r}"
            ) from exc
        args = [_eval_node(arg, variables) for arg in node.args]
        kwargs = {
            kw.arg: _eval_node(kw.value, variables)
            for kw in node.keywords
            if kw.arg is not None
        }
        return func(*args, **kwargs)
    raise ValueError(f"Unsupported BandMath expression element: {type(node).__name__}")


def _validate_strictly_increasing(wavelengths: np.ndarray, *, context: str) -> None:
    non_increasing = np.flatnonzero(np.diff(wavelengths) <= 0)
    if non_increasing.size:
        idx = int(non_increasing[0])
        raise ValueError(
            f"{context} wavelengths must be strictly increasing; "
            f"found non-increasing pair at indices {idx} and {idx + 1}: "
            f"{wavelengths[idx]} >= {wavelengths[idx + 1]}"
        )


def continuum_removal(
    arr: np.ndarray,
    wavelengths: np.ndarray,
    *,
    axis: int = 0,
    method: str = "convex_hull",
) -> np.ndarray:
    """Apply hull-quotient continuum removal along the band axis."""
    arr_axis0 = np.moveaxis(np.asarray(arr, dtype=float), axis, 0)
    if arr_axis0.shape[0] != wavelengths.size:
        raise ValueError("wavelengths length must match the band axis")
    _validate_strictly_increasing(wavelengths, context="continuum_removal")

    if method == "linear":
        continuum = _linear_continuum(arr_axis0, wavelengths)
    elif method == "convex_hull":
        continuum = _convex_hull_continuum(arr_axis0, wavelengths)
    else:
        raise ValueError("method must be 'convex_hull' or 'linear'")

    out = np.divide(
        arr_axis0, continuum, out=np.ones_like(arr_axis0), where=continuum != 0
    )
    return np.moveaxis(out, 0, axis)


def _linear_continuum(arr_axis0: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    x0 = wavelengths[0]
    x1 = wavelengths[-1]
    y0 = arr_axis0[0]
    y1 = arr_axis0[-1]
    if x1 == x0:
        return np.broadcast_to(y0, arr_axis0.shape)
    weights = ((wavelengths - x0) / (x1 - x0)).reshape(
        (-1,) + (1,) * (arr_axis0.ndim - 1)
    )
    return y0 + weights * (y1 - y0)


def _convex_hull_continuum(
    arr_axis0: np.ndarray, wavelengths: np.ndarray
) -> np.ndarray:
    flat = arr_axis0.reshape(arr_axis0.shape[0], -1)
    continuum = np.empty_like(flat, dtype=float)
    for idx in range(flat.shape[1]):
        continuum[:, idx] = _upper_hull_line(wavelengths, flat[:, idx])
    return continuum.reshape(arr_axis0.shape)


def _upper_hull_line(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    hull: list[int] = []
    for idx in range(x.size):
        hull.append(idx)
        while len(hull) >= 3:
            i, j, k = hull[-3:]
            slope_ij = (y[j] - y[i]) / (x[j] - x[i])
            slope_jk = (y[k] - y[j]) / (x[k] - x[j])
            if slope_ij <= slope_jk:
                hull.pop(-2)
            else:
                break
    return np.interp(x, x[hull], y[hull])


def spectral_binning(
    arr: np.ndarray,
    source_wavelengths: np.ndarray,
    target_wavelengths: np.ndarray,
    width: float | np.ndarray,
    *,
    axis: int = 0,
    method: str = "mean",
) -> np.ndarray:
    """Aggregate source bands into wavelength-centered bins."""
    arr_axis0 = np.moveaxis(np.asarray(arr), axis, 0)
    _validate_strictly_increasing(source_wavelengths, context="spectral_binning")
    widths = np.broadcast_to(np.asarray(width, dtype=float), target_wavelengths.shape)
    out = np.empty((target_wavelengths.size, *arr_axis0.shape[1:]), dtype=float)

    for idx, (center, bin_width) in enumerate(
        zip(target_wavelengths, widths, strict=True)
    ):
        half_width = bin_width / 2.0
        mask = np.abs(source_wavelengths - center) <= half_width
        if not np.any(mask):
            raise ValueError(
                f"No source wavelengths found within +/-{half_width} "
                f"of target bin center {center}; source range: "
                f"[{source_wavelengths.min()}, {source_wavelengths.max()}]"
            )
        values = arr_axis0[mask]
        if method == "mean":
            out[idx] = np.mean(values, axis=0)
        elif method == "median":
            out[idx] = np.median(values, axis=0)
        elif method == "weighted_mean":
            sigma = bin_width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            weights = np.exp(
                -((source_wavelengths[mask] - center) ** 2) / (2.0 * sigma**2)
            )
            weights = weights / weights.sum()
            out[idx] = np.sum(
                weights.reshape((-1,) + (1,) * (values.ndim - 1)) * values, axis=0
            )
        else:
            raise ValueError("method must be 'mean', 'median', or 'weighted_mean'")
    return np.moveaxis(out, 0, axis)


def spectral_smoothing(
    arr: np.ndarray,
    *,
    axis: int = 0,
    method: str = "savgol",
    window: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Smooth spectra along the band axis."""
    if method == "savgol":
        if window % 2 == 0:
            raise ValueError(f"savgol window must be odd, got {window}")
        return signal.savgol_filter(
            arr, window_length=window, polyorder=polyorder, axis=axis
        )
    if method == "gaussian":
        return ndimage.gaussian_filter1d(arr, sigma=window / 2.0, axis=axis)
    if method == "moving_average":
        kernel = np.ones(window, dtype=float) / float(window)
        return np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"), axis, arr
        )
    raise ValueError("method must be 'savgol', 'gaussian', or 'moving_average'")
