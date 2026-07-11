"""Tier-A primitives for image restoration.

All functions in this module are pure NumPy / SciPy. They take a plain
``ndarray`` (shape ``(..., H, W)`` for spatial filters, ``(bands, H, W)``
for spectral PCA / MNF) and return a plain ``ndarray`` of the same
shape. Carrier-aware ``Operator`` wrappers live in
:mod:`geotoolz.restore._src.operators`.

NaN convention: input NaNs are treated as missing pixels. Filters that
*restore* (denoise, despeckle, destripe) propagate NaNs through to the
output unchanged. Filters that *fill* (``gap_fill_*``) replace NaNs
with a finite estimate.
"""

from __future__ import annotations

from typing import Literal, cast

import einx
import numpy as np
from jaxtyping import Bool, Float, Num, Shaped
from scipy import ndimage
from scipy.spatial import cKDTree


_EPSILON = 1e-12
# Powers above this threshold converge numerically to nearest-neighbour
# weights but risk float overflow; the IDW path short-circuits to the
# dedicated nearest-neighbour implementation.
_IDW_POWER_THRESHOLD = 64
# Consistency constant 1 / Phi^{-1}(0.75) that scales MAD to a Gaussian
# standard-deviation estimator.
_MAD_TO_STD_SCALE = 1.4826


def _nanmean_filter(
    arr: Num[np.ndarray, "*dims"], size: int | tuple[int, ...]
) -> Float[np.ndarray, "*dims"]:
    """Uniform window mean that ignores NaN entries.

    Computes ``sum(finite) / count(finite)`` over a uniform window of
    ``size``. Windows that contain no finite values yield ``NaN``.
    """
    values = np.asarray(arr, dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    count = ndimage.uniform_filter(valid.astype(float), size=size, mode="nearest")
    total = ndimage.uniform_filter(filled, size=size, mode="nearest")
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def _spatial_size(arr: Shaped[np.ndarray, "*dims"], size: int) -> tuple[int, ...]:
    """Build a per-axis window tuple that filters only the last two axes."""
    if size <= 0:
        raise ValueError("window/size must be positive")
    return (1,) * max(arr.ndim - 2, 0) + (int(size), int(size))


def _preserve_nan(
    original: Float[np.ndarray, "*dims"], restored: Float[np.ndarray, "*dims"]
) -> Float[np.ndarray, "*dims"]:
    """Re-stamp NaN positions from ``original`` onto ``restored``."""
    return np.where(np.isnan(original), np.nan, restored)


def despeckle_lee(
    arr: Num[np.ndarray, "*batch h w"], *, window: int = 7, cu: float = 0.523
) -> Float[np.ndarray, "*batch h w"]:
    r"""Apply the classical Lee local-statistics speckle filter.

    For each pixel computes a local mean :math:`\bar{x}` and variance
    :math:`s^2` over a ``window`` x ``window`` neighbourhood, then
    returns :math:`\bar{x} + k (x - \bar{x})` with the adaptive gain
    :math:`k = \tfrac{1}{2} s^2 / (s^2 + c_u^2 \bar{x}^2)`. In flat
    regions the filter collapses to the local mean; near edges the gain
    approaches ``1`` and the pixel passes through.

    Args:
        arr: Array of shape ``(..., H, W)``. NaNs are preserved.
        window: Side length of the local window. Must be positive.
        cu: Noise coefficient of variation. ``0.523`` is the standard
            value for a single-look SAR image; halve for multi-look.

    Returns:
        Smoothed array with the same shape and NaN positions as ``arr``.
    """
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
    arr: Num[np.ndarray, "*batch h w"], *, window: int = 7, damping: float = 2.0
) -> Float[np.ndarray, "*batch h w"]:
    r"""Apply a compact Frost-style adaptive speckle smoother.

    Blends each pixel with its local mean using an edge-aware weight
    :math:`\alpha = \exp(-d \cdot c_v)`, where ``d`` is ``damping`` and
    :math:`c_v = s / |\bar{x}|` is the local coefficient of variation.
    Flat regions (small :math:`c_v`) push :math:`\alpha \to 1` and keep
    the pixel; high-variance regions (edges) push :math:`\alpha \to 0`
    and smooth.

    This is a single-pixel reduction of the full Frost kernel — fast
    and dependency-light, but missing the directional weighting of the
    canonical filter. Use ``despeckle_lee`` if you want strict
    statistical optimality.

    Args:
        arr: Array of shape ``(..., H, W)``. NaNs are preserved.
        window: Side length of the local window. Must be positive.
        damping: Edge-sensitivity exponent ``d``. Larger values keep
            more edge contrast; smaller values smooth more.

    Returns:
        Smoothed array with the same shape and NaN positions as ``arr``.
    """
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


def despeckle_refined_lee(
    arr: Num[np.ndarray, "*batch h w"], *, window: int = 7
) -> Float[np.ndarray, "*batch h w"]:
    """Apply a Refined-Lee approximation.

    The full Refined-Lee filter chooses one of eight directional
    sub-windows per pixel before applying the Lee gain. This
    dependency-light implementation skips the directional selection and
    just calls :func:`despeckle_lee` with the default ``cu``. It is
    kept under a distinct name so pipelines can swap in a true Refined-
    Lee later without renaming nodes.

    Args:
        arr: Array of shape ``(..., H, W)``. NaNs are preserved.
        window: Side length of the local window. Must be positive.

    Returns:
        Smoothed array with the same shape and NaN positions as ``arr``.
    """
    return despeckle_lee(arr, window=window, cu=0.523)


def destripe_column(
    arr: Num[np.ndarray, "*batch h w"],
    *,
    method: Literal["mean", "median", "moment_matching"] = "mean",
    axis: Literal["column", "row"] = "column",
    window: int = 21,
) -> Float[np.ndarray, "*batch h w"]:
    r"""Remove row or column striping by matching cross-track statistics.

    For ``axis="column"`` (default): collapses each column to a single
    statistic (its mean or median over rows), subtracts the difference
    from the global statistic, and re-adds the per-pixel residual so
    column-constant offsets are zeroed out.

    ``method="moment_matching"`` follows up the global recentre with a
    locally-windowed smoothing pass to handle slowly varying gain
    drift; the ``window`` controls the smoothing kernel size and is
    only consulted in that case.

    Args:
        arr: Array of shape ``(..., H, W)``. NaNs are preserved.
        method: ``"mean"`` and ``"median"`` subtract a per-column offset
            with the corresponding reducer; ``"moment_matching"``
            additionally smooths the result with a ``window`` x
            ``window`` neighbourhood.
        axis: Striping direction (``"column"`` for vertical stripes,
            ``"row"`` for horizontal).
        window: Side length of the smoothing window for
            ``method="moment_matching"``. Ignored otherwise.

    Returns:
        Destriped array with the same shape and NaN positions as
        ``arr``.

    Raises:
        ValueError: If ``arr`` has fewer than two dimensions or
            ``method`` is not one of the documented choices.
    """
    values = np.asarray(arr, dtype=float)
    if values.ndim < 2:
        raise ValueError("destripe_column expects at least two spatial dimensions")
    if method not in {"mean", "median", "moment_matching"}:
        raise ValueError("method must be 'mean', 'median', or 'moment_matching'")
    spatial_axis = -1 if axis == "column" else -2
    reduce_axis = -2 if axis == "column" else -1
    reducer = np.nanmedian if method == "median" else np.nanmean
    profile = reducer(values, axis=reduce_axis, keepdims=True)
    target = reducer(profile, axis=spatial_axis, keepdims=True)
    out = values - (profile - target)
    if method == "moment_matching":
        # Replace each pixel with its local-window mean so slowly
        # varying per-column gain drift is absorbed into the smoothing.
        out = _nanmean_filter(out, _spatial_size(out, window))
    return _preserve_nan(values, out)


def gaussian_denoise(
    arr: Num[np.ndarray, "*batch h w"], *, sigma: float = 1.0
) -> Float[np.ndarray, "*batch h w"]:
    """Gaussian smooth over the trailing two (spatial) axes.

    NaN-aware: missing pixels are excluded from both numerator and
    weight (Nadaraya-Watson style normalisation), then re-stamped onto
    the output. Non-spatial axes (e.g. bands) are filtered
    independently with ``sigma=0``.

    Args:
        arr: Array of shape ``(..., H, W)``.
        sigma: Gaussian standard deviation in pixels. ``0`` is a no-op.

    Returns:
        Smoothed array with the same shape and NaN positions as ``arr``.
    """
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


def median_denoise(
    arr: Num[np.ndarray, "*batch h w"], *, size: int = 3
) -> Float[np.ndarray, "*batch h w"]:
    """Median smooth over the trailing two (spatial) axes.

    NaNs are temporarily replaced with the global ``nanmedian`` so
    SciPy's median filter is well-defined, then re-stamped on the
    output. Edge pixels use ``mode="nearest"`` (reflected boundary).

    Args:
        arr: Array of shape ``(..., H, W)``.
        size: Side length of the median window. Must be positive.

    Returns:
        Smoothed array with the same shape and NaN positions as ``arr``.
    """
    values = np.asarray(arr, dtype=float)
    filled = np.where(np.isfinite(values), values, np.nanmedian(values))
    out = ndimage.median_filter(
        filled, size=_spatial_size(values, size), mode="nearest"
    )
    return _preserve_nan(values, out)


def bilateral_denoise(
    arr: Num[np.ndarray, "*batch h w"],
    *,
    sigma_color: float = 0.1,
    sigma_space: float = 5.0,
) -> Float[np.ndarray, "*batch h w"]:
    r"""Edge-aware denoise using a range-weighted Gaussian approximation.

    Computes a spatially smoothed estimate ``s`` of shape ``arr`` and
    blends it with the original ``x`` using a per-pixel weight
    :math:`w = \exp(-\tfrac{1}{2} ((x - s) / \sigma_c)^2)`. Where
    ``x`` is far from the local mean (edges, outliers), ``w`` is small
    and the *smoothed* estimate is suppressed in favour of the
    original, preserving edges. Where ``x`` is close to the local mean
    (flat regions), ``w`` is near 1 and the original passes through —
    the smoothing is therefore best thought of as an edge-aware
    *attenuator* of the Gaussian estimate.

    This is a single-pass approximation of a full bilateral filter (no
    per-pixel neighbourhood weighting). It is fast and dependency-
    light; for true bilateral filtering use scikit-image.

    Args:
        arr: Array of shape ``(..., H, W)``. NaNs are preserved.
        sigma_color: Range bandwidth (data units). Smaller values
            preserve more edges; larger values smooth more.
        sigma_space: Spatial bandwidth in pixels.

    Returns:
        Denoised array with the same shape and NaN positions as ``arr``.
    """
    values = np.asarray(arr, dtype=float)
    smooth = gaussian_denoise(values, sigma=sigma_space)
    safe_color = max(float(sigma_color), _EPSILON)
    weights = np.exp(-0.5 * ((values - smooth) / safe_color) ** 2)
    return _preserve_nan(values, weights * values + (1.0 - weights) * smooth)


def nl_means(
    arr: Num[np.ndarray, "*batch h w"],
    *,
    patch_size: int = 5,
    patch_distance: int = 6,
    h: float = 0.1,
) -> Float[np.ndarray, "*batch h w"]:
    """Lightweight non-local-means-style denoiser.

    True non-local-means averages each pixel with similarly-patterned
    patches elsewhere in the image, which is expensive and adds
    scikit-image as a dependency. This implementation is a
    *dependency-light approximation*: it computes a wide Gaussian
    smoothing estimate whose effective radius is set by
    ``(patch_distance + patch_size) / 6``, then blends it with the
    original using the same range-weighted scheme as
    :func:`bilateral_denoise` with bandwidth ``h``.

    Use this only when scikit-image is unavailable; for production
    NL-means, prefer ``skimage.restoration.denoise_nl_means``.

    Args:
        arr: Array of shape ``(..., H, W)``.
        patch_size: Nominal patch side length (pixels).
        patch_distance: Nominal search-window radius (pixels).
        h: Range bandwidth (data units).

    Returns:
        Denoised array with the same shape and NaN positions as ``arr``.
    """
    sigma = max((float(patch_distance) + float(patch_size)) / 6.0, 0.1)
    smooth = gaussian_denoise(arr, sigma=sigma)
    values = np.asarray(arr, dtype=float)
    weights = np.exp(-0.5 * ((values - smooth) / max(float(h), _EPSILON)) ** 2)
    return _preserve_nan(values, weights * values + (1.0 - weights) * smooth)


def pca_denoise(
    arr: Num[np.ndarray, "*dims"], *, n_components: int, axis: int = 0
) -> Float[np.ndarray, "*dims"]:
    """Reconstruct an array from its top PCA components along ``axis``.

    Convenience composition of :func:`fit_pca` and :func:`inverse_pca`.
    Keeping fewer components than bands suppresses band-uncorrelated
    noise while preserving the dominant spectral structure.

    Args:
        arr: Array with a band axis at position ``axis``; typically
            ``(bands, H, W)``. NaN pixels are mean-imputed for the fit
            and re-stamped as NaN on the output.
        n_components: Number of principal components to keep. Must be
            between 1 and the number of bands.
        axis: Position of the band axis. Defaults to ``0``.

    Returns:
        Reconstructed array with the same shape and NaN positions as
        ``arr``.

    Raises:
        ValueError: If ``n_components`` is out of range or any band is
            entirely NaN.
    """
    model = fit_pca(arr, n_components=n_components, axis=axis)
    return inverse_pca(model["scores"], model)


def fit_pca(
    arr: Num[np.ndarray, "*dims"], *, n_components: int | None = None, axis: int = 0
) -> dict[str, np.ndarray | int | tuple[int, ...]]:
    """Fit PCA over a band axis and return scores plus reconstruction state.

    NaN pixels are imputed with the per-band mean for the fit; their
    positions are recorded in the returned state so
    :func:`inverse_pca` can re-stamp them.

    Args:
        arr: Array with a band axis at position ``axis``; typically
            ``(bands, H, W)``.
        n_components: Number of components to keep. ``None`` (default)
            keeps all bands.
        axis: Position of the band axis. Defaults to ``0``.

    Returns:
        State dict with keys ``"scores"`` (projected data, band axis
        replaced by the component axis), ``"components"``, ``"mean"``,
        ``"axis"``, ``"shape"``, ``"nan_mask"``, and ``"snr"``
        (per-component variance, sorted descending). Pass it verbatim
        to :func:`inverse_pca`.

    Raises:
        ValueError: If ``n_components`` is out of range or any band is
            entirely NaN.
    """
    values = np.asarray(arr, dtype=float)
    moved = np.moveaxis(values, axis, 0)
    bands = moved.shape[0]
    keep = bands if n_components is None else int(n_components)
    if not 1 <= keep <= bands:
        raise ValueError("n_components must be between 1 and the number of bands")
    flat = moved.reshape(bands, -1)
    nan_mask = ~np.isfinite(flat)
    all_nan_bands = nan_mask.all(axis=1)
    if all_nan_bands.any():
        bad = tuple(int(i) for i in np.where(all_nan_bands)[0])
        raise ValueError(f"PCA cannot fit on bands that are entirely NaN: bands {bad}")
    means = np.nanmean(flat, axis=1, keepdims=True)
    filled = np.where(nan_mask, means, flat)
    centered = filled - means
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    components = u[:, :keep]
    # Project the centered (bands, pixels) data onto the components.
    scores = einx.dot("c k, c n -> k n", components, centered)
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
    scores: Num[np.ndarray, "*dims"],
    state: dict[str, np.ndarray | int | tuple[int, ...]],
) -> np.ndarray:
    """Reconstruct an array from PCA scores and state.

    Args:
        scores: Projected data as returned in ``state["scores"]`` (the
            leading component axis must match ``state["components"]``).
        state: Reconstruction state produced by :func:`fit_pca`.

    Returns:
        Reconstructed array with the shape recorded in ``state`` and the
        original NaN positions re-stamped.
    """
    components = np.asarray(state["components"])
    mean = np.asarray(state["mean"])[:, None]
    shape = cast(tuple[int, ...], state["shape"])
    axis = int(state["axis"])
    flat_scores = np.asarray(scores, dtype=float).reshape(components.shape[1], -1)
    restored = einx.dot("c k, k n -> c n", components, flat_scores) + mean
    nan_mask = np.asarray(state["nan_mask"])
    restored = np.where(nan_mask, np.nan, restored)
    moved_shape = np.moveaxis(np.empty(shape), axis, 0).shape
    return np.moveaxis(restored.reshape(moved_shape), 0, axis)


def _iter_planes(
    values: Float[np.ndarray, "*batch h w"],
) -> list[tuple[tuple[int, ...] | None, Float[np.ndarray, "h w"]]]:
    """Yield each 2-D ``(H, W)`` plane of ``values`` with its leading index.

    The leading index is ``None`` for a 2-D input so callers can dispatch
    on shape without a separate ``ndim`` check.
    """
    if values.ndim == 2:
        return [(None, values)]
    return [(idx, values[idx]) for idx in np.ndindex(values.shape[:-2])]


def gap_fill_nearest(
    arr: Num[np.ndarray, "*batch h w"], *, max_distance: int | None = None
) -> Float[np.ndarray, "*batch h w"]:
    """Fill NaNs from the nearest finite neighbour.

    Uses :func:`scipy.ndimage.distance_transform_edt` per 2-D plane.
    Pixels whose nearest finite neighbour is farther than
    ``max_distance`` (when given) are left as NaN.

    Args:
        arr: Array of shape ``(..., H, W)``. Planes are processed
            independently.
        max_distance: Optional maximum Euclidean fill radius in pixels.
            ``None`` (default) fills every NaN that has at least one
            finite neighbour anywhere in the plane.

    Returns:
        Filled array with the same shape as ``arr``.
    """
    values = np.asarray(arr, dtype=float)
    out = values.copy()
    for idx, plane in _iter_planes(values):
        mask = ~np.isfinite(plane)
        if not mask.any() or mask.all():
            continue
        distances, nearest = ndimage.distance_transform_edt(
            mask, return_distances=True, return_indices=True
        )
        filled = plane[tuple(nearest)]
        if max_distance is not None:
            filled = np.where(distances <= max_distance, filled, np.nan)
        result = np.where(mask, filled, plane)
        if idx is None:
            out = result
        else:
            out[idx] = result
    return out


def gap_fill_idw(
    arr: Num[np.ndarray, "*batch h w"], *, power: float = 2.0, radius: int = 5
) -> Float[np.ndarray, "*batch h w"]:
    r"""Fill NaNs with inverse-distance weighted finite neighbours.

    For each NaN pixel ``p``, finds all finite neighbours within
    ``radius`` pixels (Euclidean) and replaces it with
    :math:`\hat{p} = \sum_i w_i x_i / \sum_i w_i` where
    :math:`w_i = 1 / \max(d_i, \epsilon)^{\text{power}}`. NaNs whose
    neighbourhood is empty are left as NaN.

    Powers ``>= 64`` numerically converge to nearest-neighbour weights
    but risk overflow, so the function short-circuits to
    :func:`gap_fill_nearest` in that case.

    Args:
        arr: Array of shape ``(..., H, W)``. Planes are processed
            independently with a per-plane k-d tree.
        power: IDW exponent. ``2.0`` is the standard choice; larger
            values bias toward nearer neighbours.
        radius: Maximum search radius in pixels.

    Returns:
        Filled array with the same shape as ``arr``.
    """
    if power >= _IDW_POWER_THRESHOLD:
        return gap_fill_nearest(arr, max_distance=radius)
    values = np.asarray(arr, dtype=float)
    out = values.copy()
    for idx, plane in _iter_planes(values):
        missing = ~np.isfinite(plane)
        if not missing.any() or missing.all():
            continue
        valid = np.argwhere(np.isfinite(plane))
        tree = cKDTree(valid)
        plane_out = plane.copy()
        for row, col in np.argwhere(missing):
            neighbours = tree.query_ball_point([row, col], r=radius)
            if not neighbours:
                continue
            coords = valid[neighbours]
            dist = np.linalg.norm(coords - np.array([row, col]), axis=1)
            weights = 1.0 / np.maximum(dist, _EPSILON) ** power
            plane_out[row, col] = np.sum(
                weights * plane[coords[:, 0], coords[:, 1]]
            ) / np.sum(weights)
        if idx is None:
            out = plane_out
        else:
            out[idx] = plane_out
    return out


def gap_fill_laplacian(
    arr: Num[np.ndarray, "*batch h w"], *, iterations: int = 200
) -> Float[np.ndarray, "*batch h w"]:
    r"""Fill NaNs by iteratively solving a discrete Laplace equation.

    Iterates the 4-neighbour averaging
    :math:`u^{(k+1)}_{i,j} = \tfrac{1}{4}(u^{(k)}_{i-1,j} +
    u^{(k)}_{i+1,j} + u^{(k)}_{i,j-1} + u^{(k)}_{i,j+1})` over the
    missing pixels while clamping the finite pixels to their original
    values. Converges to the harmonic interpolant in the limit. The
    initial guess is :func:`gap_fill_nearest` for fast convergence.

    Args:
        arr: Array of shape ``(..., H, W)``.
        iterations: Number of Jacobi sweeps. Increase for larger gaps.

    Returns:
        Filled array with the same shape as ``arr``; the original
        finite pixels are preserved exactly.
    """
    values = np.asarray(arr, dtype=float)
    out = gap_fill_nearest(values)
    missing = ~np.isfinite(values)
    # Pad with edge-repeat so the 4-neighbour stencil at the raster
    # boundary uses the nearest interior pixel rather than wrapping to
    # the opposite edge (np.roll would impose periodic boundaries).
    pad_width = [(0, 0)] * (out.ndim - 2) + [(1, 1), (1, 1)]
    for _ in range(iterations):
        padded = np.pad(out, pad_width, mode="edge")
        avg = (
            padded[..., :-2, 1:-1]
            + padded[..., 2:, 1:-1]
            + padded[..., 1:-1, :-2]
            + padded[..., 1:-1, 2:]
        ) / 4.0
        out = np.where(missing, avg, values)
    return out


def gap_fill_biharmonic(
    arr: Num[np.ndarray, "*batch h w"],
) -> Float[np.ndarray, "*batch h w"]:
    """Fill NaNs with a smooth biharmonic-style two-pass Laplacian fill.

    Approximates biharmonic inpainting (which solves
    :math:`\\Delta^2 u = 0` on the masked region) with a cheap two-pass
    surrogate: a harmonic fill via :func:`gap_fill_laplacian`, then a
    short Gaussian smooth to relax the gradient discontinuity at the
    mask boundary. The original finite pixels are preserved exactly.

    Note: this is *not* the canonical biharmonic inpainting from
    scikit-image; it is intentionally dependency-light. For sharper
    boundary continuity, use ``skimage.restoration.inpaint_biharmonic``.

    Args:
        arr: Array of shape ``(..., H, W)``.

    Returns:
        Filled array with the same shape as ``arr``; the original
        finite pixels are preserved exactly.
    """
    values = np.asarray(arr, dtype=float)
    smooth = gaussian_denoise(gap_fill_laplacian(values), sigma=1.0)
    return np.where(np.isfinite(values), values, smooth)


def outlier_mask(
    arr: Num[np.ndarray, "*dims"],
    *,
    method: Literal["mad", "zscore"] = "mad",
    k: float = 3.0,
) -> Bool[np.ndarray, "*dims"]:
    r"""Flag robust global outliers.

    ``method="mad"`` (default) uses the median + median-absolute-
    deviation; ``"zscore"`` uses the mean + standard deviation. The MAD
    is scaled by 1 / Phi^{-1}(0.75) so it estimates the same scale as
    the standard deviation under a Gaussian model. Returns a boolean
    array where ``True`` marks outliers.

    When the estimated scale is zero (constant data) the mask flags any
    pixel that differs from the centre, which is consistent with
    "anything that breaks the constant pattern is an outlier". When the
    scale is non-finite (all-NaN input) returns an all-False mask.

    Args:
        arr: Input array of any shape. Statistics are global (whole
            array), not windowed.
        method: ``"mad"`` (robust, default) or ``"zscore"``.
        k: Threshold in scaled units (approximately standard
            deviations).

    Returns:
        Boolean array of the same shape; ``True`` marks outliers.

    Raises:
        ValueError: If ``method`` is not ``"mad"`` or ``"zscore"``.
    """
    values = np.asarray(arr, dtype=float)
    if method == "mad":
        center = np.nanmedian(values)
        # _MAD_TO_STD_SCALE converts MAD to a std estimate; it is the
        # consistency constant 1 / inverse_standard_normal_cdf(0.75).
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
    arr: Num[np.ndarray, "*batch h w"],
    *,
    method: Literal["mad", "zscore"] = "mad",
    k: float = 3.0,
    fill: Literal["median", "nan", "interp"] = "median",
) -> Float[np.ndarray, "*batch h w"]:
    """Replace detected outliers with a scalar or nearest-neighbour fill.

    Args:
        arr: Input array.
        method: Outlier-detection strategy. See :func:`outlier_mask`.
        k: Outlier threshold in scaled units.
        fill: ``"median"`` replaces outliers with the median of the
            inliers; ``"nan"`` sets them to NaN; ``"interp"`` fills
            them by nearest-neighbour interpolation over the spatial
            axes via :func:`gap_fill_nearest`.

    Returns:
        Array of the same shape as ``arr`` with outliers replaced.

    Raises:
        ValueError: If ``method`` or ``fill`` is not one of the
            documented choices.
    """
    values = np.asarray(arr, dtype=float)
    mask = outlier_mask(values, method=method, k=k)
    if fill == "median":
        return np.where(mask, np.nanmedian(values[~mask]), values)
    if fill == "nan":
        return np.where(mask, np.nan, values)
    if fill == "interp":
        return gap_fill_nearest(np.where(mask, np.nan, values))
    raise ValueError("fill must be 'median', 'nan', or 'interp'")


def saturation_flag(
    arr: Shaped[np.ndarray, "*dims"], *, threshold: float | None = None
) -> Bool[np.ndarray, "*dims"]:
    """Flag pixels at or above a saturation threshold.

    Args:
        arr: Input array of any dtype.
        threshold: Saturation cutoff. When ``None`` defaults to
            ``np.iinfo(dtype).max`` for integer arrays and ``1.0`` for
            float arrays (the standard top-of-atmosphere reflectance
            ceiling).

    Returns:
        Boolean array of the same shape; ``True`` marks saturated
        pixels.
    """
    values = np.asarray(arr)
    if threshold is None:
        threshold = (
            float(np.iinfo(values.dtype).max)
            if np.issubdtype(values.dtype, np.integer)
            else 1.0
        )
    return values >= threshold
