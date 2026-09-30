"""Tier-A primitives for image restoration.

All functions in this module are pure NumPy / SciPy on plain
``ndarray`` inputs. Spatial filters (despeckle, destripe, denoise,
gap fill, outliers) take ``(..., H, W)`` arrays, filter the trailing two
axes and return an array of the same shape. The spectral transforms take
an array with a band axis (``(bands, H, W)`` by default):
:func:`pca_denoise` returns a same-shaped reconstruction, while
:func:`fit_pca` / :func:`fit_mnf` return a state dict (scores plus the
fitted transform) that :func:`inverse_pca` maps back to the input shape.
Carrier-aware ``Operator`` wrappers live in
:mod:`geotoolz.restore._src.operators`.

NaN convention: input NaNs are treated as missing pixels. Filters that
*restore* (denoise, despeckle, destripe) propagate NaNs through to the
output unchanged. Filters that *fill* (``gap_fill_*``) replace NaNs
with a finite estimate.
"""

from __future__ import annotations

import warnings
from typing import Literal, cast

import einx
import numpy as np
from jaxtyping import Bool, Float, Num, Shaped
from scipy import ndimage
from scipy.spatial import cKDTree

from geotoolz._src.samples import (
    SampleLayout,
    cube_to_samples,
    sample_layout,
    samples_to_cube,
)


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


def _local_moments(
    values: Float[np.ndarray, "*batch h w"], window: int
) -> tuple[Float[np.ndarray, "*batch h w"], Float[np.ndarray, "*batch h w"]]:
    """NaN-aware local mean and (population) variance over a square window."""
    size = _spatial_size(values, window)
    mean = _nanmean_filter(values, size)
    mean_sq = _nanmean_filter(values * values, size)
    return mean, np.maximum(mean_sq - mean * mean, 0.0)


def despeckle_lee(
    arr: Num[np.ndarray, "*batch h w"], *, window: int = 7, cu: float = 0.523
) -> Float[np.ndarray, "*batch h w"]:
    r"""Apply the Lee (1980) local-statistics speckle filter.

    Speckle model: ``z = x·v`` with ``x`` the unspeckled signal and ``v``
    unit-mean multiplicative noise of coefficient of variation
    ``Cᵤ = σᵥ`` (``cu``). With the local mean ``z̄`` and variance ``σ_z²``
    over a ``window`` × ``window`` neighbourhood, the model gives
    (Lee 1980)::

        σ_x² = max((σ_z² − Cᵤ²·z̄²) / (1 + Cᵤ²), 0)
        k    = σ_x² / (σ_x² + Cᵤ²·z̄²)
        x̂    = z̄ + k·(z − z̄)

    In terms of the local coefficient of variation ``Cᵢ = σ_z / z̄`` the
    gain is ``k = (Cᵢ² − Cᵤ²) / (Cᵢ² + Cᵤ⁴)`` clipped at ``0``, so
    ``k ∈ [0, 1)``: on homogeneous speckle (``Cᵢ ≤ Cᵤ``) ``k = 0`` and
    the filter returns the local mean; at edges and point targets
    (``Cᵢ ≫ Cᵤ``) ``k → 1`` and the pixel passes through. Dropping the
    ``O(Cᵤ⁴)`` term gives the familiar linearised ``k ≈ 1 − Cᵤ²/Cᵢ²``.

    Args:
        arr: Array of shape ``(..., H, W)``. NaNs are preserved and
            excluded from the window statistics.
        window: Side length of the local window. Must be positive.
        cu: Noise coefficient of variation ``Cᵤ``. ``1/√L`` for
            ``L``-look intensity; ``0.523 ≈ √(4/π − 1)`` is the
            single-look *amplitude* value (default).

    Returns:
        Smoothed array with the same shape and NaN positions as ``arr``.

    References:
        Lee, J.-S. (1980). Digital image enhancement and noise filtering
        by use of local statistics. IEEE TPAMI, 2(2), 165-168.
    """
    values = np.asarray(arr, dtype=float)
    mean, var = _local_moments(values, window)
    noise_var = (float(cu) * mean) ** 2
    signal_var = np.maximum((var - noise_var) / (1.0 + float(cu) ** 2), 0.0)
    denom = signal_var + noise_var
    gain = np.divide(signal_var, denom, out=np.zeros_like(signal_var), where=denom > 0)
    return _preserve_nan(values, mean + gain * (values - mean))


def despeckle_frost(
    arr: Num[np.ndarray, "*batch h w"], *, window: int = 7, damping: float = 2.0
) -> Float[np.ndarray, "*batch h w"]:
    r"""Apply the Frost (1982) adaptive exponential-kernel speckle filter.

    Each output pixel is a normalised weighted mean of its
    ``window`` × ``window`` neighbourhood with the exponentially
    decaying kernel::

        m(t) = exp(−K · Cᵥ² · |t|)
        x̂    = Σₜ m(t)·z(t) / Σₜ m(t)

    where ``K`` is ``damping``, ``|t|`` is the Euclidean distance (in
    pixels) from the centre pixel and ``Cᵥ² = σ_z² / z̄²`` is the local
    squared coefficient of variation, computed over the same window.
    On homogeneous areas ``Cᵥ²`` is small, the kernel is wide and the
    filter approaches the box mean; at edges ``Cᵥ²`` is large, the
    kernel narrows onto the centre pixel and the edge is preserved.
    Where ``z̄ = 0`` the kernel is the box mean.

    Args:
        arr: Array of shape ``(..., H, W)``. NaNs are preserved and
            excluded (zero weight) from the neighbourhood sums.
        window: Side length of the kernel window. Must be positive.
        damping: Damping factor ``K``. Larger values narrow the kernel
            (keep more edge contrast); ``0`` is the box mean.

    Returns:
        Smoothed array with the same shape and NaN positions as ``arr``.

    References:
        Frost, V. S., Stiles, J. A., Shanmugan, K. S., & Holtzman, J. C.
        (1982). A model for radar images and its application to adaptive
        digital filtering of multiplicative noise. IEEE TPAMI, 4(2),
        157-166.
    """
    values = np.asarray(arr, dtype=float)
    mean, var = _local_moments(values, window)
    cv_sq = np.divide(var, mean * mean, out=np.zeros_like(var), where=mean != 0)
    rate = float(damping) * cv_sq
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    half = int(window) // 2
    lo, hi = -half, int(window) - half
    pad = [(0, 0)] * (values.ndim - 2) + [(half, window - 1 - half)] * 2
    padded = np.pad(filled, pad, mode="edge")
    padded_valid = np.pad(valid, pad, mode="edge")
    height, width = values.shape[-2:]
    total = np.zeros_like(filled)
    weight_sum = np.zeros_like(filled)
    for dy in range(lo, hi):
        for dx in range(lo, hi):
            rows = slice(dy + half, dy + half + height)
            cols = slice(dx + half, dx + half + width)
            weight = np.exp(-rate * np.hypot(dy, dx)) * padded_valid[..., rows, cols]
            total += weight * padded[..., rows, cols]
            weight_sum += weight
    out = np.divide(
        total, weight_sum, out=np.full_like(total, np.nan), where=weight_sum > 0
    )
    return _preserve_nan(values, out)


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
    window: int | None = 21,
) -> Float[np.ndarray, "*batch h w"]:
    r"""Remove row or column striping by matching per-line statistics.

    For ``axis="column"`` (default) every column ``j`` is summarised over
    its rows; ``axis="row"`` swaps the roles. Leading axes are independent
    planes.

    ``method="mean"`` / ``"median"`` remove a per-column *offset*::

        out[:, j] = z[:, j] − (s_j − s̄)

    where ``s_j`` is the column mean (median) and ``s̄`` the mean (median)
    of all ``s_j``.

    ``method="moment_matching"`` removes a per-column *gain and offset*
    (Gadallah et al. 2000): each column is linearly rescaled so its mean
    ``μ_j`` and standard deviation ``σ_j`` match a reference ``(μᵣ, σᵣ)``::

        out[:, j] = (z[:, j] − μ_j) · σᵣ_j / σ_j + μᵣ_j

    With ``window=None`` the reference is global (``μᵣ`` and ``σᵣ`` are
    the averages of ``μ_j`` and ``σ_j`` over all columns); with an integer
    ``window`` it is the moving average of ``μ_j`` / ``σ_j`` over the
    ``window`` neighbouring columns, which keeps genuine cross-track
    trends (e.g. illumination) while removing the detector-to-detector
    stripes. Within-column detail is untouched apart from the per-column
    linear map. A column with ``σ_j = 0`` only has its offset matched.

    Args:
        arr: Array of shape ``(..., H, W)``. NaNs are preserved and
            excluded from the statistics.
        method: ``"mean"``, ``"median"`` or ``"moment_matching"``.
        axis: Striping direction (``"column"`` for vertical stripes,
            ``"row"`` for horizontal).
        window: Number of neighbouring columns (rows) averaged into the
            moment-matching reference, or ``None`` for a global
            reference. Only consulted for ``method="moment_matching"``.

    Returns:
        Destriped array with the same shape and NaN positions as
        ``arr``.

    Raises:
        ValueError: If ``arr`` has fewer than two dimensions, ``method``
            is not one of the documented choices or ``window`` is not
            positive.

    References:
        Gadallah, F. L., Csillag, F., & Smith, E. J. M. (2000).
        Destriping multisensor imagery with moment matching.
        International Journal of Remote Sensing, 21(12), 2505-2511.
    """
    values = np.asarray(arr, dtype=float)
    if values.ndim < 2:
        raise ValueError("destripe_column expects at least two spatial dimensions")
    if method not in {"mean", "median", "moment_matching"}:
        raise ValueError("method must be 'mean', 'median', or 'moment_matching'")
    spatial_axis = -1 if axis == "column" else -2
    reduce_axis = -2 if axis == "column" else -1
    # All-NaN lines (e.g. a nodata column) have NaN statistics by design;
    # silence numpy's empty-slice warnings for them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method != "moment_matching":
            reducer = np.nanmedian if method == "median" else np.nanmean
            profile = reducer(values, axis=reduce_axis, keepdims=True)
            target = reducer(profile, axis=spatial_axis, keepdims=True)
            return _preserve_nan(values, values - (profile - target))
        line_mean = np.nanmean(values, axis=reduce_axis, keepdims=True)
        line_std = np.nanstd(values, axis=reduce_axis, keepdims=True)
    if window is None:
        ref_mean = np.nanmean(line_mean, axis=spatial_axis, keepdims=True)
        ref_std = np.nanmean(line_std, axis=spatial_axis, keepdims=True)
    else:
        if window <= 0:
            raise ValueError("window must be positive or None")
        size = [1] * values.ndim
        size[spatial_axis] = int(window)
        ref_mean = _nanmean_filter(line_mean, tuple(size))
        ref_std = _nanmean_filter(line_std, tuple(size))
    gain = np.divide(ref_std, line_std, out=np.ones_like(line_std), where=line_std > 0)
    return _preserve_nan(values, (values - line_mean) * gain + ref_mean)


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
    output. Edge pixels use ``mode="nearest"`` (the edge pixel is
    repeated outwards, not reflected).

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
    arr: Num[np.ndarray, "*dims"], *, n_components: int, axis: int = -3
) -> Float[np.ndarray, "*dims"]:
    """Reconstruct an array from its top PCA components along ``axis``.

    Convenience composition of :func:`fit_pca` and :func:`inverse_pca`.
    Keeping fewer components than bands suppresses band-uncorrelated
    noise while preserving the dominant spectral structure.

    Args:
        arr: Array with a band axis at position ``axis``; typically
            ``(bands, H, W)``. The components are fitted on the pixels
            that are finite in every band; NaN positions are re-stamped
            as NaN on the output.
        n_components: Number of principal components to keep. Must be
            between 1 and the number of bands.
        axis: Position of the band axis. Defaults to ``-3``.

    Returns:
        Reconstructed array with the same shape and NaN positions as
        ``arr``.

    Raises:
        ValueError: If ``n_components`` is out of range, any band is
            entirely NaN or no pixel is finite in every band.
    """
    model = fit_pca(arr, n_components=n_components, axis=axis)
    return inverse_pca(model["scores"], model)


def _band_samples(
    arr: Num[np.ndarray, "*dims"], axis: int, n_components: int | None, name: str
) -> tuple[
    Float[np.ndarray, "c n"],
    Bool[np.ndarray, "c n"],
    Bool[np.ndarray, " n"],
    SampleLayout,
    int,
]:
    """Band-major samples, their NaN mask, the complete-pixel mask, layout and ``k``.

    A pixel is *complete* when it is finite in every band; only complete
    pixels enter the fitted statistics.
    """
    values = np.asarray(arr, dtype=float)
    samples, layout = cube_to_samples(values, band_axis=axis)
    bands = layout.n_bands
    keep = bands if n_components is None else int(n_components)
    if not 1 <= keep <= bands:
        raise ValueError("n_components must be between 1 and the number of bands")
    flat = np.ascontiguousarray(samples.T)
    nan_mask = ~np.isfinite(flat)
    all_nan_bands = nan_mask.all(axis=1)
    if all_nan_bands.any():
        bad = tuple(int(i) for i in np.where(all_nan_bands)[0])
        raise ValueError(
            f"{name} cannot fit on bands that are entirely NaN: bands {bad}"
        )
    complete = ~nan_mask.any(axis=0)
    if complete.sum() < 2:
        raise ValueError(f"{name} needs at least two pixels finite in every band")
    return flat, nan_mask, complete, layout, keep


def _project(
    flat: Float[np.ndarray, "c n"],
    nan_mask: Bool[np.ndarray, "c n"],
    mean: Float[np.ndarray, " c"],
    components: Float[np.ndarray, "c k"],
) -> Float[np.ndarray, "k n"]:
    """Project mean-centred samples (NaN entries imputed by the mean) onto ``components``."""
    centered = np.where(nan_mask, 0.0, flat - mean[:, None])
    return einx.dot("c k, c n -> k n", components, centered)


def fit_pca(
    arr: Num[np.ndarray, "*dims"], *, n_components: int | None = None, axis: int = -3
) -> dict[str, np.ndarray | int | tuple[int, ...]]:
    """Fit PCA over a band axis and return scores plus reconstruction state.

    With ``X`` the ``(n, c)`` matrix of mean-centred pixels that are
    finite in every band and ``X = U·S·Vᵀ`` its SVD, the components are
    the columns of ``V`` and the explained variance of component ``i`` is
    ``sᵢ² / (n − 1)`` (the eigenvalues of the sample covariance). Every
    pixel is projected; NaN entries are imputed with the band mean for
    the projection and their positions are recorded so
    :func:`inverse_pca` can re-stamp them.

    Args:
        arr: Array with a band axis at position ``axis``; typically
            ``(bands, H, W)``.
        n_components: Number of components to keep. ``None`` (default)
            keeps all bands.
        axis: Position of the band axis. Defaults to ``-3``.

    Returns:
        State dict with keys ``"scores"`` (projected data, band axis
        replaced by the component axis), ``"components"`` (``(c, k)``
        projection), ``"loadings"`` (``(c, k)`` reconstruction; equal to
        ``"components"`` for PCA), ``"mean"``, ``"axis"``, ``"shape"``,
        ``"nan_mask"`` and ``"explained_variance"`` (``sᵢ² / (n − 1)``,
        sorted descending). Pass it verbatim to :func:`inverse_pca`.

    Raises:
        ValueError: If ``n_components`` is out of range, any band is
            entirely NaN or fewer than two pixels are finite in every
            band.
    """
    flat, nan_mask, complete, layout, keep = _band_samples(
        arr, axis, n_components, "PCA"
    )
    fit = flat[:, complete]
    mean = fit.mean(axis=1)
    u, s, _ = np.linalg.svd(fit - mean[:, None], full_matrices=False)
    components = u[:, :keep]
    scores = _project(flat, nan_mask, mean, components)
    return {
        "scores": scores.reshape((keep, *layout.sample_shape)),
        "components": components,
        "loadings": components,
        "mean": mean,
        "axis": axis,
        "shape": layout.shape,
        "nan_mask": nan_mask,
        "explained_variance": s[:keep] ** 2 / (fit.shape[1] - 1),
    }


def inverse_pca(
    scores: Num[np.ndarray, "*dims"],
    state: dict[str, np.ndarray | int | tuple[int, ...]],
) -> np.ndarray:
    """Reconstruct an array from PCA (or MNF) scores and state.

    Computes ``x̂ = L·y + x̄`` with ``L`` the ``(c, k)`` reconstruction
    loadings stored in ``state`` (the components themselves for PCA,
    ``Σ_N·A`` for MNF; see :func:`fit_mnf`).

    Args:
        scores: Projected data as returned in ``state["scores"]`` (the
            leading component axis must match ``state["loadings"]``).
        state: Reconstruction state produced by :func:`fit_pca` or
            :func:`fit_mnf`.

    Returns:
        Reconstructed array with the shape recorded in ``state`` and the
        original NaN positions re-stamped.
    """
    loadings = np.asarray(state["loadings"])
    mean = np.asarray(state["mean"])[:, None]
    shape = cast(tuple[int, ...], state["shape"])
    axis = int(state["axis"])
    flat_scores = np.asarray(scores, dtype=float).reshape(loadings.shape[1], -1)
    restored = einx.dot("c k, k n -> c n", loadings, flat_scores) + mean
    nan_mask = np.asarray(state["nan_mask"])
    restored = np.where(nan_mask, np.nan, restored)
    return samples_to_cube(restored.T, sample_layout(shape, band_axis=axis))


def shift_difference_noise_covariance(
    arr: Num[np.ndarray, "*dims"], *, axis: int = -3
) -> Float[np.ndarray, "c c"]:
    """Estimate the band noise covariance from horizontal shift differences.

    Green et al. (1988): with ``d(p) = z(p) − z(p + δ)`` the difference
    between each pixel and its right-hand neighbour (``δ`` = one step
    along the last non-band axis, i.e. along a row ``W``), and noise that
    is white in space while the signal is locally smooth,
    ``Cov(d) ≈ 2·Σ_N``, so::

        Σ_N = Cov(d) / 2

    Pairs that straddle a NaN in any band are skipped, and pairs never
    cross rows, frames or planes. The difference covariance is the
    unbiased (``n − 1``) sample covariance of ``d``.

    Args:
        arr: Array with a band axis at ``axis``, e.g. ``(C, H, W)`` or
            ``(T, C, H, W)``.
        axis: Position of the band axis. Defaults to ``-3``.

    Returns:
        ``(c, c)`` noise covariance estimate.

    Raises:
        ValueError: If fewer than two complete neighbour pairs exist.
    """
    values = np.asarray(arr, dtype=float)
    bands_last = np.moveaxis(values, axis, -1)
    diff = np.diff(bands_last, axis=-2).reshape(-1, bands_last.shape[-1])
    diff = diff[np.isfinite(diff).all(axis=1)]
    if diff.shape[0] < 2:
        raise ValueError(
            "MNF needs at least two horizontally adjacent pixel pairs that are "
            "finite in every band to estimate the noise covariance"
        )
    return np.atleast_2d(np.cov(diff, rowvar=False)) / 2.0


def fit_mnf(
    arr: Num[np.ndarray, "*dims"], *, n_components: int | None = None, axis: int = -3
) -> dict[str, np.ndarray | int | tuple[int, ...]]:
    """Fit a Minimum Noise Fraction transform (Green et al. 1988).

    Steps, with ``Σ`` the band covariance of the pixels that are finite in
    every band and ``Σ_N`` the shift-difference noise covariance
    (:func:`shift_difference_noise_covariance`):

    1. Noise whitening: ``Σ_N = L·Lᵀ`` (Cholesky) and ``W = L⁻¹``, so the
       whitened noise has identity covariance.
    2. PCA of the whitened data: ``W·Σ·Wᵀ = V·diag(λ)·Vᵀ`` with ``λ``
       sorted descending.
    3. MNF transform ``A = Wᵀ·V``; scores ``y = Aᵀ·(x − x̄)``.

    ``A`` solves the generalised eigenproblem ``Σ·a = λ·Σ_N·a`` with the
    normalisation ``Aᵀ·Σ_N·A = I``, so each score has unit noise
    variance and total variance ``λᵢ``. ``λᵢ`` is the inverse of the
    component's noise fraction; for signal uncorrelated with the noise
    its signal-to-noise ratio is ``λᵢ − 1``. Reconstruction from the
    first ``k`` components uses the loadings ``L_k = Σ_N·A_k`` (the first
    ``k`` columns of ``A⁻ᵀ``), so all components give an exact inverse.

    Args:
        arr: Array with a band axis at position ``axis``; typically
            ``(bands, H, W)`` or ``(T, bands, H, W)`` (one fit over every
            frame; noise pairs never cross frames).
        n_components: Number of components to keep. ``None`` (default)
            keeps all bands.
        axis: Position of the band axis. Defaults to ``-3``.

    Returns:
        State dict with the keys of :func:`fit_pca` except
        ``"explained_variance"``; instead ``"eigenvalues"`` (``λᵢ``,
        descending) and ``"noise_covariance"`` (``Σ_N``).
        ``"components"`` is ``A_k`` and ``"loadings"`` is ``Σ_N·A_k``.
        Pass it verbatim to :func:`inverse_pca`.

    Raises:
        ValueError: If ``n_components`` is out of range, any band is
            entirely NaN, too few complete pixels / neighbour pairs exist,
            or the noise covariance is singular (e.g. a noise-free band).

    References:
        Green, A. A., Berman, M., Switzer, P., & Craig, M. D. (1988). A
        transformation for ordering multispectral data in terms of image
        quality with implications for noise removal. IEEE TGRS, 26(1),
        65-74.
    """
    flat, nan_mask, complete, layout, keep = _band_samples(
        arr, axis, n_components, "MNF"
    )
    fit = flat[:, complete]
    mean = fit.mean(axis=1)
    cov = np.atleast_2d(np.cov(fit))
    noise_cov = shift_difference_noise_covariance(arr, axis=axis)
    try:
        chol = np.linalg.cholesky(noise_cov)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "MNF noise covariance is singular (a band has no pixel-to-pixel "
            "noise); drop constant or noise-free bands first"
        ) from err
    whiten = np.linalg.inv(chol)
    eigvals, eigvecs = np.linalg.eigh(whiten @ cov @ whiten.T)
    order = np.argsort(eigvals)[::-1]
    transform = whiten.T @ eigvecs[:, order]
    components = transform[:, :keep]
    scores = _project(flat, nan_mask, mean, components)
    return {
        "scores": scores.reshape((keep, *layout.sample_shape)),
        "components": components,
        "loadings": noise_cov @ components,
        "mean": mean,
        "axis": axis,
        "shape": layout.shape,
        "nan_mask": nan_mask,
        "eigenvalues": eigvals[order][:keep],
        "noise_covariance": noise_cov,
    }


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
