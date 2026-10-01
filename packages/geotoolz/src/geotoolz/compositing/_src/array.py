"""Tier-A primitives — pure-numpy per-pixel compositing math.

Every function here works on plain ``numpy.ndarray`` stacks with the
frame (time / source) axis leading: ``(T, ..., H, W)``. No
``GeoTensor`` knowledge, no nodata inference and no carrier rewrapping
-- the operators in :mod:`geotoolz.compositing._src.operators` judge
nodata per frame, call these primitives and rewrap the result.

Missing samples are encoded as ``NaN`` in float stacks (see
:func:`mask_frames`), so every reduction here is NaN-aware.

Examples:
    >>> import numpy as np
    >>> from geotoolz.compositing import median_composite, take_by_spatial_index
    >>> stack = np.stack([np.full((2, 2), v, dtype=float) for v in (1, 3, 2)])
    >>> median_composite(stack)[0, 0]
    np.float64(2.0)
    >>> take_by_spatial_index(stack, np.zeros((2, 2), dtype=int))[0, 0]
    np.float64(1.0)
"""

from __future__ import annotations

import warnings
from typing import Literal

import einx
import numpy as np
from jaxtyping import Bool, Float, Int, Num, Shaped


NanPolicy = Literal["ignore", "propagate"]
BlendMethod = Literal["mean", "weighted_mean", "ivw"]

_DAYS_PER_YEAR = 365


def broadcast_frame_valid(
    valid: Bool[np.ndarray, "t h w"], shape: tuple[int, ...]
) -> Bool[np.ndarray, "t *dims h w"]:
    """Broadcast ``(T, H, W)`` frame validity against a ``(T, ..., H, W)`` stack.

    Args:
        valid: Per-frame spatial validity.
        shape: Target stack shape ``(T, ..., H, W)``.

    Returns:
        A read-only broadcast view of ``valid`` with shape ``shape``.

    Examples:
        >>> broadcast_frame_valid(np.ones((2, 3, 4), bool), (2, 5, 3, 4)).shape
        (2, 5, 3, 4)
    """
    extra = len(shape) - valid.ndim
    return np.broadcast_to(
        valid.reshape((valid.shape[0], *([1] * extra), *valid.shape[1:])), shape
    )


def mask_frames(
    stack: Num[np.ndarray, "t *dims h w"], valid: Bool[np.ndarray, "t h w"]
) -> Float[np.ndarray, "t *dims h w"]:
    """Float copy of ``stack`` with every invalid frame-pixel set to NaN.

    Float stacks keep their dtype; integer stacks become ``float64`` --
    the dtype ``np.median`` / integer division already produced for them.

    Args:
        stack: ``(T, ..., H, W)`` frame stack.
        valid: ``(T, H, W)`` per-frame validity; ``False`` becomes NaN in
            every band of that frame-pixel.

    Returns:
        The masked float copy.

    Examples:
        >>> stack = np.ones((2, 1, 2), dtype=np.int16)
        >>> mask_frames(stack, np.array([[[True, False]], [[True, True]]]))[0]
        array([[ 1., nan]])
    """
    dtype = stack.dtype if np.issubdtype(stack.dtype, np.inexact) else np.float64
    out = stack.astype(dtype, copy=True)
    out[~broadcast_frame_valid(valid, out.shape)] = np.nan
    return out


def take_by_spatial_index(
    stack: Shaped[np.ndarray, "t *dims h w"],
    index: Int[np.ndarray, "h w"] | Int[np.ndarray, "*dims h w"],
) -> Shaped[np.ndarray, "*dims h w"]:
    """Select one frame per sample from a ``(T, ..., H, W)`` stack.

        out[..., y, x] = stack[index[..., y, x], ..., y, x]

    Args:
        stack: Frame stack, frame axis first.
        index: Frame index in ``[0, T)`` -- one per pixel ``(H, W)``
            (shared by every band), or one per sample ``(..., H, W)``.

    Returns:
        A new ``(..., H, W)`` array with ``stack``'s dtype.

    Examples:
        >>> stack = np.arange(8).reshape(2, 2, 2)
        >>> take_by_spatial_index(stack, np.array([[0, 1], [1, 0]]))
        array([[0, 5],
               [6, 3]])
    """
    index = np.broadcast_to(index, stack.shape[1:])
    return einx.get_at("[t] ..., ... -> ...", stack, index)


def median_composite(
    stack: Float[np.ndarray, "t *dims h w"], *, nan_policy: NanPolicy = "ignore"
) -> Float[np.ndarray, "*dims h w"]:
    """Per-pixel median over the frame axis of a NaN-masked stack.

        m(y, x) = median_t stack[t, ..., y, x]

    Args:
        stack: ``(T, ..., H, W)`` float stack; missing samples are NaN
            (see :func:`mask_frames`).
        nan_policy: ``"ignore"`` skips NaN samples (``np.nanmedian``);
            ``"propagate"`` makes any NaN sample poison the pixel.

    Returns:
        The ``(..., H, W)`` median; pixels with no finite sample are NaN.

    Examples:
        >>> stack = np.array([[[1.0]], [[np.nan]], [[3.0]]])
        >>> median_composite(stack)
        array([[2.]])
        >>> median_composite(stack, nan_policy="propagate")
        array([[nan]])
    """
    with warnings.catch_warnings():
        # All-NaN pixels are expected; callers overwrite them with a fill.
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        if nan_policy == "ignore":
            return np.nanmedian(stack, axis=0)
        return np.median(stack, axis=0)


def mean_composite(
    stack: Float[np.ndarray, "t *dims h w"],
    clear: Bool[np.ndarray, "t *dims h w"],
    *,
    min_valid: int = 1,
    nan_policy: NanPolicy = "ignore",
) -> tuple[Float[np.ndarray, "*dims h w"], Int[np.ndarray, "*dims h w"]]:
    """Per-pixel mean over the clear samples of a frame stack.

        μ(y, x) = Σ_t [clear_t] · x_t / Σ_t [clear_t]

    Args:
        stack: ``(T, ..., H, W)`` float stack; missing samples are NaN.
        clear: Same-shape mask of samples allowed to contribute.
        min_valid: Minimum contributors per pixel; pixels with fewer
            come out NaN.
        nan_policy: ``"ignore"`` also drops NaN samples from ``clear``;
            ``"propagate"`` lets a NaN clear sample poison the mean.

    Returns:
        ``(mean, count)`` -- the mean (float, NaN below ``min_valid``) and
        the per-pixel contributor count.

    Examples:
        >>> stack = np.array([[[1.0]], [[3.0]], [[np.nan]]])
        >>> mean, count = mean_composite(stack, np.ones_like(stack, bool))
        >>> mean, count
        (array([[2.]]), array([[2]]))
    """
    valid = clear & ~np.isnan(stack) if nan_policy == "ignore" else clear
    count = np.sum(valid, axis=0)
    total = np.sum(np.where(valid, stack, 0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = total / count
    values = np.where(count >= min_valid, values, np.nan)
    return values.astype(np.result_type(values.dtype, np.float32), copy=False), count


def doy_distance(
    doy: Float[np.ndarray, "*dims"], target_doy: float
) -> Float[np.ndarray, "*dims"]:
    """Circular day-of-year distance to ``target_doy``, in days.

        Δ = |doy − target_doy| mod 365,   d = min(Δ, 365 − Δ)

    so DOY 360 and target DOY 5 are 10 days apart, not 355. A 365-day
    year is assumed (DOY 366 coincides with DOY 1).

    Args:
        doy: Day(s) of year.
        target_doy: Reference day of year.

    Returns:
        The circular distance, ``float64``.

    Examples:
        >>> float(doy_distance(np.array(360.0), 5))
        10.0
    """
    delta = np.mod(
        np.abs(np.asarray(doy, dtype=np.float64) - target_doy), _DAYS_PER_YEAR
    )
    return np.minimum(delta, _DAYS_PER_YEAR - delta)


def doy_score(
    doy: Float[np.ndarray, "*dims"], target_doy: float, sigma: float
) -> Float[np.ndarray, "*dims"]:
    """Gaussian day-of-year score (Griffiths et al., 2013), peak-normalised.

        S_doy = exp(−½ · (d / σ_doy)²),   d = circular DOY distance

    Griffiths et al. score DOY with a Gaussian centred on the target DOY;
    the ``1/(σ√(2π))`` density factor is dropped so ``S_doy ∈ (0, 1]``
    shares a scale with the other scores (``S_doy = e^(−½) ≈ 0.61`` at
    ``d = σ_doy``).

    Args:
        doy: Day(s) of year.
        target_doy: Day of year the Gaussian is centred on.
        sigma: ``σ_doy`` in days.

    Returns:
        The score in ``(0, 1]``.

    Examples:
        >>> float(doy_score(np.array(180.0), 180, 30.0))
        1.0
    """
    d = doy_distance(doy, target_doy)
    return np.exp(-0.5 * (d / sigma) ** 2)


def view_angle_score(
    view_angle: Float[np.ndarray, "*dims"], sigma: float
) -> Float[np.ndarray, "*dims"]:
    """Gaussian view-zenith-angle score, best at nadir.

        S_view = exp(−½ · (θ / σ_θ)²)

    Args:
        view_angle: View zenith angle ``θ`` in degrees (sign ignored).
        sigma: ``σ_θ`` in degrees.

    Returns:
        The score in ``(0, 1]``.

    Examples:
        >>> float(view_angle_score(np.array(0.0), 15.0))
        1.0
    """
    theta = np.asarray(view_angle, dtype=np.float64)
    return np.exp(-0.5 * (theta / sigma) ** 2)


def cloud_distance_score(
    distance: Float[np.ndarray, "*dims"], *, d_req: float, d_min: float, slope: float
) -> Float[np.ndarray, "*dims"]:
    """Sigmoid distance-to-cloud score (Griffiths et al., 2013, eq. 2).

        S_cloud = 1 / (1 + exp(−k · (min(D, D_req) − (D_req − D_min) / 2)))

    Args:
        distance: ``D``, distance to the nearest cloud / shadow.
        d_req: ``D_req``, distance at which the score saturates.
        d_min: ``D_min``, the minimum distance.
        slope: ``k``, sigmoid slope per distance unit.

    Returns:
        The score in ``(0, 1)``.

    Examples:
        >>> float(cloud_distance_score(np.array(25.0), d_req=50, d_min=0, slope=0.2))
        0.5
    """
    d = np.minimum(np.asarray(distance, dtype=np.float64), d_req)
    return 1.0 / (1.0 + np.exp(-slope * (d - (d_req - d_min) / 2.0)))


def opacity_score(
    opacity: Float[np.ndarray, "*dims"], *, low: float, high: float
) -> Float[np.ndarray, "*dims"]:
    """Piecewise-linear atmospheric-opacity score (White et al., 2014).

        S_opacity = clip((τ_high − τ) / (τ_high − τ_low), 0, 1)

    i.e. 1 for ``τ ≤ τ_low``, 0 for ``τ ≥ τ_high``, linear between.

    Args:
        opacity: Unitless atmospheric opacity ``τ``.
        low: ``τ_low``.
        high: ``τ_high``.

    Returns:
        The score in ``[0, 1]``.

    Examples:
        >>> float(opacity_score(np.array(0.25), low=0.2, high=0.3))
        0.5
    """
    tau = np.asarray(opacity, dtype=np.float64)
    return np.clip((high - tau) / (high - low), 0.0, 1.0)


def bap_scores(
    scores: Float[np.ndarray, "t k h w"], weights: Float[np.ndarray, " k"]
) -> Float[np.ndarray, "t h w"]:
    """Weighted Best-Available-Pixel score per frame-pixel.

        S_t(y, x) = Σ_k w_k · s_{t,k}(y, x)

    Args:
        scores: ``(T, K, H, W)`` per-criterion scores in ``[0, 1]``
            (e.g. view, DOY, cloud distance, opacity).
        weights: ``(K,)`` criterion weights.

    Returns:
        The ``(T, H, W)`` ``float32`` total score.

    Examples:
        >>> scores = np.ones((2, 4, 1, 1))
        >>> bap_scores(scores, np.array([0.3, 0.4, 0.2, 0.1]))[:, 0, 0]
        array([1., 1.], dtype=float32)
    """
    weights = np.asarray(weights, dtype=np.float32)
    return np.einsum("k,tkhw->thw", weights, scores).astype(np.float32, copy=False)


def blend_weighted(
    stack: Float[np.ndarray, "n *dims h w"],
    weights: Float[np.ndarray, "n *dims h w"],
    *,
    nan_policy: NanPolicy = "ignore",
) -> Float[np.ndarray, "*dims"]:
    """Weighted mean over the leading source axis of a NaN-masked stack.

        x̄ = Σ_n w_n · x_n / Σ_n w_n

    Args:
        stack: ``(N, ...)`` float stack; missing samples are NaN.
        weights: Weights broadcastable to ``stack`` (``(N, 1, ..., 1)``
            scalars per source, or per-pixel inverse variances).
        nan_policy: ``"ignore"`` drops NaN samples and renormalises the
            surviving weights (all-NaN pixels become NaN);
            ``"propagate"`` makes any NaN sample poison the pixel.

    Returns:
        The blended ``(...)`` array; NaN where no weight survives.

    Examples:
        >>> stack = np.array([[1.0, np.nan], [3.0, 5.0]])
        >>> blend_weighted(stack, np.ones((2, 1)))
        array([2., 5.])
        >>> blend_weighted(stack, np.ones((2, 1)), nan_policy="propagate")
        array([ 2., nan])
    """
    if nan_policy == "propagate":
        # Any NaN in any source ↦ NaN output at that pixel.
        nan_mask = np.isnan(stack).any(axis=0)
        num = (stack * weights).sum(axis=0)
        den = np.broadcast_to(weights, stack.shape).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(den > 0, num / den, np.nan)
        return np.where(nan_mask, np.nan, result)
    # Zero the weight wherever the value is NaN, then drop NaNs from the
    # numerator. The denominator is the sum of surviving weights -- a
    # pixel with all-NaN inputs ends up with den == 0 and becomes NaN.
    valid = ~np.isnan(stack)
    safe_values = np.where(valid, stack, 0.0)
    safe_weights = np.where(valid, weights, 0.0)
    num = (safe_values * safe_weights).sum(axis=0)
    den = safe_weights.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


__all__ = [
    "BlendMethod",
    "NanPolicy",
    "bap_scores",
    "blend_weighted",
    "broadcast_frame_valid",
    "cloud_distance_score",
    "doy_distance",
    "doy_score",
    "mask_frames",
    "mean_composite",
    "median_composite",
    "opacity_score",
    "take_by_spatial_index",
    "view_angle_score",
]
