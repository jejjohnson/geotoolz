"""Normalization primitives.

Mostly pure-numpy; :func:`clahe` additionally delegates to
:func:`skimage.exposure.equalize_adapthist` for the contrast-limited
adaptive histogram equalization step.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Shaped
from skimage.exposure import equalize_adapthist

from geotoolz._src.stretch import percentile_stretch


def stat_axes(
    arr: Shaped[np.ndarray, "*dims"], *, per_band: bool = True
) -> tuple[int, ...] | None:
    """Return spatial reduction axes for per-band remote-sensing arrays.

    Statistics for a channel-first cube (``(C, H, W)`` or
    ``(T, C, H, W)``) should reduce over the trailing spatial axes so
    each band keeps its own value; a 2-D map (or a global reduction)
    uses ``None``.

    Args:
        arr: Input array whose rank decides the reduction mode.
        per_band: If ``True`` (default) and ``arr`` has three or more
            dimensions, reduce per band over ``(-2, -1)``. If ``False``
            always reduce globally.

    Returns:
        ``(-2, -1)`` for per-band reductions, or ``None`` for a global
        reduction (also for 2-D inputs).
    """
    if per_band and arr.ndim >= 3:
        return (-2, -1)
    return None


def reshape_stat(
    stat: Float[np.ndarray, "*stat"] | float,
    arr: Shaped[np.ndarray, "*dims"],
    axis: tuple[int, ...] | None,
) -> Float[np.ndarray, "*bcast"]:
    """Reshape a reduced statistic so it broadcasts over ``arr``.

    Inverse of the ``axis`` reduction: a statistic computed with
    ``np.nanmean(arr, axis=axis)`` (and friends) loses the reduced
    axes, so it can't broadcast back against ``arr`` directly when the
    kept axes are leading. This re-inserts singleton dimensions at the
    reduced positions.

    Args:
        stat: Scalar or reduced-statistic array (e.g. per-band means of
            shape ``(C,)`` for a ``(C, H, W)`` input).
        arr: The array the statistic was computed from.
        axis: The axes that were reduced, or ``None`` for a global /
            scalar statistic.

    Returns:
        A float array broadcastable against ``arr`` (returned unchanged
        when it is scalar, ``axis`` is ``None``, or its shape does not
        match the kept axes).
    """
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
    arr: Float[np.ndarray, "*dims"],
    *,
    percentiles: list[float] | tuple[float, ...] = (1.0, 99.0),
    axis: tuple[int, ...] | None = (-2, -1),
) -> dict[str, np.ndarray]:
    """Compute NaN-aware statistics over spatial axes.

    Args:
        arr: Input float array, typically a channel-first ``(C, H, W)``
            cube. NaN pixels are excluded from every statistic.
        percentiles: Percentiles (in ``[0, 100]``) to compute alongside
            the moments. Default ``(1.0, 99.0)``.
        axis: Axes to reduce over. Default ``(-2, -1)`` yields one value
            per band; ``None`` reduces globally.

    Returns:
        Dict with keys ``"mean"``, ``"std"``, ``"min"``, ``"max"``
        (each shaped like the kept axes, e.g. ``(C,)``) and
        ``"percentiles"`` of shape ``(len(percentiles), C)``.
    """
    return {
        "mean": np.nanmean(arr, axis=axis),
        "std": np.nanstd(arr, axis=axis),
        "min": np.nanmin(arr, axis=axis),
        "max": np.nanmax(arr, axis=axis),
        "percentiles": np.nanpercentile(arr, percentiles, axis=axis),
    }


def standard_scale(
    arr: Float[np.ndarray, "*dims"],
    mean: Float[np.ndarray, "*stat"] | float,
    std: Float[np.ndarray, "*stat"] | float,
    *,
    axis: tuple[int, ...] | None = (-2, -1),
) -> Float[np.ndarray, "*dims"]:
    r"""Apply z-score scaling while preserving NaN pixels.

    .. math::

        y \;=\; \frac{x - \mu}{\sigma}

    Args:
        arr: Input float array. NaN pixels propagate through untouched.
        mean: Scalar or per-band mean :math:`\mu` (shape matching the
            kept axes, e.g. ``(C,)`` for a ``(C, H, W)`` input).
        std: Scalar or per-band standard deviation :math:`\sigma`. Bands
            with ``std == 0`` fall back to a divisor of ``1`` so a
            constant band maps to zero instead of ``inf`` / ``nan``.
        axis: Axes the statistics were reduced over. Default
            ``(-2, -1)``; ``None`` for global statistics.

    Returns:
        Float array of the same shape as ``arr``.
    """
    mean_b = reshape_stat(mean, arr, axis)
    std_b = reshape_stat(std, arr, axis)
    denom = np.where(std_b != 0, std_b, 1.0)
    return (arr - mean_b) / denom


def robust_scale(
    arr: Float[np.ndarray, "*dims"],
    median: Float[np.ndarray, "*stat"] | float,
    iqr: Float[np.ndarray, "*stat"] | float,
    *,
    axis: tuple[int, ...] | None = (-2, -1),
) -> Float[np.ndarray, "*dims"]:
    r"""Apply median/IQR scaling while preserving NaN pixels.

    .. math::

        y \;=\; \frac{x - \mathrm{median}}{Q_{3} - Q_{1}}

    The outlier-robust counterpart of :func:`standard_scale` — bright
    outliers (clouds, glint, saturation) barely move the median and
    interquartile range.

    Args:
        arr: Input float array. NaN pixels propagate through untouched.
        median: Scalar or per-band median (shape matching the kept
            axes).
        iqr: Scalar or per-band interquartile range ``Q3 - Q1``. Bands
            with ``iqr == 0`` fall back to a divisor of ``1``.
        axis: Axes the statistics were reduced over. Default
            ``(-2, -1)``; ``None`` for global statistics.

    Returns:
        Float array of the same shape as ``arr``.
    """
    median_b = reshape_stat(median, arr, axis)
    iqr_b = reshape_stat(iqr, arr, axis)
    denom = np.where(iqr_b != 0, iqr_b, 1.0)
    return (arr - median_b) / denom


def minmax_scale(
    arr: Float[np.ndarray, "*dims"],
    vmin: Float[np.ndarray, "*stat"] | float,
    vmax: Float[np.ndarray, "*stat"] | float,
    *,
    out_range: tuple[float, float] = (0.0, 1.0),
    axis: tuple[int, ...] | None = (-2, -1),
) -> Float[np.ndarray, "*dims"]:
    r"""Linearly map ``[vmin, vmax]`` into ``out_range``.

    .. math::

        y \;=\; \frac{x - v_{\min}}{v_{\max} - v_{\min}}
                \cdot (o_{\max} - o_{\min}) + o_{\min}

    Args:
        arr: Input float array. NaN pixels propagate through untouched.
        vmin: Scalar or per-band lower bound (maps to ``out_range[0]``).
        vmax: Scalar or per-band upper bound (maps to ``out_range[1]``).
            Bands with ``vmax <= vmin`` fall back to a divisor of ``1``.
        out_range: ``(out_min, out_max)`` target range; must be
            increasing. Default ``(0.0, 1.0)``.
        axis: Axes the bounds were reduced over. Default ``(-2, -1)``;
            ``None`` for global bounds.

    Returns:
        Float array of the same shape as ``arr``. Values outside
        ``[vmin, vmax]`` are *not* clipped.

    Raises:
        ValueError: If ``out_range`` is not a two-element increasing
            tuple.
    """
    out_min, out_max = validate_out_range(out_range)
    vmin_b = reshape_stat(vmin, arr, axis)
    vmax_b = reshape_stat(vmax, arr, axis)
    denom = np.where(vmax_b > vmin_b, vmax_b - vmin_b, 1.0)
    return (arr - vmin_b) / denom * (out_max - out_min) + out_min


def percentile_clip(
    arr: Shaped[np.ndarray, "*dims"],
    *,
    lower: float = 1.0,
    upper: float = 99.0,
    axis: tuple[int, ...] | None = (-2, -1),
) -> Float[np.ndarray, "*dims"]:
    """Clip to percentile bounds and stretch the result into ``[0, 1]``.

    Thin delegating wrapper over the shared
    :func:`geotoolz._src.stretch.percentile_stretch` (NaN-aware; a
    constant slice maps to ``0``).

    Args:
        arr: Input array of any shape.
        lower: Lower percentile in ``[0, 100]``. Default ``1.0``.
        upper: Upper percentile in ``[0, 100]``. Default ``99.0``.
        axis: Axes to compute the percentiles over. Default ``(-2, -1)``
            stretches each leading band / time slice independently;
            ``None`` uses one global pair of thresholds.

    Returns:
        Float array of the same shape with values in ``[0, 1]`` (NaNs
        propagate through untouched).

    Raises:
        ValueError: If ``upper <= lower``.
    """
    if upper <= lower:
        raise ValueError(
            f"upper must be greater than lower; got lower={lower}, upper={upper}"
        )
    return percentile_stretch(arr, lower, upper, axis=axis)


def histogram_match(
    source: Float[np.ndarray, "*dims"],
    reference: Float[np.ndarray, "*ref"],
) -> Float[np.ndarray, "*dims"]:
    """Match source values to the empirical CDF of ``reference``.

    Per-band monotone remapping: each source value is replaced by the
    reference value at the same empirical quantile. NaN pixels in the
    source are left untouched; NaNs in the reference are excluded from
    its CDF.

    Args:
        source: Array to remap. If 3-D+ (``(C, H, W)``), bands are
            matched independently.
        reference: Reference array. Either band-matched (same leading
            band count as ``source``) or a single 2-D map applied to
            every source band.

    Returns:
        Float array of the same shape as ``source`` with its per-band
        histograms reshaped to the reference.
    """
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


def clahe(
    arr: Shaped[np.ndarray, "*dims"],
    *,
    kernel_size: int | tuple[int, int] | None = None,
    clip_limit: float = 0.01,
    nbins: int = 256,
) -> Float[np.ndarray, "*dims"]:
    """Apply contrast-limited adaptive histogram equalization per image band.

    Wraps :func:`skimage.exposure.equalize_adapthist`. Each band of a
    3-D+ cube is equalized independently. Because skimage requires
    inputs in ``[0, 1]``, every slice is rescaled to ``[0, 1]`` over its
    finite range, equalized, then mapped back to its native units. NaN
    pixels are preserved and excluded from the histograms.

    Args:
        arr: Input array; trailing axes are ``(H, W)``.
        kernel_size: Contextual-region shape for the local histograms.
            ``None`` uses skimage's default (1/8 of the slice height /
            width).
        clip_limit: Contrast-limiting clip threshold in ``[0, 1]``.
            Default ``0.01``.
        nbins: Number of histogram bins. Default ``256``.

    Returns:
        Float array of the same shape, equalized per band (constant
        slices are returned unchanged).
    """
    values = np.asarray(arr, dtype=float)
    if values.ndim >= 3:
        return np.stack(
            [
                _clahe_slice(
                    band,
                    kernel_size=kernel_size,
                    clip_limit=clip_limit,
                    nbins=nbins,
                )
                for band in values
            ]
        )
    return _clahe_slice(
        values,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
        nbins=nbins,
    )


def _clahe_slice(
    values: np.ndarray,
    *,
    kernel_size: int | tuple[int, int] | None,
    clip_limit: float,
    nbins: int,
) -> np.ndarray:
    out = np.array(values, dtype=float, copy=True)
    valid = np.isfinite(values)
    if not np.any(valid):
        return out
    # skimage.exposure.equalize_adapthist requires inputs in [0, 1] (or unsigned
    # int); arbitrary-range floats are silently rescaled by max, which mis-scales
    # the output back into the wrong native units. Rescale per-slice to [0, 1]
    # using the valid-finite range, apply CLAHE, and undo the scaling on the
    # valid pixels. NaN pixels are left untouched.
    finite_values = values[valid]
    v_min = float(finite_values.min())
    v_max = float(finite_values.max())
    if v_max <= v_min:
        # Degenerate slice (constant or near-constant); CLAHE is a no-op.
        return out
    span = v_max - v_min
    fill = (float(np.median(finite_values)) - v_min) / span
    normalised = np.where(valid, (values - v_min) / span, fill)
    np.clip(normalised, 0.0, 1.0, out=normalised)
    equalized = equalize_adapthist(
        normalised,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
        nbins=nbins,
    )
    out[valid] = (equalized[valid] * span) + v_min
    return out


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


def log_scale(
    arr: Float[np.ndarray, "*dims"], *, base: float = 10.0, eps: float = 1e-6
) -> Float[np.ndarray, "*dims"]:
    r"""Apply log scaling with a small offset for zero-valued pixels.

    .. math::

        y \;=\; \log_{\text{base}}(\max(x, 0) + \epsilon)

    Compresses heavy-tailed distributions (radar backscatter, fire
    radiative power, ...). Negative inputs are clipped to zero before
    the log; ``eps`` keeps the log finite at zero.

    Args:
        arr: Input float array. NaN pixels propagate through untouched.
        base: Logarithm base. Must be positive and not equal to ``1``.
            Default ``10.0``.
        eps: Small positive offset added before the log. Default
            ``1e-6``.

    Returns:
        Float array of the same shape.

    Raises:
        ValueError: If ``base`` or ``eps`` is out of range.
    """
    if base <= 0 or base == 1.0:
        raise ValueError(f"base must be positive and not equal to 1; got {base}")
    if eps <= 0:
        raise ValueError(f"eps must be positive; got {eps}")
    return np.log(np.maximum(arr, 0.0) + eps) / np.log(base)


def asinh_scale(
    arr: Float[np.ndarray, "*dims"], *, a: float = 1.0
) -> Float[np.ndarray, "*dims"]:
    r"""Apply inverse-hyperbolic-sine scaling.

    .. math::

        y \;=\; \mathrm{asinh}(x / a)

    Linear near zero, logarithmic for ``|x| >> a``; symmetric and
    well-defined for negative values (unlike ``log``).

    Args:
        arr: Input float array. NaN pixels propagate through untouched.
        a: Scale parameter marking the linear-to-log transition. Must
            be strictly positive. Default ``1.0``.

    Returns:
        Float array of the same shape.

    Raises:
        ValueError: If ``a <= 0``.
    """
    if a <= 0:
        raise ValueError(f"a must be positive; got {a}")
    return np.arcsinh(arr / a)


def power_scale(
    arr: Float[np.ndarray, "*dims"], *, gamma: float = 0.5
) -> Float[np.ndarray, "*dims"]:
    r"""Apply non-negative power scaling.

    .. math::

        y \;=\; \max(x, 0)^{\gamma}

    A simple gamma-style brightness curve; negative inputs are clipped
    to zero before the power to avoid complex results.

    Args:
        arr: Input float array. NaN pixels propagate through untouched.
        gamma: Power exponent. ``< 1`` brightens midtones, ``> 1``
            darkens. Must be strictly positive. Default ``0.5``.

    Returns:
        Float array of the same shape.

    Raises:
        ValueError: If ``gamma <= 0``.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive; got {gamma}")
    return np.maximum(arr, 0.0) ** gamma


def validate_out_range(out_range: tuple[float, ...]) -> tuple[float, float]:
    """Validate and return a two-element increasing output range.

    Args:
        out_range: Candidate ``(out_min, out_max)`` tuple.

    Returns:
        The validated ``(out_min, out_max)`` pair.

    Raises:
        ValueError: If ``out_range`` does not have exactly two elements
            or is not strictly increasing.
    """
    if len(out_range) != 2:
        raise ValueError(f"out_range must contain exactly two values; got {out_range}")
    out_min, out_max = out_range
    if out_max <= out_min:
        raise ValueError(f"out_range must be increasing; got {out_range}")
    return out_min, out_max
