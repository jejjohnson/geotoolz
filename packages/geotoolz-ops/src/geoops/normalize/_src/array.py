"""Normalization primitives.

Mostly pure-numpy; :func:`clahe` additionally delegates to
:func:`skimage.exposure.equalize_adapthist` for the contrast-limited
adaptive histogram equalization step.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import Float, Shaped
from pipekit_array import array_namespace

from geoops._src.geo import reject_geotensor
from geoops._src.optional import optional_name
from geoops._src.shape import BAND_AXIS
from geoops._src.valid import namespace_of


if TYPE_CHECKING:
    pass


def stat_axes(
    arr: Shaped[np.ndarray, "*dims"], *, per_band: bool = True
) -> tuple[int, ...] | None:
    """Return the reduction axes for per-band remote-sensing statistics.

    Per-band statistics keep only the band axis (``-3``) and reduce over
    every other axis: the spatial ``(H, W)`` axes of a ``(C, H, W)`` cube,
    and also the time axis of a ``(T, C, H, W)`` stack, so a stack yields
    one ``(C,)`` statistic per band pooled over all frames (never a
    ``(T, C)`` table). A 2-D map (or a global reduction) uses ``None``.

    Args:
        arr: Input array whose rank decides the reduction mode.
        per_band: If ``True`` (default) and ``arr`` has three or more
            dimensions, reduce over every axis except the band axis. If
            ``False`` always reduce globally.

    Returns:
        ``(-2, -1)`` for a 3-D cube, ``(-4, -2, -1)`` for a 4-D stack
        (every axis but ``-3``), or ``None`` for a global reduction (also
        for 2-D inputs).

    Examples:
        >>> stat_axes(np.zeros((3, 4, 4))), stat_axes(np.zeros((2, 3, 4, 4)))
        ((-2, -1), (-4, -2, -1))
    """
    if per_band and arr.ndim >= 3:
        return tuple(axis for axis in range(-arr.ndim, 0) if axis != BAND_AXIS)
    return None


def reshape_stat(
    stat: Float[np.ndarray, "*stat"] | float,
    arr: Shaped[np.ndarray, "*dims"],
    reduce_axes: tuple[int, ...] | None,
) -> Float[np.ndarray, "*bcast"]:
    """Reshape a reduced statistic so it broadcasts over ``arr``.

    Inverse of the ``reduce_axes`` reduction: a statistic computed with
    ``np.nanmean(arr, axis=axis)`` (and friends) loses the reduced
    axes, so it can't broadcast back against ``arr`` directly when the
    kept axes are leading. This re-inserts singleton dimensions at the
    reduced positions.

    Args:
        stat: Scalar or reduced-statistic array (e.g. per-band means of
            shape ``(C,)`` for a ``(C, H, W)`` input).
        arr: The array the statistic was computed from.
        reduce_axes: The axes that were reduced, or ``None`` for a global /
            scalar statistic.

    Returns:
        A float array broadcastable against ``arr`` (returned unchanged
        when it is scalar or ``reduce_axes`` is ``None``).

    Raises:
        ValueError: If a non-scalar ``stat`` does not have the shape of the
            kept axes (e.g. a 2-entry per-band statistic for a 3-band
            input); it would otherwise broadcast silently along ``x``.
    """
    xp = namespace_of(arr) if hasattr(arr, "dtype") else np
    stat_arr = xp.asarray(stat)
    if stat_arr.dtype != np.float64:  # float64 as before, so float16 cannot overflow
        stat_arr = stat_arr.astype(np.float64)
    if stat_arr.ndim == 0 or reduce_axes is None:
        return stat_arr

    axes = tuple(a % arr.ndim for a in reduce_axes)
    kept_axes = tuple(i for i in range(arr.ndim) if i not in axes)
    kept_shape = tuple(arr.shape[i] for i in kept_axes)
    if stat_arr.size == 1:
        return stat_arr.reshape(())
    if stat_arr.shape != kept_shape:
        raise ValueError(
            f"statistic of shape {stat_arr.shape} does not match the kept axes "
            f"{kept_shape} of an input of shape {arr.shape}; pass one value "
            "per band (or a scalar)"
        )

    shape = [1] * arr.ndim
    for stat_axis, arr_axis in enumerate(kept_axes):
        shape[arr_axis] = stat_arr.shape[stat_axis]
    return stat_arr.reshape(shape)


def per_band_stats(
    arr: Float[np.ndarray, "*dims"],
    *,
    percentiles: list[float] | tuple[float, ...] = (1.0, 99.0),
    reduce_axes: tuple[int, ...] | None = (-2, -1),
) -> dict[str, np.ndarray]:
    """Compute NaN-aware statistics over spatial axes.

    Args:
        arr: Input float array, typically a channel-first ``(C, H, W)``
            cube. NaN pixels are excluded from every statistic.
        percentiles: Percentiles (in ``[0, 100]``) to compute alongside
            the moments. Default ``(1.0, 99.0)``.
        reduce_axes: Axes to reduce over. Default ``(-2, -1)`` yields one value
            per band; ``None`` reduces globally.

    Returns:
        Dict with keys ``"mean"``, ``"std"``, ``"min"``, ``"max"``
        (each shaped like the kept axes, e.g. ``(C,)``) and
        ``"percentiles"`` of shape ``(len(percentiles), C)``.
    """
    return {
        "mean": np.nanmean(arr, axis=reduce_axes),
        "std": np.nanstd(arr, axis=reduce_axes),
        "min": np.nanmin(arr, axis=reduce_axes),
        "max": np.nanmax(arr, axis=reduce_axes),
        "percentiles": np.nanpercentile(arr, percentiles, axis=reduce_axes),
    }


#: The statistics `band_stats` computes by default, in its output order.
BAND_STATS: tuple[str, ...] = ("mean", "std", "min", "max")


def band_stats(
    x: Float[np.ndarray, "*dims"],
    valid: Shaped[np.ndarray, "*dims"] | None = None,
    *,
    band_axis: int | None = BAND_AXIS,
    stats: tuple[str, ...] | list[str] = BAND_STATS,
) -> dict[str, Float[np.ndarray, "*stat"]]:
    r"""Per-band mean, std, min and max of the valid pixels, on any backend.

    .. math::

        \mu_c \;=\; \frac{1}{|V_c|} \sum_{i \in V_c} x_{c,i}

    where :math:`V_c` is the set of valid elements of band :math:`c`, pooled
    over every axis except ``band_axis`` (the frames of a ``(T, C, H, W)``
    stack too), and likewise for ``std``, ``min`` and ``max``. Written on the
    array namespace: NumPy, JAX (under ``jax.jit`` too, one trace per shape)
    and dask, where each statistic is a lazy tree reduction.

    Args:
        x: Float array with the band axis at ``band_axis``.
        valid: Boolean array broadcasting to ``x``; ``False`` leaves an
            element out of every statistic. Non-finite elements are always
            left out; ``None`` keeps every finite element.
        band_axis: The axis kept, one statistic per entry; ``None`` reduces
            globally to scalars.
        stats: Which of ``"mean"``, ``"std"``, ``"min"``, ``"max"`` to compute.

    Returns:
        One array per requested statistic, ``(C,)`` (or scalar for
        ``band_axis=None``), on ``x``'s backend; a band without valid
        elements gives ``NaN``.

    Raises:
        TypeError: ``x`` is a georeader ``GeoTensor``.
        ValueError: An unknown statistic name.

    Examples:
        >>> x = np.array([[[1.0, 3.0]], [[2.0, -9.0]]])  # (C=2, H=1, W=2)
        >>> band_stats(x, x != -9.0, stats=("mean", "max"))
        {'mean': array([2., 2.]), 'max': array([3., 2.])}
        >>> # a (T, C, H, W) stack pools its frames: still one value per band
        >>> band_stats(np.ones((4, 3, 2, 2)), stats=("min",))["min"].shape
        (3,)
        >>> float(band_stats(x, x != -9.0, band_axis=None, stats=("min",))["min"])
        1.0
    """
    unknown = [name for name in stats if name not in BAND_STATS]
    if unknown:
        raise ValueError(f"unknown statistics {unknown}; choose from {BAND_STATS}")
    xp, data, axes = _band_reduction(x, valid, band_axis, "band_stats")
    reducers = {
        "mean": xp.nanmean,
        "std": xp.nanstd,
        "min": xp.nanmin,
        "max": xp.nanmax,
    }
    return {name: reducers[name](data, axis=axes) for name in stats}


def band_quantiles(
    x: Float[np.ndarray, "*dims"],
    valid: Shaped[np.ndarray, "*dims"] | None = None,
    *,
    band_axis: int | None = BAND_AXIS,
    q: tuple[float, ...] | list[float] = (0.25, 0.5, 0.75),
) -> Float[np.ndarray, "levels ..."]:
    r"""Per-band quantiles of the valid pixels, on any backend.

    :math:`q_c(p)` is the :math:`p`-th quantile (linear interpolation, as
    ``np.nanpercentile``) of the valid elements of band :math:`c`, pooled
    over every axis except ``band_axis``.

    On dask an exact quantile needs every pixel of a band in one block, so
    ``dask.array.nanquantile`` rechunks the reduced axes to a single chunk
    and the band axis to ``"auto"``: a task holds whole bands, as many as fit
    dask's ``array.chunk-size`` (at least one), never the whole cube.

    Args:
        x: Float array with the band axis at ``band_axis``.
        valid: Boolean array broadcasting to ``x``; ``False`` leaves an
            element out. Non-finite elements are always left out; ``None``
            keeps every finite element.
        band_axis: The axis kept; ``None`` gives one quantile per level.
        q: Quantile levels in ``[0, 1]``, static (they fix the output shape).

    Returns:
        ``(len(q), C)`` (or ``(len(q),)`` for ``band_axis=None``) on ``x``'s
        backend; a band without valid elements gives ``NaN``.

    Raises:
        TypeError: ``x`` is a georeader ``GeoTensor``.
        ValueError: A level outside ``[0, 1]``.

    Examples:
        >>> x = np.arange(8.0).reshape(2, 2, 2)  # (C=2, H=2, W=2)
        >>> band_quantiles(x, q=(0.0, 0.5, 1.0))
        array([[0. , 4. ],
               [1.5, 5.5],
               [3. , 7. ]])
        >>> # IQR per band, nodata left out
        >>> lo, hi = band_quantiles(x, x != 0.0, q=(0.25, 0.75))
        >>> (hi - lo).tolist()
        [1.0, 1.5]
    """
    levels = np.asarray(q, dtype=float)
    if levels.ndim != 1 or np.any((levels < 0) | (levels > 1)):
        raise ValueError(f"quantile levels must be a sequence in [0, 1]; got {q}")
    xp, data, axes = _band_reduction(x, valid, band_axis, "band_quantiles")
    return xp.nanquantile(data, xp.asarray(levels, dtype=data.dtype), axis=axes)


def _band_reduction(
    x: Any, valid: Any, band_axis: int | None, name: str
) -> tuple[Any, Any, tuple[int, ...]]:
    """Namespace, float data with invalid elements ``NaN``, and the reduced axes."""
    reject_geotensor(x, name)
    xp = namespace_of(x)
    # float64, as the NumPy operators' masked copies, so a float32 cube's
    # statistics do not depend on the backend (JAX needs x64 for this)
    data = x if x.dtype == np.float64 else x.astype(np.float64)
    # the nan* reducers skip NaN only: an infinity is left out as well
    keep = xp.isfinite(data) if valid is None else valid & xp.isfinite(data)
    data = xp.where(keep, data, xp.nan)
    kept = None if band_axis is None else band_axis % data.ndim
    # an explicit tuple, never None: dask.array.nanquantile rejects a global axis
    return xp, data, tuple(axis for axis in range(data.ndim) if axis != kept)


def standard_scale(
    arr: Float[np.ndarray, "*dims"],
    mean: Float[np.ndarray, "*stat"] | float,
    std: Float[np.ndarray, "*stat"] | float,
    *,
    reduce_axes: tuple[int, ...] | None = (-2, -1),
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
        reduce_axes: Axes the statistics were reduced over. Default
            ``(-2, -1)``; ``None`` for global statistics.

    Returns:
        Float array of the same shape as ``arr``.
    """
    mean_b = reshape_stat(mean, arr, reduce_axes)
    std_b = reshape_stat(std, arr, reduce_axes)
    denom = namespace_of(arr).where(std_b != 0, std_b, 1.0)
    return (arr - mean_b) / denom


def robust_scale(
    arr: Float[np.ndarray, "*dims"],
    median: Float[np.ndarray, "*stat"] | float,
    iqr: Float[np.ndarray, "*stat"] | float,
    *,
    reduce_axes: tuple[int, ...] | None = (-2, -1),
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
        reduce_axes: Axes the statistics were reduced over. Default
            ``(-2, -1)``; ``None`` for global statistics.

    Returns:
        Float array of the same shape as ``arr``.
    """
    median_b = reshape_stat(median, arr, reduce_axes)
    iqr_b = reshape_stat(iqr, arr, reduce_axes)
    denom = namespace_of(arr).where(iqr_b != 0, iqr_b, 1.0)
    return (arr - median_b) / denom


def minmax_scale(
    arr: Float[np.ndarray, "*dims"],
    vmin: Float[np.ndarray, "*stat"] | float,
    vmax: Float[np.ndarray, "*stat"] | float,
    *,
    out_range: tuple[float, float] = (0.0, 1.0),
    reduce_axes: tuple[int, ...] | None = (-2, -1),
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
        reduce_axes: Axes the bounds were reduced over. Default ``(-2, -1)``;
            ``None`` for global bounds.

    Returns:
        Float array of the same shape as ``arr``. Values outside
        ``[vmin, vmax]`` are *not* clipped.

    Raises:
        ValueError: If ``out_range`` is not a two-element increasing
            tuple.
    """
    out_min, out_max = validate_out_range(out_range)
    vmin_b = reshape_stat(vmin, arr, reduce_axes)
    vmax_b = reshape_stat(vmax, arr, reduce_axes)
    denom = namespace_of(arr).where(vmax_b > vmin_b, vmax_b - vmin_b, 1.0)
    return (arr - vmin_b) / denom * (out_max - out_min) + out_min


def inverse_standard_scale(
    arr: Float[np.ndarray, "*dims"],
    mean: Float[np.ndarray, "*stat"] | float,
    std: Float[np.ndarray, "*stat"] | float,
    *,
    reduce_axes: tuple[int, ...] | None = (-2, -1),
) -> Float[np.ndarray, "*dims"]:
    r"""Undo :func:`standard_scale`.

    .. math::

        x \;=\; y \cdot \sigma + \mu

    Args:
        arr: Scaled float array. NaN pixels propagate through untouched.
        mean: Scalar or per-band mean :math:`\mu`, as for `standard_scale`.
        std: Scalar or per-band standard deviation :math:`\sigma`; a zero
            ``std`` is read as ``1``, as `standard_scale` divides by it.
        reduce_axes: Axes the statistics were reduced over. Default
            ``(-2, -1)``; ``None`` for global statistics.

    Returns:
        Float array of the same shape as ``arr``.

    Raises:
        TypeError: ``arr`` is a georeader ``GeoTensor``.

    Examples:
        >>> x = np.array([[[1.0, 3.0]]])
        >>> inverse_standard_scale(standard_scale(x, 2.0, 0.5), 2.0, 0.5)
        array([[[1., 3.]]])
        >>> # a constant band (std 0) round-trips too
        >>> inverse_standard_scale(standard_scale(x, 1.0, 0.0), 1.0, 0.0)
        array([[[1., 3.]]])
    """
    reject_geotensor(arr, "inverse_standard_scale")
    mean_b = reshape_stat(mean, arr, reduce_axes)
    std_b = reshape_stat(std, arr, reduce_axes)
    return arr * namespace_of(arr).where(std_b != 0, std_b, 1.0) + mean_b


def inverse_robust_scale(
    arr: Float[np.ndarray, "*dims"],
    median: Float[np.ndarray, "*stat"] | float,
    iqr: Float[np.ndarray, "*stat"] | float,
    *,
    reduce_axes: tuple[int, ...] | None = (-2, -1),
) -> Float[np.ndarray, "*dims"]:
    r"""Undo :func:`robust_scale`.

    .. math::

        x \;=\; y \cdot (Q_{3} - Q_{1}) + \mathrm{median}

    Args:
        arr: Scaled float array. NaN pixels propagate through untouched.
        median: Scalar or per-band median, as for `robust_scale`.
        iqr: Scalar or per-band interquartile range; a zero ``iqr`` is
            read as ``1``.
        reduce_axes: Axes the statistics were reduced over. Default
            ``(-2, -1)``; ``None`` for global statistics.

    Returns:
        Float array of the same shape as ``arr``.

    Raises:
        TypeError: ``arr`` is a georeader ``GeoTensor``.

    Examples:
        >>> x = np.array([[[1.0, 3.0]]])
        >>> inverse_robust_scale(robust_scale(x, 2.0, 4.0), 2.0, 4.0)
        array([[[1., 3.]]])
        >>> # per-band statistics for a (C, H, W) cube
        >>> cube = np.ones((2, 1, 2))
        >>> inverse_robust_scale(cube, [0.0, 1.0], [2.0, 3.0])[:, 0, 0]
        array([2., 4.])
    """
    reject_geotensor(arr, "inverse_robust_scale")
    median_b = reshape_stat(median, arr, reduce_axes)
    iqr_b = reshape_stat(iqr, arr, reduce_axes)
    return arr * namespace_of(arr).where(iqr_b != 0, iqr_b, 1.0) + median_b


def inverse_minmax_scale(
    arr: Float[np.ndarray, "*dims"],
    vmin: Float[np.ndarray, "*stat"] | float,
    vmax: Float[np.ndarray, "*stat"] | float,
    *,
    out_range: tuple[float, float] = (0.0, 1.0),
    reduce_axes: tuple[int, ...] | None = (-2, -1),
) -> Float[np.ndarray, "*dims"]:
    r"""Undo :func:`minmax_scale`: map ``out_range`` back onto ``[vmin, vmax]``.

    .. math::

        x \;=\; \frac{y - o_{\min}}{o_{\max} - o_{\min}}
                \cdot (v_{\max} - v_{\min}) + v_{\min}

    Args:
        arr: Scaled float array. NaN pixels propagate through untouched.
        vmin: Scalar or per-band lower bound, as for `minmax_scale`.
        vmax: Scalar or per-band upper bound; ``vmax <= vmin`` reads the
            span as ``1``.
        out_range: The ``(out_min, out_max)`` range `minmax_scale` mapped
            into. Default ``(0.0, 1.0)``.
        reduce_axes: Axes the bounds were reduced over. Default
            ``(-2, -1)``; ``None`` for global bounds.

    Returns:
        Float array of the same shape as ``arr``.

    Raises:
        TypeError: ``arr`` is a georeader ``GeoTensor``.
        ValueError: If ``out_range`` is not a two-element increasing tuple.

    Examples:
        >>> x = np.array([[[1.0, 3.0]]])
        >>> inverse_minmax_scale(minmax_scale(x, 1.0, 3.0), 1.0, 3.0)
        array([[[1., 3.]]])
        >>> y = minmax_scale(x, 1.0, 3.0, out_range=(0.0, 255.0))
        >>> inverse_minmax_scale(y, 1.0, 3.0, out_range=(0.0, 255.0))
        array([[[1., 3.]]])
    """
    reject_geotensor(arr, "inverse_minmax_scale")
    out_min, out_max = validate_out_range(out_range)
    vmin_b = reshape_stat(vmin, arr, reduce_axes)
    vmax_b = reshape_stat(vmax, arr, reduce_axes)
    span = namespace_of(arr).where(vmax_b > vmin_b, vmax_b - vmin_b, 1.0)
    return (arr - out_min) / (out_max - out_min) * span + vmin_b


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
    window: int | tuple[int, int] | None = None,
    clip_limit: float = 0.01,
    nbins: int = 256,
) -> Float[np.ndarray, "*dims"]:
    """Apply contrast-limited adaptive histogram equalization per image band.

    Wraps :func:`skimage.exposure.equalize_adapthist`. Each ``(H, W)``
    slice -- every band of a ``(C, H, W)`` cube, every band of every
    frame of a ``(T, C, H, W)`` stack -- is equalized independently. Because skimage requires
    inputs in ``[0, 1]``, every slice is rescaled to ``[0, 1]`` over its
    finite range, equalized, then mapped back to its native units. NaN
    pixels are preserved and excluded from the histograms.

    Args:
        arr: Input array; trailing axes are ``(H, W)``.
        window: Contextual-region shape for the local histograms.
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
    out = np.empty_like(values)
    for idx in np.ndindex(values.shape[:-2]):
        out[idx] = _clahe_slice(
            values[idx],
            window=window,
            clip_limit=clip_limit,
            nbins=nbins,
        )
    return out


def _clahe_slice(
    values: np.ndarray,
    *,
    window: int | tuple[int, int] | None,
    clip_limit: float,
    nbins: int,
) -> np.ndarray:
    equalize_adapthist = optional_name(
        "skimage.exposure",
        "equalize_adapthist",
        "morph",
        feature="geoops.normalize",
    )
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
        kernel_size=window,
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
    check_log_scale(base, eps)
    xp = array_namespace(arr)
    return xp.log(xp.maximum(arr, 0.0) + eps) / math.log(base)


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
    check_positive("a", a)
    return array_namespace(arr).asinh(arr / a)


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
    check_positive("gamma", gamma)
    return array_namespace(arr).maximum(arr, 0.0) ** gamma


def check_log_scale(base: float, eps: float) -> None:
    """Raise unless ``base > 0``, ``base != 1`` and ``eps > 0`` (`log_scale`).

    Example:
        >>> check_log_scale(10.0, 1e-6)
    """
    if base <= 0 or base == 1.0:
        raise ValueError(f"base must be positive and not equal to 1; got {base}")
    check_positive("eps", eps)


def check_positive(name: str, value: float) -> None:
    """Raise ``ValueError`` naming ``name`` unless ``value > 0``.

    Example:
        >>> check_positive("gamma", 0.5)
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")


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
