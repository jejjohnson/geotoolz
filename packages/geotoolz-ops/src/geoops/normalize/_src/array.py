"""Normalization primitives, on the array namespace where they can be.

Every primitive here runs on NumPy; the per-band statistics, the scale
kernels, histogram matching and :func:`clahe` also run on JAX (under
``jax.jit`` too) and on dask. None needs scikit-image: :func:`clahe`
reproduces :func:`skimage.exposure.equalize_adapthist`, which the tests
keep as an oracle.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import Float, Shaped
from pipekit_array import array_namespace

from geoops._src.geo import reject_geotensor
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

    Examples:
        >>> x = np.array([[[1.0, 3.0]]])
        >>> inverse_standard_scale(standard_scale(x, 2.0, 0.5), 2.0, 0.5)
        array([[[1., 3.]]])
        >>> # a constant band (std 0) round-trips too
        >>> inverse_standard_scale(standard_scale(x, 1.0, 0.0), 1.0, 0.0)
        array([[[1., 3.]]])
    """
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

    Examples:
        >>> x = np.array([[[1.0, 3.0]]])
        >>> inverse_robust_scale(robust_scale(x, 2.0, 4.0), 2.0, 4.0)
        array([[[1., 3.]]])
        >>> # per-band statistics for a (C, H, W) cube
        >>> cube = np.ones((2, 1, 2))
        >>> inverse_robust_scale(cube, [0.0, 1.0], [2.0, 3.0])[:, 0, 0]
        array([2., 4.])
    """
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
        ValueError: If ``out_range`` is not a two-element increasing tuple.

    Examples:
        >>> x = np.array([[[1.0, 3.0]]])
        >>> inverse_minmax_scale(minmax_scale(x, 1.0, 3.0), 1.0, 3.0)
        array([[[1., 3.]]])
        >>> y = minmax_scale(x, 1.0, 3.0, out_range=(0.0, 255.0))
        >>> inverse_minmax_scale(y, 1.0, 3.0, out_range=(0.0, 255.0))
        array([[[1., 3.]]])
    """
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
    reference value at the same empirical quantile (`reference_cdf` then
    `match_cdf`). NaN pixels in the source are left untouched; NaNs in the
    reference are excluded from its CDF.

    Args:
        source: Array to remap. If 3-D+ (``(C, H, W)``), bands are
            matched independently.
        reference: Reference array. Either band-matched (same leading
            band count as ``source``) or a single 2-D map applied to
            every source band.

    Returns:
        Float array of the same shape as ``source`` with its per-band
        histograms reshaped to the reference.

    Example:
        >>> histogram_match(np.array([[0.0, 1.0, 2.0]]), np.array([[10.0, 20.0, 30.0]]))
        array([[10., 20., 30.]])
    """
    source = np.asarray(source, dtype=float)
    reference = np.asarray(reference, dtype=float)
    banded = source.ndim >= 3 and (
        reference.ndim == 2
        or (reference.ndim >= 3 and source.shape[0] == reference.shape[0])
    )
    n_bands = source.shape[0] if banded else 1
    flat = source.reshape(n_bands, -1)
    ref = reference.reshape(1 if reference.ndim == 2 or not banded else n_bands, -1)
    table = reference_cdf(ref, np.isfinite(ref))
    if table[0].shape[0] != n_bands:  # one 2-D map for every band
        table = tuple(np.repeat(t, n_bands, axis=0) for t in table)
    return match_cdf(flat, np.isfinite(flat), *table).reshape(source.shape)


def reference_cdf(
    reference: Float[np.ndarray, "C N"], valid: Shaped[np.ndarray, "C N"]
) -> tuple[
    Float[np.ndarray, "C M"], Float[np.ndarray, "C M"], Shaped[np.ndarray, "C 1"]
]:
    r"""The per-band empirical CDF of ``reference``, as a padded table for `match_cdf`.

    For each band, the distinct valid values :math:`v_1 < \dots < v_m` and
    their cumulative fractions :math:`G(v_k) = \#\{x \le v_k\} / n`, the
    points that scikit-image's ``match_histograms`` interpolates through.
    Bands have different numbers of distinct values, so each row is padded to
    the longest with levels above ``1`` and its maximum, which an
    interpolation at a fraction in ``[0, 1]`` never reaches. Host NumPy only:
    a reference is concrete, and its table is a constant for `match_cdf`.

    Args:
        reference: ``(C, N)`` values, one band per row.
        valid: ``(C, N)`` booleans; ``False`` leaves a value out.

    Returns:
        ``(levels, values, ok)``: ``(C, M)`` fractions, ``(C, M)`` values, and
        ``(C, 1)`` whether the band has any valid value.

    Example:
        >>> levels, values, ok = reference_cdf(np.array([[1.0, 1.0, 3.0]]), None)
        >>> levels.tolist(), values.tolist(), ok.tolist()
        ([[0.6666666666666666, 1.0]], [[1.0, 3.0]], [[True]])
    """
    reference = np.asarray(reference, dtype=float)
    if valid is None:
        valid = np.isfinite(reference)
    rows = []
    for band, keep in zip(reference, np.asarray(valid, dtype=bool), strict=True):
        values, counts = np.unique(band[keep], return_counts=True)
        rows.append((np.cumsum(counts) / max(int(keep.sum()), 1), values))
    width = max(1, *(values.size for _, values in rows))
    levels = np.empty((len(rows), width))
    table = np.empty((len(rows), width))
    for row, (fractions, values) in enumerate(rows):
        m = values.size
        levels[row, :m] = fractions
        levels[row, m:] = 2.0 + np.arange(width - m)
        table[row, :m] = values
        table[row, m:] = values[-1] if m else np.nan
    ok = np.array([[values.size > 0] for _, values in rows], dtype=bool)
    return levels, table, ok


def match_cdf(
    x: Float[np.ndarray, "C N"],
    valid: Shaped[np.ndarray, "C N"],
    levels: Float[np.ndarray, "C M"],
    values: Float[np.ndarray, "C M"],
    ok: Shaped[np.ndarray, "C 1"],
) -> Float[np.ndarray, "C N"]:
    r"""Map each band of ``x`` through its CDF onto a `reference_cdf` table.

    .. math::

        y \;=\; G_c^{-1}\big(F_c(x)\big), \qquad
        F_c(x) \;=\; \frac{\#\{\text{valid } x' \le x\}}{n_c}

    :math:`F_c` comes from a sort and ``searchsorted`` (invalid values sort
    last as ``+inf``), and :math:`G_c^{-1}` is a linear interpolation through
    the table, so this is exact rank matching with static shapes: it runs on
    NumPy and JAX, under ``jax.jit`` too (one trace per shape), and on a
    dask block holding whole bands.

    Args:
        x: ``(C, N)`` float values, one band per row.
        valid: ``(C, N)`` booleans; ``False`` leaves a value out of the CDF.
        levels: ``(C, M)`` fractions from `reference_cdf`.
        values: ``(C, M)`` values from `reference_cdf`.
        ok: ``(C, 1)`` from `reference_cdf`; a band whose reference has no
            valid value keeps ``x``.

    Returns:
        ``(C, N)`` on ``x``'s backend; invalid entries keep ``x``'s value.

    Example:
        >>> table = reference_cdf(np.array([[10.0, 20.0, 30.0]]), None)
        >>> x = np.array([[2.0, 0.0, 1.0]])
        >>> match_cdf(x, np.isfinite(x), *table)
        array([[30., 10., 20.]])
    """
    xp = namespace_of(x)
    count = xp.sum(valid, axis=-1, keepdims=True)
    ordered = xp.sort(xp.where(valid, x, xp.inf), axis=-1)
    rows = []
    for band in range(x.shape[0]):
        rank = xp.searchsorted(ordered[band], x[band], side="right")
        fraction = rank / xp.maximum(count[band], 1)
        rows.append(xp.interp(fraction, levels[band], values[band]))
    matched = xp.stack(rows)
    return xp.where(valid & (count > 0) & ok, matched, x)


#: scikit-image's grey levels for CLAHE (14-bit), which `clahe` reproduces.
NR_OF_GRAY = 2**14


def clahe(
    arr: Shaped[np.ndarray, "*dims"],
    *,
    window: int | tuple[int, int] | None = None,
    clip_limit: float = 0.01,
    nbins: int = 256,
) -> Float[np.ndarray, "*dims"]:
    """Apply contrast-limited adaptive histogram equalization per image band.

    The algorithm of :func:`skimage.exposure.equalize_adapthist`, written on
    the array namespace so it runs on NumPy and on JAX (under ``jax.jit`` too,
    one trace per shape) without scikit-image. Each ``(H, W)`` slice (every
    band of a ``(C, H, W)`` cube, every band of every frame of a
    ``(T, C, H, W)`` stack) is equalized independently:

    1. rescale to ``[0, 1]`` over its finite range, quantise to 14-bit grey
       levels and bin them into ``nbins``;
    2. reflect-pad to whole tiles of ``window``, take each tile's histogram,
       clip it at ``clip_limit`` times the tile size and redistribute the
       excess as scikit-image does;
    3. map each pixel through the cumulative histograms of its four nearest
       tiles, blended bilinearly, and map the result back to native units.

    NaN pixels are excluded (filled with the slice median before the
    histograms, as before) and stay NaN. A constant or all-NaN slice is
    returned unchanged.

    Args:
        arr: Input array; trailing axes are ``(H, W)``.
        window: Contextual-region shape for the local histograms.
            ``None`` uses scikit-image's default (1/8 of the slice height /
            width).
        clip_limit: Contrast-limiting clip threshold in ``[0, 1]``.
            Default ``0.01``.
        nbins: Number of histogram bins. Default ``256``.

    Returns:
        Float array of the same shape, equalized per band.

    Examples:
        >>> ramp = np.linspace(0.0, 1.0, 64).reshape(8, 8) ** 2
        >>> out = clahe(ramp, window=4)
        >>> out.shape, float(out.min()), float(out.max())
        ((8, 8), 0.0, 1.0)
        >>> # a stack is equalized slice by slice
        >>> clahe(np.stack([ramp, ramp]), window=4).shape
        (2, 8, 8)
    """
    xp = namespace_of(arr)
    values = arr if arr.dtype == np.float64 else arr.astype(np.float64)
    shape = values.shape
    height, width = shape[-2:]
    kernel = _clahe_kernel(window, (height, width))
    v = xp.reshape(values, (-1, height, width))
    valid = xp.isfinite(v)
    vmin = xp.min(xp.where(valid, v, xp.inf), axis=(-2, -1), keepdims=True)
    vmax = xp.max(xp.where(valid, v, -xp.inf), axis=(-2, -1), keepdims=True)
    ok = vmax > vmin  # False for a constant or all-NaN slice
    span = xp.where(ok, vmax - vmin, 1.0)
    low = xp.where(ok, vmin, 0.0)
    median = xp.nanmedian(xp.where(ok & valid, v, xp.nan), axis=(-2, -1), keepdims=True)
    unit = xp.where(valid, (v - low) / span, (median - low) / span)
    unit = xp.where(ok, xp.clip(unit, 0.0, 1.0), 0.0)
    equalized = _equalize_adapthist(unit, kernel, clip_limit, nbins, xp)
    out = xp.where(valid & ok, equalized * span + low, v)
    return xp.reshape(out, shape)


def _clahe_kernel(
    window: int | tuple[int, int] | None, shape: tuple[int, int]
) -> tuple[int, int]:
    """scikit-image's ``kernel_size`` for a ``(H, W)`` slice."""
    if window is None:
        return (max(shape[0] // 8, 1), max(shape[1] // 8, 1))
    if isinstance(window, int | np.integer):
        return (int(window), int(window))
    if len(window) != 2:
        raise ValueError(f"window must be an int or (h, w); got {window}")
    return (int(window[0]), int(window[1]))


def _equalize_adapthist(
    unit: Any, kernel: tuple[int, int], clip_limit: float, nbins: int, xp: Any
) -> Any:
    """``equalize_adapthist`` on ``(L, H, W)`` slices in ``[0, 1]``."""
    # img_as_uint, then rescale_intensity onto the 14-bit grey levels
    as_uint = xp.round(unit * 65535.0)
    lo = xp.min(as_uint, axis=(-2, -1), keepdims=True)
    hi = xp.max(as_uint, axis=(-2, -1), keepdims=True)
    grey = (as_uint - lo) / xp.where(hi > lo, hi - lo, 1.0) * (NR_OF_GRAY - 1)
    grey = xp.round(grey).astype(xp.int64)
    result = _clahe_grey(grey, kernel, clip_limit, nbins, xp)
    lo = xp.min(result, axis=(-2, -1), keepdims=True)
    hi = xp.max(result, axis=(-2, -1), keepdims=True)
    return (result - lo) / xp.where(hi > lo, hi - lo, 1.0)


def _clahe_grey(
    grey: Any, kernel: tuple[int, int], clip_limit: float, nbins: int, xp: Any
) -> Any:
    """scikit-image's ``_clahe`` on ``(L, H, W)`` 14-bit grey levels, 2-D tiles."""
    n_slices, height, width = grey.shape
    (k0, k1) = kernel
    pad_start = (k0 // 2, k1 // 2)
    pad_end = tuple(
        (k - s % k) % k + math.ceil(k / 2)
        for k, s in zip(kernel, (height, width), strict=True)
    )
    image = xp.pad(
        grey,
        ((0, 0), (pad_start[0], pad_end[0]), (pad_start[1], pad_end[1])),
        mode="reflect",
    )
    image = image // (1 + NR_OF_GRAY // nbins)
    rows, cols = image.shape[-2:]

    # one histogram per interior tile, offset by half a tile
    n0, n1 = rows // k0 - 1, cols // k1 - 1
    tiles = image[:, k0 // 2 : k0 // 2 + n0 * k0, k1 // 2 : k1 // 2 + n1 * k1]
    tiles = xp.reshape(tiles, (n_slices, n0, k0, n1, k1))
    tiles = xp.reshape(xp.transpose(tiles, (0, 1, 3, 2, 4)), (-1, k0 * k1))
    elements = k0 * k1
    clim = int(max(clip_limit * elements, 1)) if clip_limit > 0.0 else elements
    hist = _clip_histogram(_bincount_rows(tiles, nbins, xp), clim, xp)
    # map_histogram: the scaled cumulative histogram, truncated to integers
    cdf = xp.cumsum(hist, axis=-1).astype(xp.float64) * ((NR_OF_GRAY - 1) / elements)
    cdf = xp.minimum(cdf, NR_OF_GRAY - 1).astype(xp.int64)
    cdf = xp.reshape(cdf, (n_slices, n0, n1, nbins))
    maps = xp.pad(cdf, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")

    # every pixel through its four nearest tiles' maps, blended bilinearly
    p0, p1 = rows // k0, cols // k1
    blocks = xp.reshape(image, (n_slices, p0, k0, p1, k1))
    blocks = xp.reshape(
        xp.transpose(blocks, (0, 1, 3, 2, 4)), (n_slices, p0 * p1, k0 * k1)
    )
    row_frac = xp.reshape(xp.broadcast_to((np.arange(k0) / k0)[:, None], kernel), (-1,))
    col_frac = xp.reshape(xp.broadcast_to((np.arange(k1) / k1)[None, :], kernel), (-1,))
    result = xp.zeros(blocks.shape, dtype=xp.float32)
    for e0 in (0, 1):
        for e1 in (0, 1):
            edge = xp.reshape(
                maps[:, e0 : e0 + p0, e1 : e1 + p1], (n_slices, p0 * p1, nbins)
            )
            mapped = xp.take_along_axis(edge, blocks, axis=-1)
            weight = (col_frac if e1 else 1 - col_frac) * (
                row_frac if e0 else 1 - row_frac
            )
            result = result + (mapped * xp.asarray(weight)).astype(xp.float32)
    result = result.astype(xp.int64).astype(xp.float64)  # skimage casts back to uint16
    result = xp.reshape(result, (n_slices, p0, p1, k0, k1))
    result = xp.reshape(xp.transpose(result, (0, 1, 3, 2, 4)), (n_slices, rows, cols))
    return result[:, pad_start[0] : rows - pad_end[0], pad_start[1] : cols - pad_end[1]]


def _bincount_rows(idx: Any, nbins: int, xp: Any) -> Any:
    """``(R, nbins)`` counts of each row of integer ``idx`` in ``[0, nbins)``."""
    n_rows = idx.shape[0]
    flat = xp.reshape(idx + xp.arange(n_rows)[:, None] * nbins, (-1,))
    if xp.__name__ == "numpy":
        counts = np.bincount(flat, minlength=n_rows * nbins)
    else:  # jax.numpy: a static length keeps the shape known under jit
        counts = xp.bincount(flat, length=n_rows * nbins)
    return xp.reshape(counts, (n_rows, nbins))


def _clip_histogram(hist: Any, clim: int, xp: Any) -> Any:
    """scikit-image's ``clip_histogram`` on every row of ``hist`` at once."""
    nbins = hist.shape[-1]
    excess = xp.sum(xp.where(hist > clim, hist - clim, 0), axis=-1, keepdims=True)
    hist = xp.minimum(hist, clim)
    bin_incr = excess // nbins  # the average bin increment
    upper = clim - bin_incr  # bins above this go straight to the limit
    low = hist < upper
    excess = excess - xp.sum(low, axis=-1, keepdims=True) * bin_incr
    hist = xp.where(low, hist + bin_incr, hist)
    mid = (hist >= upper) & (hist < clim)
    excess = excess + xp.sum(xp.where(mid, hist - clim, 0), axis=-1, keepdims=True)
    hist = xp.where(mid, clim, hist)
    return _redistribute(hist, excess, clim, xp)


def _redistribute_step(
    index: Any, hist: Any, excess: Any, alive: Any, clim: int, xp: Any
) -> tuple[Any, Any]:
    """One ``index`` of scikit-image's loop that spreads the remaining excess."""
    active = alive & (excess > 0)
    under = hist < clim
    n_under = xp.sum(under, axis=-1, keepdims=True)
    step = xp.maximum(1, n_under // xp.where(active, excess, 1))
    position = xp.arange(hist.shape[-1])
    picked = (position >= index) & ((position - index) % step == 0) & under & active
    return hist + picked, excess - xp.sum(picked, axis=-1, keepdims=True)


def _redistribute(hist: Any, excess: Any, clim: int, xp: Any) -> Any:
    """Spread the excess left after clipping, until none is left or none moves."""
    nbins = hist.shape[-1]
    alive = excess > 0
    if xp.__name__ != "numpy":  # jax.numpy: the same loops as lax primitives
        from jax import lax

        def one_pass(state: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
            hist, excess, alive = state

            def body(index: Any, carry: tuple[Any, Any]) -> tuple[Any, Any]:
                return _redistribute_step(index, *carry, alive, clim, xp)

            new_hist, new_excess = lax.fori_loop(0, nbins, body, (hist, excess))
            return (
                new_hist,
                new_excess,
                alive & (new_excess > 0) & (new_excess != excess),
            )

        hist, _, _ = lax.while_loop(
            lambda state: xp.any(state[2]), one_pass, (hist, excess, alive)
        )
        return hist
    while alive.any():
        previous = excess
        for index in range(nbins):
            hist, excess = _redistribute_step(index, hist, excess, alive, clim, xp)
            if not (alive & (excess > 0)).any():
                break
        alive = alive & (excess > 0) & (excess != previous)
    return hist


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
