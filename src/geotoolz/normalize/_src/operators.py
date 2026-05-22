"""Tier-B Operators — carrier-aware normalization transforms.

Each Operator wraps a Tier-A primitive in
:mod:`geotoolz.normalize._src.array`. The carrier-aware wrappers handle:

* fitting per-band statistics from the input scene
  (``fit_on_call=True``),
* JSON-safe ``get_config()`` via the shared
  :func:`~geotoolz.radiometry._src.operators._coef_as_jsonable` helper
  (ndarray leaves become plain Python lists for Hydra / YAML
  round-trip),
* NaN-aware reductions over the spatial ``(H, W)`` axes.

The display-prep min-max stretch with **scalar** bounds lives in
:class:`geotoolz.radiometry.MinMax`; the per-scene robust percentile
stretch lives in :class:`geotoolz.radiometry.PercentileClip`. This
module is the per-band normaliser shop used by ML pipelines:
``StandardScaler``, ``RobustScaler``, ``MinMaxScaler``, and the
fixed-stats alias ``Normalize``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pipekit import Operator

from geotoolz.normalize._src.array import (
    asinh_scale,
    clahe,
    histogram_match,
    log_scale,
    minmax_scale,
    per_band_stats,
    percentile_clip,
    power_scale,
    robust_scale,
    standard_scale,
    stat_axes,
    validate_out_range,
)
from geotoolz.radiometry._src.operators import _coef_as_jsonable


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


def _stat_as_jsonable(value: Any) -> float | list[Any] | None:
    """JSON-safe coercion for cached statistics.

    Mirrors :func:`_coef_as_jsonable` (Python ``float`` for scalars,
    ``list[float]`` for 1-D) and additionally:

    * passes ``None`` through unchanged (stats may be unset before fit),
    * preserves shape for ``ndim > 1`` via ``ndarray.tolist()`` (the
      ``PerBandStats`` cache stores ``percentiles`` as a 2-D array of
      shape ``(n_percentiles, n_bands)``).
    """
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return _coef_as_jsonable(arr)
    return arr.tolist()


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


class PerBandStats(Operator):
    """Compute NaN-aware per-band statistics and cache them.

    Side-effect-only operator: returns the input unchanged after caching
    a ``stats`` dict containing per-band ``mean``, ``std``, ``min``,
    ``max``, and ``percentiles`` over the spatial ``(H, W)`` axes. Use
    this in a ``Tap``-style stage to inspect a scene before applying a
    downstream normaliser.

    Args:
        percentiles: Percentiles (in ``[0, 100]``) to cache alongside
            mean/std/min/max. Default ``[1.0, 99.0]``.

    Examples:
        >>> from geotoolz.normalize import PerBandStats
        >>> op = PerBandStats(percentiles=[2.0, 98.0])
        >>> _ = op(scene)
        >>> op.stats["mean"]   # one entry per band
        [...]
    """

    def __init__(self, *, percentiles: list[float] | None = None) -> None:
        self.percentiles = [1.0, 99.0] if percentiles is None else list(percentiles)
        self.stats: dict[str, Any] = {}

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        stats = per_band_stats(arr, percentiles=self.percentiles, axis=stat_axes(arr))
        self.stats = {key: _stat_as_jsonable(value) for key, value in stats.items()}
        return gt

    def get_config(self) -> dict[str, Any]:
        return {"percentiles": self.percentiles}


class CLAHE(Operator):
    """Contrast-Limited Adaptive Histogram Equalization.

    Wraps :func:`skimage.exposure.equalize_adapthist` and applies it
    independently per band for ``(C, H, W)`` carriers while preserving
    NaN positions and GeoTensor metadata.
    """

    def __init__(
        self,
        *,
        kernel_size: int | tuple[int, int] | None = None,
        clip_limit: float = 0.01,
        nbins: int = 256,
    ) -> None:
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.nbins = nbins

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = clahe(
            np.asarray(gt),
            kernel_size=self.kernel_size,
            clip_limit=self.clip_limit,
            nbins=self.nbins,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "kernel_size": self.kernel_size,
            "clip_limit": self.clip_limit,
            "nbins": self.nbins,
        }


class StandardScaler(Operator):
    r"""Per-band z-score normalisation.

    .. math::

        y \;=\; \frac{x - \mu}{\sigma}

    Statistics reduce over the spatial axes ``(-2, -1)`` so a
    ``(C, H, W)`` carrier yields per-band ``mu`` / ``sigma`` of shape
    ``(C,)``. Pass cached training-set statistics via ``mean`` / ``std``
    for inference, or set ``fit_on_call=True`` to fit on the first
    scene seen. When ``sigma == 0`` (a constant band) the divisor falls
    back to ``1`` so the band collapses to zero instead of producing
    ``inf`` / ``nan``.

    Args:
        mean: Per-band mean (scalar, list, or ndarray) or ``None``.
        std: Per-band std (scalar, list, or ndarray) or ``None``.
        fit_on_call: If ``True`` and ``mean`` / ``std`` are unset, fit
            them from the first call using ``np.nanmean`` /
            ``np.nanstd``.

    Examples:
        >>> from geotoolz.normalize import StandardScaler
        >>> # Inference: cached per-band statistics from training.
        >>> scaler = StandardScaler(mean=[0.1, 0.2], std=[0.05, 0.07])
        >>> normed = scaler(scene)
        >>>
        >>> # Training-time fit-and-apply on a single scene:
        >>> fit = StandardScaler(fit_on_call=True)
        >>> _ = fit(train_scene)
        >>> fit.mean  # cached per-band ndarray
    """

    def __init__(
        self,
        *,
        mean: np.ndarray | list[float] | float | None = None,
        std: np.ndarray | list[float] | float | None = None,
        fit_on_call: bool = False,
    ) -> None:
        self.mean = _array_or_none(mean)
        self.std = _array_or_none(std)
        self.fit_on_call = fit_on_call

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        axis = stat_axes(arr)
        if self.fit_on_call and (self.mean is None or self.std is None):
            self.mean = np.nanmean(arr, axis=axis)
            self.std = np.nanstd(arr, axis=axis)
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler requires mean/std or fit_on_call=True")
        return gt.array_as_geotensor(
            standard_scale(arr, self.mean, self.std, axis=axis)
        )

    def inverse(self, gt: GeoTensor) -> GeoTensor:
        """Invert a previously applied standard scaling."""
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler must be fitted before inverse()")
        arr = np.asarray(gt, dtype=float)
        axis = stat_axes(arr)
        from geotoolz.normalize._src.array import reshape_stat

        mean = reshape_stat(self.mean, arr, axis)
        std = reshape_stat(self.std, arr, axis)
        scale = np.where(std != 0, std, 1.0)
        return gt.array_as_geotensor(arr * scale + mean)

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": _stat_as_jsonable(self.mean),
            "std": _stat_as_jsonable(self.std),
            "fit_on_call": self.fit_on_call,
        }


class RobustScaler(Operator):
    r"""Per-band median / IQR scaling (robust to outliers).

    .. math::

        y \;=\; \frac{x - \mathrm{median}}{Q_{3} - Q_{1}}

    Identical role to :class:`StandardScaler` but uses the median and
    interquartile range instead of mean and std — robust against
    bright-pixel outliers (cumulus, glint, saturation). When
    ``iqr == 0`` the divisor falls back to ``1``.

    Args:
        median: Per-band median (scalar, list, or ndarray) or ``None``.
        iqr: Per-band IQR (``Q3 - Q1``).
        fit_on_call: If ``True`` and stats are unset, fit from the
            first call using ``np.nanpercentile`` at ``[25, 50, 75]``.

    Examples:
        >>> from geotoolz.normalize import RobustScaler
        >>> op = RobustScaler(fit_on_call=True)
        >>> _ = op(scene)
        >>> op.median, op.iqr
    """

    def __init__(
        self,
        *,
        median: np.ndarray | list[float] | float | None = None,
        iqr: np.ndarray | list[float] | float | None = None,
        fit_on_call: bool = False,
    ) -> None:
        self.median = _array_or_none(median)
        self.iqr = _array_or_none(iqr)
        self.fit_on_call = fit_on_call

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        axis = stat_axes(arr)
        if self.fit_on_call and (self.median is None or self.iqr is None):
            q25, q50, q75 = np.nanpercentile(arr, [25.0, 50.0, 75.0], axis=axis)
            self.median = q50
            self.iqr = q75 - q25
        if self.median is None or self.iqr is None:
            raise ValueError("RobustScaler requires median/iqr or fit_on_call=True")
        return gt.array_as_geotensor(
            robust_scale(arr, self.median, self.iqr, axis=axis)
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "median": _stat_as_jsonable(self.median),
            "iqr": _stat_as_jsonable(self.iqr),
            "fit_on_call": self.fit_on_call,
        }


class MinMaxScaler(Operator):
    r"""Per-band min-max scaling into an arbitrary output range.

    .. math::

        y \;=\; \frac{x - v_{\min}}{v_{\max} - v_{\min}}
                \cdot (o_{\max} - o_{\min}) + o_{\min}

    Per-band variant of :class:`geotoolz.radiometry.MinMax`. The
    radiometry version takes **scalar** bounds and is the display-prep
    pick when you already know fixed reflectance limits; this version
    fits **per-band** bounds (one ``vmin`` / ``vmax`` per channel) or
    accepts cached training-time bounds for inference. When
    ``vmax == vmin`` for a band the divisor falls back to ``1``.

    Args:
        vmin: Per-band lower bound, scalar / list / ndarray, or
            ``None`` to fit.
        vmax: Per-band upper bound.
        out_range: ``(out_min, out_max)`` range to map into. Default
            ``(0.0, 1.0)``.
        fit_on_call: If ``True`` and bounds are unset, fit from the
            first call using ``np.nanmin`` / ``np.nanmax``.

    Examples:
        >>> from geotoolz.normalize import MinMaxScaler
        >>> # Map per-band [vmin, vmax] into [0, 1]:
        >>> op = MinMaxScaler(vmin=[0.0, 0.0], vmax=[0.3, 0.4])
        >>> scaled = op(scene)
        >>>
        >>> # Or fit on the scene and emit a uint8-style range:
        >>> op = MinMaxScaler(fit_on_call=True, out_range=(0.0, 255.0))
        >>> _ = op(scene)
    """

    def __init__(
        self,
        *,
        vmin: np.ndarray | list[float] | float | None = None,
        vmax: np.ndarray | list[float] | float | None = None,
        out_range: tuple[float, float] = (0.0, 1.0),
        fit_on_call: bool = False,
    ) -> None:
        self.vmin = _array_or_none(vmin)
        self.vmax = _array_or_none(vmax)
        self.out_range = validate_out_range(tuple(out_range))
        self.fit_on_call = fit_on_call

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        axis = stat_axes(arr)
        if self.fit_on_call and (self.vmin is None or self.vmax is None):
            self.vmin = np.nanmin(arr, axis=axis)
            self.vmax = np.nanmax(arr, axis=axis)
        if self.vmin is None or self.vmax is None:
            raise ValueError("MinMaxScaler requires vmin/vmax or fit_on_call=True")
        out = minmax_scale(
            arr, self.vmin, self.vmax, out_range=self.out_range, axis=axis
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "vmin": _stat_as_jsonable(self.vmin),
            "vmax": _stat_as_jsonable(self.vmax),
            "out_range": list(self.out_range),
            "fit_on_call": self.fit_on_call,
        }


class HistogramStretch(Operator):
    r"""Per-band percentile stretch into ``out_range`` for visualisation.

    Lighter-weight cousin of :class:`geotoolz.radiometry.PercentileClip`:
    clips to ``[P_lower, P_upper]`` using NaN-aware percentiles, then
    maps the result into ``out_range`` instead of fixed ``[0, 1]``.

    Args:
        out_range: Two-element increasing tuple. Default
            ``(0.0, 1.0)``.
        lower: Lower percentile. Default ``2.0``.
        upper: Upper percentile. Default ``98.0``.

    Examples:
        >>> from geotoolz.normalize import HistogramStretch
        >>> # Standard "satellite RGB to display byte range" stretch.
        >>> op = HistogramStretch(out_range=(0.0, 255.0))
        >>> display = op(reflectance_scene)
    """

    def __init__(
        self,
        *,
        out_range: tuple[float, float] = (0.0, 1.0),
        lower: float = 2.0,
        upper: float = 98.0,
    ) -> None:
        self.out_range = validate_out_range(tuple(out_range))
        self.lower = lower
        self.upper = upper

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        clipped = percentile_clip(
            arr, lower=self.lower, upper=self.upper, axis=stat_axes(arr)
        )
        out_min, out_max = self.out_range
        return gt.array_as_geotensor(clipped * (out_max - out_min) + out_min)

    def get_config(self) -> dict[str, Any]:
        return {
            "out_range": list(self.out_range),
            "lower": self.lower,
            "upper": self.upper,
        }


class HistogramMatch(Operator):
    """Match a GeoTensor histogram to a reference GeoTensor.

    Reshapes the per-band empirical CDF of the input to match the
    reference. Useful for visual harmonisation across scenes acquired
    under different illumination.

    The reference is a live GeoTensor — not JSON / YAML serialisable —
    so ``forbid_in_yaml = True``.

    Args:
        reference: Reference GeoTensor whose per-band CDF the input
            will be matched against.

    Examples:
        >>> from geotoolz.normalize import HistogramMatch
        >>> op = HistogramMatch(reference=reference_scene)
        >>> matched = op(source_scene)
    """

    # Holds a live GeoTensor reference, which is not JSON/YAML
    # serialisable.
    forbid_in_yaml = True

    def __init__(self, *, reference: GeoTensor) -> None:
        self.reference = reference

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = histogram_match(
            np.asarray(gt, dtype=float),
            np.asarray(self.reference, dtype=float),
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"reference_shape": list(self.reference.shape)}


class LogScale(Operator):
    r"""Logarithmic scaling with an epsilon offset for zeros.

    .. math::

        y \;=\; \log_{\text{base}}(\max(x, 0) + \epsilon)

    Compresses heavy-tailed distributions (radar backscatter, fire
    radiative power, etc.) before downstream training. ``eps`` keeps
    the log finite at zero.

    Args:
        base: Logarithm base. Must be ``> 0`` and ``!= 1``. Default
            ``10.0``.
        eps: Small positive offset added before the log. Default
            ``1e-6``.

    Examples:
        >>> from geotoolz.normalize import LogScale
        >>> log_sar = LogScale(base=10.0, eps=1e-6)(sar_backscatter)
    """

    def __init__(self, *, base: float = 10.0, eps: float = 1e-6) -> None:
        self.base = base
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            log_scale(np.asarray(gt, dtype=float), base=self.base, eps=self.eps)
        )

    def get_config(self) -> dict[str, Any]:
        return {"base": self.base, "eps": self.eps}


class AsinhScale(Operator):
    r"""Inverse-hyperbolic-sine scaling.

    .. math::

        y \;=\; \mathrm{asinh}(x / a)

    Linear near zero, logarithmic for ``|x| >> a``. Symmetric and
    well-defined for negative values (unlike ``log``) — a common pick
    for astronomy and signed radar quantities.

    Args:
        a: Scale parameter. Must be strictly positive. Default
            ``1.0``.

    Examples:
        >>> from geotoolz.normalize import AsinhScale
        >>> out = AsinhScale(a=0.1)(scene)
    """

    def __init__(self, *, a: float = 1.0) -> None:
        self.a = a

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(asinh_scale(np.asarray(gt, dtype=float), a=self.a))

    def get_config(self) -> dict[str, Any]:
        return {"a": self.a}


class PowerScale(Operator):
    r"""Non-negative power scaling.

    .. math::

        y \;=\; \max(x, 0)^{\gamma}

    A simple ``gamma``-style brightness curve. ``gamma < 1`` brightens
    midtones; ``gamma > 1`` darkens. Differs from
    :class:`geotoolz.radiometry.Gamma` only in convention
    (``Gamma`` raises to ``1/g``).

    Args:
        gamma: Power exponent. Must be strictly positive. Default
            ``0.5`` (square-root).

    Examples:
        >>> from geotoolz.normalize import PowerScale
        >>> out = PowerScale(gamma=0.5)(scene)
    """

    def __init__(self, *, gamma: float = 0.5) -> None:
        self.gamma = gamma

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            power_scale(np.asarray(gt, dtype=float), gamma=self.gamma)
        )

    def get_config(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class Normalize(StandardScaler):
    r"""Fixed-stats per-band z-score normaliser.

    Convenience alias for :class:`StandardScaler` with mandatory
    ``mean`` / ``std`` (no ``fit_on_call`` knob exposed). The canonical
    inference-time normaliser: cache stats once at training time,
    instantiate at inference.

    Args:
        mean: Per-band mean (scalar, list, or ndarray).
        std: Per-band std (scalar, list, or ndarray).

    Examples:
        >>> from geotoolz.normalize import Normalize
        >>> import numpy as np
        >>> op = Normalize(
        ...     mean=np.array([0.1, 0.2, 0.15]),
        ...     std=np.array([0.05, 0.07, 0.06]),
        ... )
        >>> normed = op(scene)
    """

    def __init__(
        self,
        *,
        mean: np.ndarray | list[float] | float,
        std: np.ndarray | list[float] | float,
    ) -> None:
        super().__init__(mean=mean, std=std, fit_on_call=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": _stat_as_jsonable(self.mean),
            "std": _stat_as_jsonable(self.std),
        }


class ZeroOne(Operator):
    r"""Scale the current scene extent into ``[0, 1]``.

    .. math::

        y \;=\; \frac{x - \min(x)}{\max(x) - \min(x)}

    Stateless per-scene min-max stretch. ``per_band=True`` (default)
    stretches each band independently; ``per_band=False`` uses a
    single global min / max.

    Args:
        per_band: Compute min / max per band rather than globally.

    Examples:
        >>> from geotoolz.normalize import ZeroOne
        >>> display = ZeroOne(per_band=True)(scene)
    """

    def __init__(self, *, per_band: bool = True) -> None:
        self.per_band = per_band

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        axis = stat_axes(arr, per_band=self.per_band)
        out = minmax_scale(
            arr,
            np.nanmin(arr, axis=axis),
            np.nanmax(arr, axis=axis),
            out_range=(0.0, 1.0),
            axis=axis,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"per_band": self.per_band}
