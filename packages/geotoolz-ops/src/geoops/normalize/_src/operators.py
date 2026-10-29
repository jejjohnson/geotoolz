"""Tier-B Operators — carrier-aware normalization transforms.

Each Operator wraps a Tier-A primitive in
:mod:`geoops.normalize._src.array`. Operators accept either a
``GeoArray`` or a plain ``np.ndarray`` and return the same carrier
kind (rewrap via :func:`geoops._src.wrap.wrap_like`). The
carrier-aware wrappers handle:

* fitting per-band statistics (``fit(x)`` / ``fit_on_call=True``) into
  fitted ``*_`` attributes that ``get_config()`` never serialises, with a
  pure ``transform(x)`` -- the scalers satisfy
  :class:`pipekit.protocols.FittableTransformer`,
* JSON-safe ``get_config()`` via the shared
  :func:`geoops._src.config.jsonable` helper (ndarray leaves become
  plain Python lists for Hydra / YAML round-trip),
* NaN-aware reductions over every axis except the band axis (``-3``):
  the spatial ``(H, W)`` axes of a ``(C, H, W)`` cube, plus the time
  axis of a ``(T, C, H, W)`` stack -- per-band statistics are ``(C,)``
  pooled over every frame, so one fit (and one stretch) serves the whole
  time series. :class:`CLAHE` equalizes each frame's bands separately.

Nodata: a pixel is invalid when any band is non-finite or equals the
carrier's ``fill_value`` (see :mod:`geoops._src.valid`;
per frame for ``(T, C, H, W)``). Invalid pixels are excluded from every
fitted statistic and hold ``NaN`` in every output. Normalized values
live on a new scale where the input's fill value (``0`` especially) can be
a real value, so outputs always declare ``fill_value=NaN``.

The display-prep min-max stretch with **scalar** bounds lives in
:class:`geoops.radiometry.MinMaxStretch`; the per-scene robust percentile
stretch lives in :class:`geoops.radiometry.PercentileClip`. This
module is the per-band normaliser shop used by ML pipelines:
``StandardScaler``, ``RobustScaler``, ``MinMaxScaler``, and the
fixed-stats alias ``Normalize``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
from pipekit import Operator, get_adapter

from geoops._src.config import jsonable
from geoops._src.fitted import fit_once, require_concrete
from geoops._src.geo import reject_geotensor
from geoops._src.kernels import named, pixel_kernel
from geoops._src.shape import DIMS
from geoops._src.stretch import check_percentiles, stretch_between
from geoops._src.valid import (
    invalid_pixel_mask,
    invalid_values,
    mask_invalid_to_nan,
    namespace_of,
    valid_pixels,
    wrap_filled,
)
from geoops.normalize._src.array import (
    BAND_STATS,
    asinh_scale,
    band_quantiles,
    band_stats,
    check_log_scale,
    check_positive,
    clahe,
    histogram_match,
    log_scale,
    minmax_scale,
    power_scale,
    reshape_stat,
    robust_scale,
    standard_scale,
    validate_out_range,
)


if TYPE_CHECKING:
    from geoarray import GeoArray


def _valid_mask(x: Any) -> np.ndarray:
    """Validity mask of ``x``: per pixel (per frame for 4-D), else elementwise."""
    if np.ndim(x) < 2:
        return ~invalid_values(x)
    return valid_pixels(x, keep_time=True)


def _masked(x: Any) -> tuple[np.ndarray, np.ndarray]:
    """Float64 copy of ``x`` with invalid pixels set to ``NaN``, plus the mask."""
    valid = _valid_mask(x)
    return mask_invalid_to_nan(x, valid=valid, dtype=float), valid


def _rewrap(gt: Any, out: np.ndarray, valid: np.ndarray) -> Any:
    """Write ``NaN`` into invalid pixels and rewrap like ``gt`` with a NaN fill."""
    return wrap_filled(gt, out, fill_value=np.nan, valid=valid)


def _check_per_band(stat: Any, gt: Any, dim: str = "band") -> None:
    """Raise unless ``stat`` is a scalar or one value per band of ``gt``."""
    stat_arr = np.asarray(stat, dtype=float)
    if stat_arr.size == 1:
        return
    adapter = get_adapter(gt)
    dims = adapter.dims(gt) if adapter is not None else DIMS[-np.ndim(gt) :]
    shape = tuple(gt.shape)
    kept = (shape[dims.index(dim)],) if dim in dims else ()
    if stat_arr.shape != kept:
        raise ValueError(
            f"statistic of shape {stat_arr.shape} does not match the kept axes "
            f"{kept} of an input of shape {shape}; pass one value per band "
            "(or a scalar)"
        )


def carrier_band_stats(
    x: Any,
    *,
    dim: str = "band",
    per_band: bool = True,
    stats: tuple[str, ...] | list[str] = BAND_STATS,
    quantiles: tuple[float, ...] | list[float] = (),
) -> dict[str, Any]:
    """Per-band statistics of a carrier's valid pixels, by dim name, on its backend.

    The carrier form of `band_stats` and `band_quantiles`. The band dim is
    found by name, so a ``(time, band, y, x)`` stack pools its frames into
    one ``(C,)`` statistic per band. Nodata follows geoops' pixel rule
    (`geoops._src.valid`): a pixel is invalid in every band when any of its
    bands is non-finite or equals the carrier's fill, judged per frame.

    Args:
        x: A NumPy or JAX `GeoArray` (under ``jax.jit`` too), an
            ``xr.DataArray`` (NumPy or dask), a `DimArray` or a plain 2-D to
            4-D array in the layout ``("time", "band", "y", "x")[-ndim:]``.
        dim: The band dim. A carrier without it gives scalar statistics.
        per_band: ``False`` reduces over every dim, bands too, to scalars
            (the pixel rule still uses ``dim``).
        stats: Names for `band_stats`; ``()`` for none.
        quantiles: Levels in ``[0, 1]`` for `band_quantiles`, returned under
            ``"quantiles"`` as ``(len(quantiles), C)``; ``()`` for none.

    Returns:
        One array per statistic on the data's backend: NumPy, JAX (tracers
        under ``jit``), or lazy dask arrays to compute together
        (``dask.compute(*out.values())``).

    Raises:
        TypeError: ``x`` is a georeader ``GeoTensor``.
        ValueError: An unknown statistic, or a level outside ``[0, 1]``.

    Examples:
        >>> scene = np.array([[[1.0, 3.0]], [[2.0, np.nan]]])  # (band, y, x)
        >>> # the NaN in band 1 invalidates that pixel in band 0 too
        >>> carrier_band_stats(scene, stats=("mean",))
        {'mean': array([1., 2.])}
        >>> out = carrier_band_stats(scene, stats=(), quantiles=(0.5,))
        >>> out["quantiles"].shape  # (len(quantiles), C)
        (1, 2)
        >>> float(carrier_band_stats(scene, per_band=False, stats=("max",))["max"])
        2.0
    """
    carrier, _ = named(x, name="carrier_band_stats")
    adapter = get_adapter(carrier)
    dims = tuple(adapter.dims(carrier))
    data = adapter.data(carrier)
    valid = ~invalid_pixel_mask(carrier, dim=dim)
    axis = dims.index(dim) if per_band and dim in dims else None
    out = dict(band_stats(data, valid, band_axis=axis, stats=stats))
    if quantiles:
        out["quantiles"] = band_quantiles(data, valid, band_axis=axis, q=quantiles)
    return out


def _scene_stats(x: Any, **kwargs: Any) -> dict[str, Any]:
    """`carrier_band_stats` of ``x``, with lazy dask statistics computed in one pass.

    A per-call statistic feeds an element-wise kernel as a constant, so on
    dask the statistics (not the data) are computed here, together, and the
    transform that uses them stays lazy. NumPy and JAX values (tracers under
    ``jax.jit``) pass through.
    """
    stats = carrier_band_stats(x, **kwargs)
    if any(type(v).__module__.startswith("dask.") for v in stats.values()):
        import dask

        stats = dict(zip(stats, dask.compute(*stats.values()), strict=True))
    return stats


_FUNCTIONAL = (
    "geoops.normalize.carrier_band_stats, which returns the statistics as "
    "arrays inside the trace."
)


def _host_stats(
    x: Any, call: str, alternative: str, **kwargs: Any
) -> dict[str, np.ndarray]:
    """Per-band statistics of ``x`` to store on an operator, as host float arrays.

    Stored statistics are always concrete NumPy arrays: computed on dask in
    one pass, copied off a JAX device, and refused inside a ``jax.jit`` trace
    before anything is computed. They stay strict JSON and pickle to workers.
    """
    require_concrete(x, call, alternative=alternative)
    stats = _scene_stats(x, **kwargs)
    return {key: np.asarray(value, dtype=float) for key, value in stats.items()}


def _fit_stats(op: Any, x: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    """`_host_stats` for ``op.fit``."""
    name = type(op).__name__
    alternative = (
        f"Call {name}().fit(x) eagerly, outside the trace, and apply the fitted "
        f"operator inside it; or use {_FUNCTIONAL}"
    )
    return _host_stats(x, f"{name}.fit", alternative, **kwargs)


def _standard_inverse(x: Any, *, mean: Any, std: Any) -> Any:
    """``x * std + mean``, a zero ``std`` read as ``1`` (`StandardScaler.inverse`)."""
    std_b = reshape_stat(std, x, None)
    return x * namespace_of(x).where(std_b != 0, std_b, 1.0) + reshape_stat(
        mean, x, None
    )


def _robust_inverse(x: Any, *, median: Any, iqr: Any) -> Any:
    """``x * iqr + median``, a zero ``iqr`` read as ``1`` (`RobustScaler.inverse`)."""
    iqr_b = reshape_stat(iqr, x, None)
    return x * namespace_of(x).where(iqr_b != 0, iqr_b, 1.0) + reshape_stat(
        median, x, None
    )


def _minmax_inverse(
    x: Any, *, vmin: Any, vmax: Any, out_range: tuple[float, float]
) -> Any:
    """Map ``out_range`` back onto ``[vmin, vmax]`` (`MinMaxScaler.inverse`)."""
    vmin_b = reshape_stat(vmin, x, None)
    vmax_b = reshape_stat(vmax, x, None)
    span = namespace_of(x).where(vmax_b > vmin_b, vmax_b - vmin_b, 1.0)
    out_min, out_max = out_range
    return (x - out_min) / (out_max - out_min) * span + vmin_b


def _stretch_kernel(x: Any, *, lo: Any, hi: Any, out_range: tuple[float, float]) -> Any:
    """Clip to ``[lo, hi]`` per band and map into ``out_range`` (`HistogramStretch`)."""
    out_min, out_max = out_range
    return stretch_between(x, lo, hi) * (out_max - out_min) + out_min


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
    downstream normaliser. Nodata (fill / non-finite) pixels are
    excluded from every statistic.

    Backends: NumPy, eager JAX and dask, where one compute of the
    statistics (not of the data) runs and the input passes through still
    lazy. Under ``jax.jit`` writing ``self.stats`` would leak tracers, so it
    raises; use `carrier_band_stats` inside a trace instead.

    Args:
        percentiles: Percentiles (in ``[0, 100]``) to cache alongside
            mean/std/min/max. Default ``[1.0, 99.0]``.

    Examples:
        >>> from geoops.normalize import PerBandStats
        >>> op = PerBandStats(percentiles=[2.0, 98.0])
        >>> _ = op(scene)
        >>> op.stats["mean"]   # one entry per band
        [...]
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax", "dask")
    # writes self.stats: eager on JAX, refused under jax.jit (the contract checks)
    traceable: ClassVar[bool] = False

    def __init__(self, *, percentiles: list[float] | None = None) -> None:
        self.percentiles = [1.0, 99.0] if percentiles is None else list(percentiles)
        self.stats: dict[str, Any] = {}

    def _apply(self, gt: Any) -> Any:
        stats = _host_stats(
            gt,
            "PerBandStats",
            f"Call it eagerly, outside the trace; or use {_FUNCTIONAL}",
            quantiles=tuple(p / 100.0 for p in self.percentiles),
        )
        # no levels gives no "quantiles" entry; keep the old empty list
        stats["percentiles"] = stats.pop("quantiles", np.empty(0))
        self.stats = {key: jsonable(value) for key, value in stats.items()}
        return gt


class CLAHE(Operator):
    """Contrast-Limited Adaptive Histogram Equalization.

    Wraps :func:`skimage.exposure.equalize_adapthist` and applies it
    independently per band for ``(C, H, W)`` carriers while preserving
    NaN positions and GeoArray metadata. Nodata (fill / non-finite)
    pixels are excluded from the histograms and hold the output fill.

    Args:
        window: Contextual-region shape for the local histograms.
            ``None`` uses skimage's default (1/8 of the image height /
            width). Lists (e.g. from Hydra / OmegaConf round-trips)
            are normalised to tuples.
        clip_limit: Contrast-limiting clip threshold in ``[0, 1]``.
            Default ``0.01``.
        nbins: Number of histogram bins. Default ``256``.

    Examples:
        >>> from geoops.normalize import CLAHE
        >>> out = CLAHE(window=(8, 8), clip_limit=0.03)(scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        window: int | tuple[int, int] | list[int] | None = None,
        clip_limit: float = 0.01,
        nbins: int = 256,
    ) -> None:
        # Accept list inputs (e.g. round-trips through Hydra/OmegaConf,
        # which materialise sequences as lists) and normalise to tuple
        # so :func:`equalize_adapthist` receives its expected type.
        if isinstance(window, list):
            window = tuple(window)
        self.window = window
        self.clip_limit = clip_limit
        self.nbins = nbins

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        arr, valid = _masked(gt)
        out = clahe(
            arr,
            window=self.window,
            clip_limit=self.clip_limit,
            nbins=self.nbins,
        )
        return _rewrap(gt, out, valid)

    def get_config(self) -> dict[str, Any]:
        if isinstance(self.window, tuple):
            window: int | list[int] | None = list(self.window)
        else:
            window = self.window
        return {
            "window": window,
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
    for inference, call :meth:`fit` on a training scene, or set
    ``fit_on_call=True`` to fit on the first scene seen. When
    ``sigma == 0`` (a constant band) the divisor falls back to ``1`` so
    the band collapses to zero instead of producing ``inf`` / ``nan``.
    Nodata (fill / non-finite) pixels are excluded from the fit and hold
    the output fill value.

    Fitted operator (see *Fitted operators* in the concepts guide):
    :meth:`fit` learns ``mean_`` / ``std_`` (which then take precedence
    over the constructor statistics), :meth:`transform` applies them
    without mutating the operator, and :meth:`inverse` undoes the
    scaling. Learned statistics are never part of ``get_config()``; to
    persist them, build ``StandardScaler(mean=op.mean_, std=op.std_)``.

    Backends: :meth:`fit` runs on NumPy, eager JAX and dask (one compute of
    the statistics, not of the data) and always stores host NumPy arrays.
    Under ``jax.jit`` it raises, and so does a first ``fit_on_call``
    call: fit eagerly, then apply the fitted operator inside the trace,
    where its statistics are constants. :meth:`transform` and
    :meth:`inverse` run everywhere, lazily on dask.

    Args:
        mean: Per-band mean (scalar, list, or ndarray) or ``None``.
        std: Per-band std (scalar, list, or ndarray) or ``None``.
        fit_on_call: If ``True`` and the operator is not fitted yet,
            fit ``mean_`` / ``std_`` on the first call and reuse them for
            every later call. The fit
            runs once even under concurrent calls (``pipekit.ThreadMap``),
            but which scene it sees is then the first to arrive -- call
            :meth:`fit` before parallel use for a deterministic fit.

    Attributes:
        mean_: Fitted per-band mean, or ``None`` before :meth:`fit`.
        std_: Fitted per-band std, or ``None`` before :meth:`fit`.

    Examples:
        >>> from geoops.normalize import StandardScaler
        >>> # Inference: cached per-band statistics from training.
        >>> scaler = StandardScaler(mean=[0.1, 0.2], std=[0.05, 0.07])
        >>> normed = scaler(scene)
        >>>
        >>> # Training-time fit, then apply:
        >>> fit = StandardScaler().fit(train_scene)
        >>> normed = fit(scene)
        >>> fit.mean_  # fitted per-band ndarray
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax", "dask")

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
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, gt: GeoArray | np.ndarray) -> Self:
        """Learn per-band ``mean_`` / ``std_`` from the valid pixels of ``gt``.

        Returns:
            This operator, for chaining.
        """
        stats = _fit_stats(self, gt, stats=("mean", "std"))
        self.std_ = stats["std"]
        self.mean_ = stats["mean"]  # published last: fit_once checks ``mean_``
        return self

    def _stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean_ is not None and self.std_ is not None:
            return self.mean_, self.std_
        if self.mean is not None and self.std is not None:
            return self.mean, self.std
        raise ValueError("StandardScaler requires mean/std, fit(), or fit_on_call=True")

    def transform(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        """Z-score ``gt`` with the fitted (or constructor) statistics.

        Raises:
            ValueError: If the operator has neither fitted nor constructor
                statistics.
        """
        mean, std = self._stats()
        _check_per_band(mean, gt)
        _check_per_band(std, gt)
        # the band dim is last inside the kernel, so (C,) stats broadcast as is
        return pixel_kernel(
            standard_scale,
            gt,
            dim="band",
            name=type(self).__name__,
            mean=mean,
            std=std,
            reduce_axes=None,
        )

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        if self.fit_on_call:
            fit_once(self, gt, lambda: self.mean_ is not None)
        return self.transform(gt)

    def inverse(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        """Invert a previously applied standard scaling.

        Args:
            gt: Scaled carrier (GeoArray or plain ndarray).

        Returns:
            The un-scaled carrier, same kind as ``gt``; nodata pixels
            hold the output fill value.

        Raises:
            ValueError: If the scaler has no statistics yet.
        """
        mean, std = self._stats()
        _check_per_band(mean, gt)
        _check_per_band(std, gt)
        return pixel_kernel(
            _standard_inverse,
            gt,
            dim="band",
            name=f"{type(self).__name__}.inverse",
            mean=mean,
            std=std,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": jsonable(self.mean),
            "std": jsonable(self.std),
            "fit_on_call": self.fit_on_call,
        }


class RobustScaler(Operator):
    r"""Per-band median / IQR scaling (robust to outliers).

    .. math::

        y \;=\; \frac{x - \mathrm{median}}{Q_{3} - Q_{1}}

    Identical role to :class:`StandardScaler` but uses the median and
    interquartile range instead of mean and std — robust against
    bright-pixel outliers (cumulus, glint, saturation). When
    ``iqr == 0`` the divisor falls back to ``1``. Nodata (fill /
    non-finite) pixels are excluded from the fit and hold the output fill.

    Fitted operator with the same ``fit`` / ``transform`` / ``inverse``
    contract as :class:`StandardScaler` (fitted state ``median_`` /
    ``iqr_``, never part of ``get_config()``).

    Args:
        median: Per-band median (scalar, list, or ndarray) or ``None``.
        iqr: Per-band IQR (``Q3 - Q1``).
        fit_on_call: If ``True`` and the operator is not fitted yet, fit
            on the first call (the 25th, 50th and 75th percentiles) and
            reuse the statistics afterwards.

    Attributes:
        median_: Fitted per-band median, or ``None`` before :meth:`fit`.
        iqr_: Fitted per-band IQR, or ``None`` before :meth:`fit`.

    Examples:
        >>> from geoops.normalize import RobustScaler
        >>> op = RobustScaler().fit(scene)
        >>> op.median_, op.iqr_
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax", "dask")

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
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None

    def fit(self, gt: GeoArray | np.ndarray) -> Self:
        """Learn per-band ``median_`` / ``iqr_`` from the valid pixels of ``gt``.

        Returns:
            This operator, for chaining.
        """
        quantiles = (0.25, 0.5, 0.75)
        q25, q50, q75 = _fit_stats(self, gt, stats=(), quantiles=quantiles)["quantiles"]
        self.iqr_ = q75 - q25
        self.median_ = q50  # published last: fit_once checks ``median_``
        return self

    def _stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self.median_ is not None and self.iqr_ is not None:
            return self.median_, self.iqr_
        if self.median is not None and self.iqr is not None:
            return self.median, self.iqr
        raise ValueError("RobustScaler requires median/iqr, fit(), or fit_on_call=True")

    def transform(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        """Scale ``gt`` with the fitted (or constructor) median / IQR."""
        median, iqr = self._stats()
        _check_per_band(median, gt)
        _check_per_band(iqr, gt)
        return pixel_kernel(
            robust_scale,
            gt,
            dim="band",
            name="RobustScaler",
            median=median,
            iqr=iqr,
            reduce_axes=None,
        )

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        if self.fit_on_call:
            fit_once(self, gt, lambda: self.median_ is not None)
        return self.transform(gt)

    def inverse(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        """Invert a previously applied robust scaling (``x = y * iqr + median``)."""
        median, iqr = self._stats()
        _check_per_band(median, gt)
        _check_per_band(iqr, gt)
        return pixel_kernel(
            _robust_inverse,
            gt,
            dim="band",
            name="RobustScaler.inverse",
            median=median,
            iqr=iqr,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "median": jsonable(self.median),
            "iqr": jsonable(self.iqr),
            "fit_on_call": self.fit_on_call,
        }


class MinMaxScaler(Operator):
    r"""Per-band min-max scaling into an arbitrary output range.

    .. math::

        y \;=\; \frac{x - v_{\min}}{v_{\max} - v_{\min}}
                \cdot (o_{\max} - o_{\min}) + o_{\min}

    Per-band variant of :class:`geoops.radiometry.MinMaxStretch`. The
    radiometry version takes **scalar** bounds and is the display-prep
    pick when you already know fixed reflectance limits; this version
    fits **per-band** bounds (one ``vmin`` / ``vmax`` per channel) or
    accepts cached training-time bounds for inference. When
    ``vmax == vmin`` for a band the divisor falls back to ``1``. Nodata
    (fill / non-finite) pixels are excluded from the fit and hold the
    output fill value.

    Fitted operator with the same ``fit`` / ``transform`` / ``inverse``
    contract as :class:`StandardScaler` (fitted state ``vmin_`` /
    ``vmax_``, never part of ``get_config()``).

    Args:
        vmin: Per-band lower bound, scalar / list / ndarray, or
            ``None`` to fit.
        vmax: Per-band upper bound.
        out_range: ``(out_min, out_max)`` range to map into. Default
            ``(0.0, 1.0)``.
        fit_on_call: If ``True`` and the operator is not fitted yet, fit
            the bounds on the first call and reuse them afterwards.

    Attributes:
        vmin_: Fitted per-band lower bound, or ``None`` before :meth:`fit`.
        vmax_: Fitted per-band upper bound, or ``None`` before :meth:`fit`.

    Examples:
        >>> from geoops.normalize import MinMaxScaler
        >>> # Map per-band [vmin, vmax] into [0, 1]:
        >>> op = MinMaxScaler(vmin=[0.0, 0.0], vmax=[0.3, 0.4])
        >>> scaled = op(scene)
        >>>
        >>> # Or fit on the scene and emit a uint8-style range:
        >>> op = MinMaxScaler(out_range=(0.0, 255.0)).fit(scene)
        >>> display = op(scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax", "dask")

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
        self.vmin_: np.ndarray | None = None
        self.vmax_: np.ndarray | None = None

    def fit(self, gt: GeoArray | np.ndarray) -> Self:
        """Learn per-band ``vmin_`` / ``vmax_`` from the valid pixels of ``gt``.

        Returns:
            This operator, for chaining.
        """
        stats = _fit_stats(self, gt, stats=("min", "max"))
        self.vmax_ = stats["max"]
        self.vmin_ = stats["min"]  # published last: fit_once checks ``vmin_``
        return self

    def _stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self.vmin_ is not None and self.vmax_ is not None:
            return self.vmin_, self.vmax_
        if self.vmin is not None and self.vmax is not None:
            return self.vmin, self.vmax
        raise ValueError("MinMaxScaler requires vmin/vmax, fit(), or fit_on_call=True")

    def transform(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        """Map ``gt`` from the fitted (or constructor) bounds into ``out_range``."""
        vmin, vmax = self._stats()
        _check_per_band(vmin, gt)
        _check_per_band(vmax, gt)
        return pixel_kernel(
            minmax_scale,
            gt,
            dim="band",
            name="MinMaxScaler",
            vmin=vmin,
            vmax=vmax,
            out_range=self.out_range,
            reduce_axes=None,
        )

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        if self.fit_on_call:
            fit_once(self, gt, lambda: self.vmin_ is not None)
        return self.transform(gt)

    def inverse(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        """Map ``out_range`` values back to the ``[vmin, vmax]`` input scale."""
        vmin, vmax = self._stats()
        _check_per_band(vmin, gt)
        _check_per_band(vmax, gt)
        return pixel_kernel(
            _minmax_inverse,
            gt,
            dim="band",
            name="MinMaxScaler.inverse",
            vmin=vmin,
            vmax=vmax,
            out_range=self.out_range,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "vmin": jsonable(self.vmin),
            "vmax": jsonable(self.vmax),
            "out_range": list(self.out_range),
            "fit_on_call": self.fit_on_call,
        }


class HistogramStretch(Operator):
    r"""Per-band percentile stretch into ``out_range`` for visualisation.

    Lighter-weight cousin of :class:`geoops.radiometry.PercentileClip`
    (same shared percentile stretch, same ``lower`` / ``upper``
    vocabulary): clips to ``[P_lower, P_upper]`` using NaN-aware
    percentiles, then maps the result into ``out_range`` instead of
    fixed ``[0, 1]``. Percentiles follow the ``normalize`` per-band
    statistics convention: per band, pooled over the frames of a
    ``(T, C, H, W)`` stack (``PercentileClip`` stretches each frame
    independently by default). Nodata (fill / non-finite) pixels are
    excluded from the percentiles and hold the output fill value.

    Args:
        out_range: Two-element increasing tuple. Default
            ``(0.0, 1.0)``.
        lower: Lower percentile. Default ``2.0``.
        upper: Upper percentile. Default ``98.0``; must exceed ``lower``.

    Examples:
        >>> from geoops.normalize import HistogramStretch
        >>> # Standard "satellite RGB to display byte range" stretch.
        >>> op = HistogramStretch(out_range=(0.0, 255.0))
        >>> display = op(reflectance_scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax", "dask")

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

    def _apply(self, gt: Any) -> Any:
        check_percentiles(self.lower, self.upper)
        q = (self.lower / 100.0, self.upper / 100.0)
        lo, hi = _scene_stats(gt, stats=(), quantiles=q)["quantiles"]
        # the band dim is last inside the kernel, so (C,) bounds broadcast as is
        return pixel_kernel(
            _stretch_kernel,
            gt,
            dim="band",
            name="HistogramStretch",
            lo=lo,
            hi=hi,
            out_range=self.out_range,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "out_range": list(self.out_range),
            "lower": self.lower,
            "upper": self.upper,
        }


class HistogramMatch(Operator):
    """Match a GeoArray histogram to a reference GeoArray.

    Reshapes the per-band empirical CDF of the input to match the
    reference. Useful for visual harmonisation across scenes acquired
    under different illumination. Nodata (fill / non-finite) pixels of
    both the input and the reference are excluded from the CDFs; input
    nodata pixels hold the output fill value.

    A ``(T, C, H, W)`` stack is matched per band with one CDF pooled over
    every frame (like the module's other fitted statistics), so the
    relative brightness of the frames survives the matching; a 4-D
    reference likewise pools its frames per band.

    The reference is a live GeoArray — not JSON / YAML serialisable —
    so ``forbid_in_yaml = True``.

    Args:
        reference: Reference GeoArray whose per-band CDF the input
            will be matched against.

    Examples:
        >>> from geoops.normalize import HistogramMatch
        >>> op = HistogramMatch(reference=reference_scene)
        >>> matched = op(source_scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    # Holds a live GeoArray reference, which is not JSON/YAML
    # serialisable.
    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, reference: GeoArray | np.ndarray) -> None:
        self.reference = reference

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        reject_geotensor(self.reference, "HistogramMatch (reference)")
        arr, valid = _masked(gt)
        reference, _ = _masked(self.reference)
        # The primitive matches along a leading band axis: put the band
        # axis of a (T, C, H, W) stack (or reference) first, so each band
        # pools its frames instead of a frame being taken for a band.
        if reference.ndim == 4:
            reference = np.moveaxis(reference, 1, 0)
        if arr.ndim == 4:
            matched = np.moveaxis(
                histogram_match(np.moveaxis(arr, 1, 0), reference), 0, 1
            )
        else:
            matched = histogram_match(arr, reference)
        return _rewrap(gt, matched, valid)

    def get_config(self) -> dict[str, Any]:
        # Debug payload: the reference is a runtime raster (forbid_in_yaml).
        reference = np.asarray(self.reference)
        return {
            "reference": {
                "shape": list(reference.shape),
                "dtype": str(reference.dtype),
            }
        }


class LogScale(Operator):
    r"""Logarithmic scaling with an epsilon offset for zeros.

    .. math::

        y \;=\; \log_{\text{base}}(\max(x, 0) + \epsilon)

    Compresses heavy-tailed distributions (radar backscatter, fire
    radiative power, etc.) before downstream training. ``eps`` keeps
    the log finite at zero. Nodata (fill / non-finite) pixels hold the
    output fill value.

    Args:
        base: Logarithm base. Must be ``> 0`` and ``!= 1``. Default
            ``10.0``.
        eps: Small positive offset added before the log. Default
            ``1e-6``.

    Examples:
        >>> from geoops.normalize import LogScale
        >>> log_sar = LogScale(base=10.0, eps=1e-6)(sar_backscatter)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax", "dask")

    def __init__(self, *, base: float = 10.0, eps: float = 1e-6) -> None:
        self.base = base
        self.eps = eps

    def _apply(self, gt: Any) -> Any:
        check_log_scale(self.base, self.eps)
        return pixel_kernel(
            log_scale, gt, dim="band", name="LogScale", base=self.base, eps=self.eps
        )


class AsinhScale(Operator):
    r"""Inverse-hyperbolic-sine scaling.

    .. math::

        y \;=\; \mathrm{asinh}(x / a)

    Linear near zero, logarithmic for ``|x| >> a``. Symmetric and
    well-defined for negative values (unlike ``log``) — a common pick
    for astronomy and signed radar quantities. Nodata (fill /
    non-finite) pixels hold the output fill value.

    Args:
        a: Scale parameter. Must be strictly positive. Default
            ``1.0``.

    Examples:
        >>> from geoops.normalize import AsinhScale
        >>> out = AsinhScale(a=0.1)(scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax", "dask")

    def __init__(self, *, a: float = 1.0) -> None:
        self.a = a

    def _apply(self, gt: Any) -> Any:
        check_positive("a", self.a)
        return pixel_kernel(asinh_scale, gt, dim="band", name="AsinhScale", a=self.a)


class PowerScale(Operator):
    r"""Non-negative power scaling.

    .. math::

        y \;=\; \max(x, 0)^{\gamma}

    A simple ``gamma``-style brightness curve. ``gamma < 1`` brightens
    midtones; ``gamma > 1`` darkens. Differs from
    :class:`geoops.radiometry.Gamma` only in convention
    (``Gamma`` raises to ``1/g``). Nodata (fill / non-finite) pixels
    hold the output fill value.

    Args:
        gamma: Power exponent. Must be strictly positive. Default
            ``0.5`` (square-root).

    Examples:
        >>> from geoops.normalize import PowerScale
        >>> out = PowerScale(gamma=0.5)(scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax", "dask")

    def __init__(self, *, gamma: float = 0.5) -> None:
        self.gamma = gamma

    def _apply(self, gt: Any) -> Any:
        check_positive("gamma", self.gamma)
        return pixel_kernel(
            power_scale, gt, dim="band", name="PowerScale", gamma=self.gamma
        )


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
        >>> from geoops.normalize import Normalize
        >>> import numpy as np
        >>> op = Normalize(
        ...     mean=np.array([0.1, 0.2, 0.15]),
        ...     std=np.array([0.05, 0.07, 0.06]),
        ... )
        >>> normed = op(scene)
    """

    # fixed statistics: _apply never fits, so the transform runs anywhere
    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax", "dask")

    def __init__(
        self,
        *,
        mean: np.ndarray | list[float] | float,
        std: np.ndarray | list[float] | float,
    ) -> None:
        super().__init__(mean=mean, std=std, fit_on_call=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": jsonable(self.mean),
            "std": jsonable(self.std),
        }


class ZeroOne(Operator):
    r"""Scale the current scene extent into ``[0, 1]``.

    .. math::

        y \;=\; \frac{x - \min(x)}{\max(x) - \min(x)}

    Stateless per-scene min-max stretch. ``per_band=True`` (default)
    stretches each band independently; ``per_band=False`` uses a
    single global min / max. Nodata (fill / non-finite) pixels are
    excluded from the min / max and hold the output fill value.

    Args:
        per_band: Compute min / max per band rather than globally.

    Examples:
        >>> from geoops.normalize import ZeroOne
        >>> display = ZeroOne(per_band=True)(scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax", "dask")

    def __init__(self, *, per_band: bool = True) -> None:
        self.per_band = per_band

    def _apply(self, gt: Any) -> Any:
        stats = _scene_stats(gt, per_band=self.per_band, stats=("min", "max"))
        return pixel_kernel(
            minmax_scale,
            gt,
            dim="band",
            name="ZeroOne",
            vmin=stats["min"],
            vmax=stats["max"],
            out_range=(0.0, 1.0),
            reduce_axes=None,
        )
