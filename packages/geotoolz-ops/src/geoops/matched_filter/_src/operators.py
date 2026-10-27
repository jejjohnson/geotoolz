"""Carrier-aware matched-filter Operators.

Thin ``pipekit.Operator`` wrappers around the Tier-A primitives in
:mod:`geoops.matched_filter._src.array`. Matched-filter math is
metadata-independent (covariance / Mahalanobis algebra over spectra),
so every operator here accepts either a ``GeoArray`` or a plain
``np.ndarray`` cube. Operators whose output is an image (score maps)
rewrap it to match the input carrier via
:func:`geoops._src.wrap.wrap_like`; the estimator operators return
non-carrier results (mean vectors, `NumpyLinearOperator`, background
dataclasses) unchanged for either input kind.

Band axis: the spectral axis defaults to ``-3`` (the carrier's
``(C, H, W)`` / ``(T, C, H, W)`` layout). A 2-D ``(H, W)`` map has no
band axis and raises ``ValueError`` (its rows are never read as bands); a
2-D ``(N, C)`` sample matrix needs an explicit ``axis``. On a
``(T, C, H, W)`` stack the background statistics are fitted once over the
pixels of every frame (one background for the time series) and score maps
come back as ``(T, 1, H, W)``.

Nodata: pixels that are non-finite or equal the carrier's
``fill_value`` in any band are excluded from every fitted
statistic (see :mod:`geoops._src.valid`); score maps are a new
quantity, so they hold ``NaN`` at those pixels and declare
``fill_value=NaN`` whatever the input's fill value.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import numpy as np
from pipekit import Operator

from geoops.carrier import (
    band_dim,
    callable_name,
    cube_to_samples,
    dim_axis,
    fit_once,
    invalid_values,
    jsonable,
    keep_band_axis,
    reject_config_summary,
    reject_geotensor,
    require_ndim,
    restore_fill,
    wrap_like,
)
from geoops.matched_filter._src.array import (
    AdaptiveBackground,
    ClusterBackground,
    CovMethod,
    CovShrinkageMethod,
    MeanMethod,
    NumpyLinearOperator,
    StreamingBackgroundResult,
    WelfordAccumulator,
    adaptive_window_background,
    apply_adaptive_mf,
    apply_cluster_mf,
    apply_image,
    apply_pixel,
    detection_threshold,
    estimate_cov_empirical,
    estimate_cov_lowrank,
    estimate_cov_shrunk,
    estimate_mean,
    gmm_cluster_background,
    matched_filter_snr,
    shrink_covariance,
    validate_mf_inputs,
)


if TYPE_CHECKING:
    from geoarray import GeoArray


# Small relative perturbation for finite-difference target linearization; callers
# needing scale-specific control can use NonlinearTargetFromObs(amplitude=...).
TARGET_FINITE_DIFFERENCE_EPSILON = 1e-4


class MatchedFilter(Operator):
    """Per-pixel matched-filter score map.

    Computes ``y = (x - mu)^T Sigma^-1 t / (t^T Sigma^-1 t)`` for every
    pixel spectrum and returns a single-band score map. Accepts either a
    ``GeoArray`` (georeferencing preserved on the output) or a plain
    ``np.ndarray`` (plain array returned) — the math is metadata-free.

    Fitted operator (see *Fitted operators* in the concepts guide):

    * :meth:`fit` estimates the background statistics the constructor
      left unset (``mean`` and / or ``cov_op``) with ``mean_method`` /
      ``cov_method`` and stores the full background in ``mean_`` /
      ``cov_op_`` (supplied components are copied through unchanged);
    * :meth:`transform` scores a cube with that background without
      mutating the operator;
    * calling the operator fits the missing statistics once, on the first
      call (``fit_on_call=False``), or estimates them from each cube
      without storing anything (``fit_on_call=True``, a per-scene
      background). Neither path writes fitted arrays into
      ``get_config()``.

    Nodata pixels (non-finite or equal to the input's fill value in any
    band) are left out of the fit and hold ``NaN`` in the score map,
    which declares ``fill_value=NaN``.

    Args:
        mean: Background mean spectrum ``(c,)``; fitted from the data
            when ``None``.
        cov_op: Background covariance as a `NumpyLinearOperator` or raw
            ``(c, c)`` matrix (a nested list, e.g. from a config reload,
            is coerced to an ndarray); fitted from the data when ``None``.
        target: Target signature ``(c,)``. Must be set before applying.
        fit_on_call: Estimate mean and covariance from every cube the
            operator is called on (a per-scene background, ignoring any
            supplied or fitted statistics) instead of fitting the missing
            ones once.
        mean_method: Mean estimator used when fitting. Default
            ``"median"``.
        cov_method: Covariance estimator used when fitting —
            ``"empirical"``, ``"ledoit_wolf"``, ``"oas"``, or
            ``"lowrank"``. Default ``"ledoit_wolf"``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.

    Attributes:
        mean_: Background mean from :meth:`fit`, or ``None``.
        cov_op_: Background covariance from :meth:`fit`, or ``None``.

    Examples:
        >>> mf = MatchedFilter(target=t).fit(background_scene)
        >>> scores = mf(scene)  # reuses the fitted background
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        mean: np.ndarray | None = None,
        cov_op: NumpyLinearOperator | np.ndarray | None = None,
        target: np.ndarray | None = None,
        fit_on_call: bool = False,
        mean_method: MeanMethod = "median",
        cov_method: CovMethod = "ledoit_wolf",
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.cov_op = (
            cov_op
            if cov_op is None or isinstance(cov_op, NumpyLinearOperator)
            else np.asarray(cov_op, dtype=float)
        )
        self.target = None if target is None else np.asarray(target, dtype=float)
        self.fit_on_call = fit_on_call
        self.mean_method = mean_method
        self.cov_method = cov_method
        self.dim = band_dim(dim, axis, "MatchedFilter")
        self.mean_: np.ndarray | None = None
        self.cov_op_: NumpyLinearOperator | np.ndarray | None = None

    def _estimate(
        self, cube: np.ndarray, *, axis: int, keep_supplied: bool
    ) -> tuple[np.ndarray, NumpyLinearOperator | np.ndarray]:
        """Background ``(mean, cov_op)`` of ``cube``.

        ``axis`` is the band position, resolved from the carrier before its
        names were dropped. ``keep_supplied`` keeps the constructor's
        ``mean`` / ``cov_op`` and only estimates the missing ones.
        """
        mean = self.mean if keep_supplied else None
        cov_op = self.cov_op if keep_supplied else None
        if mean is None:
            mean = estimate_mean(cube, method=self.mean_method, axis=axis)
        if cov_op is None:
            if self.cov_method == "empirical":
                cov_op = estimate_cov_empirical(cube, mean=mean, ridge=1e-8, axis=axis)
            elif self.cov_method == "lowrank":
                cov_op = estimate_cov_lowrank(cube, mean=mean, axis=axis)
            else:
                cov_op = estimate_cov_shrunk(
                    cube, mean=mean, method=self.cov_method, axis=axis
                )
        return np.asarray(mean, dtype=float), cov_op

    def fit(self, gt: GeoArray | np.ndarray) -> Self:
        """Estimate the unset background statistics from the valid pixels of ``gt``.

        Returns:
            This operator, for chaining.
        """
        axis = dim_axis(gt, self.dim, name="MatchedFilter")
        cube, _ = _mask_invalid(gt, axis, type(self).__name__)
        mean, cov_op = self._estimate(cube, axis=axis, keep_supplied=True)
        self.cov_op_ = cov_op
        self.mean_ = mean  # published last: the fit-once check reads ``mean_``
        return self

    def _background(self) -> tuple[np.ndarray, NumpyLinearOperator | np.ndarray]:
        if self.mean_ is not None and self.cov_op_ is not None:
            return self.mean_, self.cov_op_
        if self.mean is not None and self.cov_op is not None:
            return self.mean, self.cov_op
        raise ValueError(
            "MatchedFilter requires mean and cov_op, fit(), or a call on data"
        )

    def transform(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        """Score ``gt`` against the fitted (or supplied) background.

        Raises:
            ValueError: If no background is available yet or ``target``
                is unset.
        """
        mean, cov_op = self._background()
        return self._score(gt, mean, cov_op)

    def _score(
        self,
        gt: GeoArray | np.ndarray,
        mean: np.ndarray,
        cov_op: NumpyLinearOperator | np.ndarray,
        masked: tuple[np.ndarray, np.ndarray | None] | None = None,
    ) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="MatchedFilter")
        if self.target is None:
            raise ValueError("target must be supplied before applying MatchedFilter")
        cube, valid = (
            _mask_invalid(gt, axis, type(self).__name__) if masked is None else masked
        )
        out = apply_image(cube, mean=mean, cov_op=cov_op, target=self.target, axis=axis)
        return wrap_like(
            gt,
            keep_band_axis(_restore(gt, out, valid), gt),
            fill_value=np.nan,
        )

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        if self.fit_on_call:
            # Per-scene background: estimated locally, never stored, so
            # concurrent calls share no mutable state.
            axis = dim_axis(gt, self.dim, name="MatchedFilter")
            masked = _mask_invalid(gt, axis, type(self).__name__)
            mean, cov_op = self._estimate(masked[0], axis=axis, keep_supplied=False)
            return self._score(gt, mean, cov_op, masked)
        if self.mean is None or self.cov_op is None:
            fit_once(self, gt, lambda: self.mean_ is not None)
        return self.transform(gt)

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean is None else np.asarray(self.mean).tolist(),
            "cov_op": _cov_config(self.cov_op),
            "target": None if self.target is None else np.asarray(self.target).tolist(),
            "fit_on_call": self.fit_on_call,
            "mean_method": self.mean_method,
            "cov_method": self.cov_method,
            "dim": self.dim,
        }


class MatchedFilterPixel(Operator):
    """Single-pixel matched-filter score.

    Scalar counterpart of :class:`MatchedFilter`: scores one pixel
    spectrum (any 1-D array-like) against fixed background statistics
    and returns a ``float``.

    Args:
        mean: Background mean spectrum ``(c,)``.
        cov_op: Background covariance as a `NumpyLinearOperator` or raw
            ``(c, c)`` matrix.
        target: Target signature ``(c,)``.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        mean: np.ndarray,
        cov_op: NumpyLinearOperator | np.ndarray,
        target: np.ndarray,
    ) -> None:
        self.mean = mean
        self.cov_op = cov_op
        self.target = target

    def _apply(self, pixel: np.ndarray) -> float:
        return apply_pixel(
            pixel, mean=self.mean, cov_op=self.cov_op, target=self.target
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": np.asarray(self.mean).tolist(),
            "cov_op": _cov_config(self.cov_op),
            "target": np.asarray(self.target).tolist(),
        }


class MatchedFilterSNR(Operator):
    """Theoretical detection SNR ``amplitude * sqrt(t^T Sigma^-1 t)``.

    Graph source: computed from its configuration alone. An optional
    input is accepted and ignored, so it can close a ``Sequential``;
    it returns the scalar
    matched-filter output SNR for the configured amplitude, background
    covariance, and target.

    Args:
        amplitude: Target amplitude.
        cov_op: Background covariance as a `NumpyLinearOperator` or raw
            ``(c, c)`` matrix.
        target: Target signature ``(c,)``.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        amplitude: float,
        cov_op: NumpyLinearOperator | np.ndarray,
        target: np.ndarray,
    ) -> None:
        self.amplitude = amplitude
        self.cov_op = cov_op
        self.target = target

    def _apply(self, _input: Any | None = None) -> float:
        reject_geotensor(_input, type(self).__name__)
        return matched_filter_snr(
            amplitude=self.amplitude, cov_op=self.cov_op, target=self.target
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "amplitude": self.amplitude,
            "cov_op": _cov_config(self.cov_op),
            "target": np.asarray(self.target).tolist(),
        }


class DetectionThreshold(Operator):
    """Gaussian false-alarm score threshold.

    Graph source: computed from its configuration alone. An optional
    input is accepted and ignored, so it can close a ``Sequential``;
    it returns the scalar score
    threshold whose exceedance probability under the Gaussian no-target
    null equals ``false_alarm_rate``.

    Args:
        false_alarm_rate: Desired false-alarm probability, strictly
            between 0 and 1.
        cov_op: Background covariance as a `NumpyLinearOperator` or raw
            ``(c, c)`` matrix.
        target: Target signature ``(c,)``.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        false_alarm_rate: float,
        cov_op: NumpyLinearOperator | np.ndarray,
        target: np.ndarray,
    ) -> None:
        self.false_alarm_rate = false_alarm_rate
        self.cov_op = cov_op
        self.target = target

    def _apply(self, _input: Any | None = None) -> float:
        reject_geotensor(_input, type(self).__name__)
        return detection_threshold(
            false_alarm_rate=self.false_alarm_rate,
            cov_op=self.cov_op,
            target=self.target,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "false_alarm_rate": self.false_alarm_rate,
            "cov_op": _cov_config(self.cov_op),
            "target": np.asarray(self.target).tolist(),
        }


class ValidateMFInputs(Operator):
    """Fail-fast pass-through validator for matched-filter inputs.

    Raises ``ValueError`` early for an all-zero target, a singular
    covariance, or a non-positive filter gain; otherwise returns its
    runtime input unchanged (whatever its type), so it can be inserted
    anywhere in a pipeline as a guard.

    Args:
        cov_op: Background covariance as a `NumpyLinearOperator` or raw
            ``(c, c)`` matrix.
        target: Target signature ``(c,)``.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax")

    def __init__(
        self,
        *,
        cov_op: NumpyLinearOperator | np.ndarray,
        target: np.ndarray,
    ) -> None:
        self.cov_op = cov_op
        self.target = target

    def _apply(self, value: Any = None) -> Any:
        reject_geotensor(value, type(self).__name__)
        validate_mf_inputs(cov_op=self.cov_op, target=self.target)
        return value

    def get_config(self) -> dict[str, Any]:
        return {
            "cov_op": _cov_config(self.cov_op),
            "target": np.asarray(self.target).tolist(),
        }


class EstimateMean(Operator):
    """Estimate a spectral background mean vector from a cube.

        Accepts a ``GeoArray`` or plain ``np.ndarray`` cube and returns the
        plain ``(c,)`` mean spectrum (a vector, not a carrier). See
    :func:`geoops.matched_filter._src.array.estimate_mean` for the
        estimator details. Nodata pixels (non-finite or equal to the input's
        fill value in any band) are ignored.

        Args:
            method: Mean estimator — ``"mean"``, ``"median"``, ``"trimmed"``,
                or ``"huber"``.
            trim_proportion: Fraction cut from each tail for
                ``method="trimmed"``.
            huber_c: Huber tuning constant for ``method="huber"``.
            dim: Name of the band dim (default ``"band"``); its position is read
                from the carrier's dims.
            axis: Deprecated: the band axis position. ``-3`` maps to
                ``dim="band"``; removed in the next minor release.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        method: MeanMethod = "mean",
        trim_proportion: float = 0.1,
        huber_c: float = 1.345,
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.method = method
        self.trim_proportion = trim_proportion
        self.huber_c = huber_c
        self.dim = band_dim(dim, axis, "EstimateMean")

    def _apply(self, gt: GeoArray | np.ndarray) -> np.ndarray:
        axis = dim_axis(gt, self.dim, name="EstimateMean")
        return estimate_mean(
            _mask_invalid(gt, axis, type(self).__name__)[0],
            method=self.method,
            trim_proportion=self.trim_proportion,
            huber_c=self.huber_c,
            axis=axis,
        )


class EstimateCovEmpirical(Operator):
    """Estimate an empirical covariance operator.

    Accepts a ``GeoArray`` or plain ``np.ndarray`` cube and returns a
    `NumpyLinearOperator` (not a carrier) wrapping the ``(c, c)`` sample
    covariance. Nodata pixels (non-finite or equal to the input's fill
    value in any band) are ignored.

    Args:
        mean: Optional precomputed mean spectrum ``(c,)`` to centre on;
            the sample mean is used when ``None``.
        ridge: Optional Tikhonov ridge added to the diagonal.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        mean: np.ndarray | None = None,
        ridge: float = 0.0,
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.mean = mean
        self.ridge = ridge
        self.dim = band_dim(dim, axis, "EstimateCovEmpirical")

    def _apply(self, gt: GeoArray | np.ndarray) -> NumpyLinearOperator:
        axis = dim_axis(gt, self.dim, name="EstimateCovEmpirical")
        return estimate_cov_empirical(
            _mask_invalid(gt, axis, type(self).__name__)[0],
            mean=self.mean,
            ridge=self.ridge,
            axis=axis,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean is None else np.asarray(self.mean).tolist(),
            "ridge": self.ridge,
            "dim": self.dim,
        }


class EstimateCovShrunk(Operator):
    """Estimate a diagonal-target shrinkage covariance operator.

    Accepts a ``GeoArray`` or plain ``np.ndarray`` cube and returns a
    `NumpyLinearOperator` (not a carrier) wrapping the covariance shrunk
    toward a scaled-identity target. Nodata pixels (non-finite or equal
    to the input's fill value in any band) are ignored.

    Args:
        mean: Optional precomputed mean spectrum ``(c,)`` to centre on;
            the sample mean is used when ``None``.
        method: Shrinkage-intensity estimator, ``"ledoit_wolf"`` or
            ``"oas"``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        mean: np.ndarray | None = None,
        method: CovShrinkageMethod = "ledoit_wolf",
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.mean = mean
        self.method = method
        self.dim = band_dim(dim, axis, "EstimateCovShrunk")

    def _apply(self, gt: GeoArray | np.ndarray) -> NumpyLinearOperator:
        axis = dim_axis(gt, self.dim, name="EstimateCovShrunk")
        return estimate_cov_shrunk(
            _mask_invalid(gt, axis, type(self).__name__)[0],
            mean=self.mean,
            method=self.method,
            axis=axis,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean is None else np.asarray(self.mean).tolist(),
            "method": self.method,
            "dim": self.dim,
        }


class EstimateCovLowRank(Operator):
    """Estimate a low-rank-plus-Tikhonov covariance operator.

    Accepts a ``GeoArray`` or plain ``np.ndarray`` cube and returns a
    `NumpyLinearOperator` (not a carrier) approximating the covariance
    by its top-``rank`` eigenpairs plus a Tikhonov diagonal. Nodata
    pixels (non-finite or equal to the input's fill value in any band)
    are ignored.

    Args:
        mean: Optional precomputed mean spectrum ``(c,)`` to centre on;
            the sample mean is used when ``None``.
        rank: Number of leading eigenpairs kept.
        tikhonov: Diagonal regulariser added after the low-rank
            reconstruction.
        seed: Seed for the randomized range finder; the default
            ``0`` makes the estimate deterministic.
        n_oversamples: Extra random probe vectors beyond ``rank``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        mean: np.ndarray | None = None,
        rank: int = 10,
        tikhonov: float = 1e-3,
        seed: int | None = 0,
        n_oversamples: int = 10,
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.mean = mean
        self.rank = rank
        self.tikhonov = tikhonov
        self.seed = seed
        self.n_oversamples = n_oversamples
        self.dim = band_dim(dim, axis, "EstimateCovLowRank")

    def _apply(self, gt: GeoArray | np.ndarray) -> NumpyLinearOperator:
        axis = dim_axis(gt, self.dim, name="EstimateCovLowRank")
        return estimate_cov_lowrank(
            _mask_invalid(gt, axis, type(self).__name__)[0],
            mean=self.mean,
            rank=self.rank,
            tikhonov=self.tikhonov,
            seed=self.seed,
            n_oversamples=self.n_oversamples,
            axis=axis,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean is None else np.asarray(self.mean).tolist(),
            "rank": self.rank,
            "tikhonov": self.tikhonov,
            "seed": self.seed,
            "n_oversamples": self.n_oversamples,
            "dim": self.dim,
        }


class GMMClusterBackground(Operator):
    """Estimate clustered background statistics with a NumPy GMM backend.

    Runs the deterministic diagonal-GMM clustering (k-means initialised)
    of :func:`geoops.matched_filter._src.array.gmm_cluster_background`.
    Accepts a ``GeoArray`` or plain ``np.ndarray`` cube and returns a
    `ClusterBackground` (labels plus per-cluster statistics, not a
    carrier), typically wired into :class:`ApplyClusterMF`. Nodata pixels
    (non-finite or equal to the input's fill value in any band) are left
    out of the fit and labelled ``-1``.

    Args:
        n_clusters: Number of mixture components.
        cov_estimator: Per-cluster covariance estimator —
            ``"empirical"``, ``"ledoit_wolf"``, or ``"oas"``.
        seed: Seed for the clustering; fixed seeds give
            reproducible backgrounds.
        bayesian: If ``True``, prune components with negligible mixture
            weight, so fewer than ``n_clusters`` clusters may remain.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        n_clusters: int,
        cov_estimator: Literal["empirical", "ledoit_wolf", "oas"] = "ledoit_wolf",
        seed: int | None = 0,
        bayesian: bool = False,
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.cov_estimator = cov_estimator
        self.seed = seed
        self.bayesian = bayesian
        self.dim = band_dim(dim, axis, "GMMClusterBackground")

    def _apply(self, gt: GeoArray | np.ndarray) -> ClusterBackground:
        axis = dim_axis(gt, self.dim, name="GMMClusterBackground")
        return gmm_cluster_background(
            _mask_invalid(gt, axis, type(self).__name__)[0],
            n_clusters=self.n_clusters,
            cov_estimator=self.cov_estimator,
            seed=self.seed,
            bayesian=self.bayesian,
            axis=axis,
        )


class AdaptiveWindowBackground(Operator):
    """Per-pixel local mean and diagonal variance over a square window.

    Accepts a ``GeoArray`` or plain ``np.ndarray`` 3-D cube and returns
    an `AdaptiveBackground` (a pair of statistic cubes, not a carrier).
    Nodata pixels (non-finite or equal to the input's fill value in any
    band) carry no window weight and are ``NaN`` in both cubes.

    Args:
        window: Side length of the square window; positive odd
            integer. Default ``7``.
        pad_mode: Boundary mode forwarded to
            :func:`scipy.ndimage.uniform_filter`. Default ``"reflect"``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        window: int = 7,
        pad_mode: str = "reflect",
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.window = window
        self.pad_mode = pad_mode
        self.dim = band_dim(dim, axis, "AdaptiveWindowBackground")

    def _apply(self, gt: GeoArray | np.ndarray) -> AdaptiveBackground:
        axis = dim_axis(gt, self.dim, name="AdaptiveWindowBackground")
        require_ndim(gt, 3, type(self).__name__)
        return adaptive_window_background(
            _mask_invalid(gt, axis, type(self).__name__)[0],
            window=self.window,
            pad_mode=self.pad_mode,
            axis=axis,
        )


class ApplyClusterMF(Operator):
    """Apply a matched filter dispatched by per-pixel cluster label.

    Accepts a ``GeoArray`` or plain ``np.ndarray`` cube and returns the
    score map as the same carrier kind. Nodata pixels (non-finite or
    equal to the input's fill value in any band, or labelled ``-1``)
    hold ``NaN`` (``fill_value=NaN``).

    The ``cluster`` background is supplied at apply time as a runtime input
    so the operator config (``target``, ``axis``) is hydra-/YAML-safe and
    fully describes the operator's init-time state. Pair this operator
    with :class:`GMMClusterBackground` in a pipeline to wire the cluster
    state through dataflow.

    Args:
        target: Target signature ``(c,)``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        target: np.ndarray,
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.target = target
        self.dim = band_dim(dim, axis, "ApplyClusterMF")

    def _apply(
        self, gt: GeoArray | np.ndarray, cluster: ClusterBackground
    ) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="ApplyClusterMF")
        cube, valid = _mask_invalid(gt, axis, type(self).__name__)
        out = apply_cluster_mf(cube, cluster=cluster, target=self.target, axis=axis)
        return wrap_like(
            gt,
            keep_band_axis(_restore(gt, out, valid), gt),
            fill_value=np.nan,
        )

    def get_config(self) -> dict[str, Any]:
        return {"target": np.asarray(self.target).tolist(), "dim": self.dim}


class ApplyAdaptiveMF(Operator):
    """Apply a matched filter against per-pixel local window statistics.

    Scores each pixel with ``α = tᵀΛ⁻¹(x − μ) / (tᵀΛ⁻¹t)`` where ``μ`` and
    ``Λ = diag(σ² + ridge)`` are the pixel's own local mean and diagonal
    variance from :class:`AdaptiveWindowBackground` (see
    :func:`geoops.matched_filter._src.array.apply_adaptive_mf`).
    Accepts a ``GeoArray`` or plain ``np.ndarray`` 3-D cube and returns
    the score map as the same carrier kind. Nodata pixels (non-finite or
    equal to the input's fill value in any band) and pixels whose window
    statistics are undefined hold ``NaN`` (``fill_value=NaN``).

    The ``background`` is supplied at apply time as a runtime input so the
    config (``target``, ``ridge``, ``axis``) stays hydra-/YAML-safe; pair
    it with :class:`AdaptiveWindowBackground` in a pipeline::

        bg = AdaptiveWindowBackground(window=7)(gt)
        scores = ApplyAdaptiveMF(target=t)(gt, bg)

    Args:
        target: Target signature ``(c,)``.
        ridge: Non-negative variance floor added to every local ``σ²``.
            Default ``0.0``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        target: np.ndarray,
        ridge: float = 0.0,
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.target = target
        self.ridge = ridge
        self.dim = band_dim(dim, axis, "ApplyAdaptiveMF")

    def _apply(
        self, gt: GeoArray | np.ndarray, background: AdaptiveBackground
    ) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="ApplyAdaptiveMF")
        require_ndim(gt, 3, type(self).__name__)
        cube, valid = _mask_invalid(gt, axis, type(self).__name__)
        out = apply_adaptive_mf(
            cube,
            background=background,
            target=self.target,
            ridge=self.ridge,
            axis=axis,
        )
        return wrap_like(
            gt,
            keep_band_axis(_restore(gt, out, valid), gt),
            fill_value=np.nan,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "target": np.asarray(self.target).tolist(),
            "ridge": self.ridge,
            "dim": self.dim,
        }


class StreamingBackground(Operator):
    """Aggregate mean/covariance across an iterable of cubes.

    Streams every cube through a `WelfordAccumulator` and returns a
    `StreamingBackgroundResult` (mean vector plus covariance operator,
    not a carrier). Cubes may be ``GeoArray`` or plain ``np.ndarray``
    and can be mixed freely. Nodata pixels (non-finite or equal to each
    cube's fill value in any band) are not accumulated.

    The cubes iterable is supplied at apply time (runtime dataflow) so the
    operator config remains hydra-/YAML-serialisable: ``cov_kind`` and
    ``axis`` fully describe the operator's init-time state.

    Args:
        cov_kind: ``"shrunk"`` (Ledoit-Wolf toward scaled identity,
            default — identical to `estimate_cov_shrunk` on the
            concatenated pixels, since the accumulator streams the
            fourth moment LW needs) or ``"empirical"``. A ``1e-8`` ridge
            is added to the diagonal in both cases.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        cov_kind: Literal["empirical", "shrunk"] = "shrunk",
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.cov_kind = cov_kind
        self.dim = band_dim(dim, axis, "StreamingBackground")

    def _apply(
        self, cubes: Iterable[GeoArray | np.ndarray]
    ) -> StreamingBackgroundResult:
        acc: WelfordAccumulator | None = None
        for cube in cubes:
            axis = dim_axis(cube, self.dim, name="StreamingBackground")
            samples, _ = cube_to_samples(
                _mask_invalid(cube, axis, type(self).__name__)[0],
                band_axis=axis,
                dtype=float,
            )
            if acc is None:
                acc = WelfordAccumulator.empty(samples.shape[1])
            acc.update(samples)
        if acc is None:
            raise ValueError("cubes must contain at least one cube")
        cov = acc.covariance()
        if self.cov_kind == "shrunk":
            # Exact Ledoit-Wolf: the accumulator's streamed m4 = Σ‖xₖ − x̄‖⁴
            # gives the fourth moment LW needs without holding the samples.
            cov = shrink_covariance(
                cov,
                method="ledoit_wolf",
                n_samples=acc.count,
                fourth_moment=acc.m4 / acc.count,
            )
        elif self.cov_kind != "empirical":
            raise ValueError(f"unknown cov_kind {self.cov_kind!r}")
        cov = cov + 1e-8 * np.eye(cov.shape[0], dtype=float)
        return StreamingBackgroundResult(mean=acc.mean, cov_op=NumpyLinearOperator(cov))


class LinearTargetFromObs(Operator):
    """Finite-difference tangent-linear target from an observation model.

    Perturbs the background state by a small scaled pattern, pushes both
    states through ``obs_model``, and divides the response difference by
    the perturbation size — a per-band linearised (Jacobian-action)
    target signature ``(c,)``. Accepts a ``GeoArray`` or plain
    ``np.ndarray`` cube and returns a plain vector (not a carrier).

    Args:
        obs_model: Callable mapping a state cube to a spectral response
            (either a per-band vector or a cube that is averaged over
            space). Must be callable.
        vmr_background: Optional explicit background state; the incoming
            cube is used when ``None``.
        pattern: Perturbation pattern — ``"uniform"``, ``"impulse"``, or
            an explicit array broadcastable to the background state.
        pixel: Spatial index of the impulse for ``pattern="impulse"``;
            the first flat pixel when ``None``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        obs_model: Any,
        vmr_background: np.ndarray | None = None,
        pattern: str | np.ndarray = "uniform",
        pixel: tuple[int, int] | None = None,
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.obs_model = obs_model
        self.vmr_background = vmr_background
        self.pattern = pattern
        self.pixel = pixel
        self.dim = band_dim(dim, axis, "LinearTargetFromObs")

    def _apply(self, gt: GeoArray | np.ndarray) -> np.ndarray:
        return (
            _target_from_obs(
                self.obs_model,
                gt,
                vmr_background=self.vmr_background,
                pattern=self.pattern,
                pixel=self.pixel,
                amplitude=TARGET_FINITE_DIFFERENCE_EPSILON,
                axis=dim_axis(gt, self.dim, name="LinearTargetFromObs"),
            )
            / TARGET_FINITE_DIFFERENCE_EPSILON
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "vmr_background": None
            if self.vmr_background is None
            else np.asarray(self.vmr_background).tolist(),
            "pattern": self.pattern
            if isinstance(self.pattern, str)
            else np.asarray(self.pattern).tolist(),
            "obs_model": callable_name(self.obs_model),
            "pixel": self.pixel,
            "dim": self.dim,
        }


class NonlinearTargetFromObs(Operator):
    """Finite-amplitude target from a callable observation model.

    Like :class:`LinearTargetFromObs` but with a caller-chosen finite
    perturbation ``amplitude`` and no division by it: the returned
    ``(c,)`` signature is the raw response difference, which retains the
    observation model's nonlinearity at that amplitude. Accepts a
    ``GeoArray`` or plain ``np.ndarray`` cube and returns a plain
    vector (not a carrier).

    Args:
        obs_model: Callable mapping a state cube to a spectral response.
            Must be callable.
        vmr_background: Optional explicit background state; the incoming
            cube is used when ``None``.
        amplitude: Finite perturbation amplitude. Default ``1.0``.
        pattern: Perturbation pattern — ``"uniform"``, ``"impulse"``, or
            an explicit array broadcastable to the background state.
        pixel: Spatial index of the impulse for ``pattern="impulse"``;
            the first flat pixel when ``None``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        obs_model: Any,
        vmr_background: np.ndarray | None = None,
        amplitude: float = 1.0,
        pattern: str | np.ndarray = "uniform",
        pixel: tuple[int, int] | None = None,
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.obs_model = obs_model
        self.vmr_background = vmr_background
        self.amplitude = amplitude
        self.pattern = pattern
        self.pixel = pixel
        self.dim = band_dim(dim, axis, "NonlinearTargetFromObs")

    def _apply(self, gt: GeoArray | np.ndarray) -> np.ndarray:
        return _target_from_obs(
            self.obs_model,
            gt,
            vmr_background=self.vmr_background,
            pattern=self.pattern,
            pixel=self.pixel,
            amplitude=self.amplitude,
            axis=dim_axis(gt, self.dim, name="NonlinearTargetFromObs"),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "vmr_background": None
            if self.vmr_background is None
            else np.asarray(self.vmr_background).tolist(),
            "amplitude": self.amplitude,
            "pattern": self.pattern
            if isinstance(self.pattern, str)
            else np.asarray(self.pattern).tolist(),
            "obs_model": callable_name(self.obs_model),
            "pixel": self.pixel,
            "dim": self.dim,
        }


class ColumnEnhancement(Operator):
    """Convenience trace-gas column-enhancement matched-filter composite.

    Chains :class:`EstimateMean`, a covariance estimator, an optional
    :class:`LinearTargetFromObs`, and :class:`MatchedFilter` into one
    call. With ``obs_model=None`` the target defaults to a uniform unit
    spectrum. Accepts a ``GeoArray`` or plain ``np.ndarray`` cube and
    returns the enhancement score map as the same carrier kind.

    Args:
        gas: Gas label recorded in the config (e.g. ``"CH4"``); metadata
            only, not used in the math.
        sensor: Sensor label recorded in the config (e.g. ``"EMIT"``);
            metadata only.
        obs_model: Optional callable observation model used to derive a
            linearised target; a uniform target is used when ``None``.
        mean_method: Background-mean estimator. Default ``"median"``.
        cov_method: Covariance estimator — ``"ledoit_wolf"``, ``"oas"``,
            or ``"lowrank"``. Default ``"ledoit_wolf"``.
        target_pattern: Perturbation pattern for target linearisation.
        rank: Rank for ``cov_method="lowrank"``; ``None`` means the
            default ``10``.
        tikhonov: Tikhonov regulariser for ``cov_method="lowrank"``;
            ``None`` means the default ``1e-3``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        gas: str = "CH4",
        sensor: str = "EMIT",
        obs_model: Any | None = None,
        mean_method: MeanMethod = "median",
        cov_method: CovShrinkageMethod | Literal["lowrank"] = "ledoit_wolf",
        target_pattern: str = "uniform",
        rank: int | None = None,
        tikhonov: float | None = None,
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.gas = gas
        self.sensor = sensor
        self.obs_model = reject_config_summary(obs_model, "obs_model")
        self.mean_method = mean_method
        self.cov_method = cov_method
        self.target_pattern = target_pattern
        self.rank = rank
        self.tikhonov = tikhonov
        self.dim = band_dim(dim, axis, "ColumnEnhancement")

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="ColumnEnhancement")
        _check_spectral_axis(gt, axis, type(self).__name__)
        mean = EstimateMean(method=self.mean_method, dim=self.dim)(gt)
        if self.cov_method == "lowrank":
            cov_op = EstimateCovLowRank(
                mean=mean,
                rank=10 if self.rank is None else self.rank,
                tikhonov=1e-3 if self.tikhonov is None else self.tikhonov,
                dim=self.dim,
            )(gt)
        else:
            cov_op = EstimateCovShrunk(mean=mean, method=self.cov_method, dim=self.dim)(
                gt
            )
        if self.obs_model is None:
            target = np.ones_like(mean)
        else:
            target = LinearTargetFromObs(
                obs_model=self.obs_model,
                pattern=self.target_pattern,
                dim=self.dim,
            )(gt)
        return MatchedFilter(mean=mean, cov_op=cov_op, target=target, dim=self.dim)(gt)

    def get_config(self) -> dict[str, Any]:
        return {
            "gas": self.gas,
            "sensor": self.sensor,
            # A callable has no config form: the summary makes from_state
            # refuse the reload instead of silently using a uniform target.
            "obs_model": None
            if self.obs_model is None
            else {"callable": callable_name(self.obs_model)},
            "mean_method": self.mean_method,
            "cov_method": self.cov_method,
            "target_pattern": self.target_pattern,
            "rank": self.rank,
            "tikhonov": self.tikhonov,
            "dim": self.dim,
        }


def _cov_config(cov_op: NumpyLinearOperator | np.ndarray | None) -> Any:
    """Serialise a covariance operator (or raw matrix) for get_config."""
    if cov_op is None:
        return None
    if isinstance(cov_op, NumpyLinearOperator):
        return jsonable(cov_op.matrix)
    return jsonable(np.asarray(cov_op))


def _check_spectral_axis(gt: Any, axis: int, name: str) -> None:
    """Raise a clear ``ValueError`` when ``axis`` is not an axis of ``gt``.

    The default spectral axis ``-3`` needs a ``(C, H, W)`` cube or a
    ``(T, C, H, W)`` stack: a 2-D ``(H, W)`` map has no band axis, and its
    rows must not be read as bands. A 2-D ``(N, C)`` sample matrix is
    accepted with an explicit ``axis`` (e.g. ``axis=1``).
    """
    ndim = np.ndim(gt)
    if not -ndim <= axis < ndim:
        raise ValueError(
            f"{name}: spectral axis {axis} does not exist on a {ndim}-D input "
            f"of shape {np.shape(gt)}; pass a (C, H, W) cube or a (T, C, H, W) "
            "stack (or set `axis` for a sample matrix)"
        )


def _mask_invalid(
    gt: GeoArray | np.ndarray, axis: int, name: str = "matched filter"
) -> tuple[np.ndarray, np.ndarray | None]:
    """Cube with nodata pixels set to NaN in every band, plus the validity map.

    A pixel is invalid when any band along ``axis`` is non-finite or equals
    the carrier's ``fill_value``. Returns ``(np.asarray(gt), None)``
    unchanged when every pixel is valid (the fast path); otherwise a float
    copy with invalid pixels NaN and the boolean validity map over the
    non-band axes (the score-map shape of :func:`apply_image`).
    """
    _check_spectral_axis(gt, axis, name)
    cube = np.asarray(gt)
    invalid = invalid_values(gt)
    if cube.ndim < 2 or not invalid.any():
        return cube, None
    invalid_pixels = invalid.any(axis=axis, keepdims=True)
    out = np.array(cube, dtype=np.result_type(cube.dtype, np.float32), copy=True)
    out[np.broadcast_to(invalid_pixels, out.shape)] = np.nan
    return out, ~np.squeeze(invalid_pixels, axis=axis)


def _restore(
    gt: GeoArray | np.ndarray, out: np.ndarray, valid: np.ndarray | None
) -> np.ndarray:
    """Write ``NaN`` into invalid pixels of a score map."""
    if valid is None:
        return out
    return restore_fill(out, valid, np.nan)


def _target_from_obs(
    obs_model: Any,
    gt: GeoArray | np.ndarray,
    *,
    vmr_background: np.ndarray | None,
    pattern: str | np.ndarray,
    pixel: tuple[int, int] | None,
    amplitude: float,
    axis: int = -3,
) -> np.ndarray:
    if not callable(obs_model):
        raise TypeError("obs_model must be callable for the NumPy target wrappers")
    base = (
        _mask_invalid(gt, axis)[0]
        if vmr_background is None
        else np.asarray(vmr_background, dtype=float)
    )
    perturb = _target_pattern(base, pattern=pattern, pixel=pixel, axis=axis) * amplitude
    y0 = np.asarray(obs_model(base), dtype=float)
    y1 = np.asarray(obs_model(base + perturb), dtype=float)
    target = y1 - y0
    if target.ndim > 1:
        # Average the response over valid (finite) pixels only.
        samples, _ = cube_to_samples(target, band_axis=axis, dtype=float)
        finite = np.isfinite(samples).all(axis=1)
        return np.mean(samples if finite.all() else samples[finite], axis=0)
    return target.reshape(-1)


def _target_pattern(
    base: np.ndarray,
    *,
    pattern: str | np.ndarray,
    pixel: tuple[int, int] | None,
    axis: int = -3,
) -> np.ndarray:
    if not isinstance(pattern, str):
        return np.asarray(pattern, dtype=float)
    out = np.zeros_like(np.asarray(base), dtype=float)
    if pattern == "uniform":
        out[...] = 1.0
    elif pattern == "impulse":
        if pixel is None:
            flat = out.reshape(-1)
            flat[0] = 1.0
        else:
            # Insert the spectral axis as a full slice at position ``axis`` and
            # use ``pixel`` to index the remaining spatial axes in order.
            index: list[Any] = list(pixel)
            index.insert(axis % out.ndim, slice(None))
            out[tuple(index)] = 1.0
    else:
        raise ValueError(f"unknown target pattern {pattern!r}")
    return out
