"""Carrier-aware matched-filter Operators."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pipekit import Operator
from geotoolz.matched_filter._src.array import (
    AdaptiveBackground,
    ClusterBackground,
    CovMethod,
    CovShrinkageMethod,
    MeanMethod,
    NumpyLinearOperator,
    StreamingBackgroundResult,
    WelfordAccumulator,
    adaptive_window_background,
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
    from georeader.geotensor import GeoTensor


# Small relative perturbation for finite-difference target linearization; callers
# needing scale-specific control can use NonlinearTargetFromObs(amplitude=...).
TARGET_FINITE_DIFFERENCE_EPSILON = 1e-4


class MatchedFilter(Operator):
    """Per-pixel matched-filter score map.

    Computes ``y = (x - μ)^T Σ^-1 t / (t^T Σ^-1 t)`` and returns a
    single-band ``GeoTensor`` that preserves the input georeferencing.
    """

    def __init__(
        self,
        *,
        mean: np.ndarray | None = None,
        cov_op: NumpyLinearOperator | np.ndarray | None = None,
        target: np.ndarray | None = None,
        fit_on_call: bool = False,
        mean_method: MeanMethod = "median",
        cov_method: CovMethod = "ledoit_wolf",
        axis: int = 0,
    ) -> None:
        self.mean = mean
        self.cov_op = cov_op
        self.target = target
        self.fit_on_call = fit_on_call
        self.mean_method = mean_method
        self.cov_method = cov_method
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        mean = self.mean
        cov_op = self.cov_op
        if self.fit_on_call or mean is None or cov_op is None:
            mean = estimate_mean(
                np.asarray(gt), method=self.mean_method, axis=self.axis
            )
            if self.cov_method == "empirical":
                cov_op = estimate_cov_empirical(
                    np.asarray(gt), mean=mean, ridge=1e-8, axis=self.axis
                )
            elif self.cov_method == "lowrank":
                cov_op = estimate_cov_lowrank(np.asarray(gt), mean=mean, axis=self.axis)
            else:
                cov_op = estimate_cov_shrunk(
                    np.asarray(gt), mean=mean, method=self.cov_method, axis=self.axis
                )
            self.mean = mean
            self.cov_op = cov_op
        if self.target is None:
            raise ValueError("target must be supplied before applying MatchedFilter")
        out = apply_image(
            np.asarray(gt), mean=mean, cov_op=cov_op, target=self.target, axis=self.axis
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean is None else np.asarray(self.mean).tolist(),
            "cov_op": _cov_config(self.cov_op),
            "target": None if self.target is None else np.asarray(self.target).tolist(),
            "fit_on_call": self.fit_on_call,
            "mean_method": self.mean_method,
            "cov_method": self.cov_method,
            "axis": self.axis,
        }


class MatchedFilterPixel(Operator):
    """Single-pixel matched-filter score."""

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
    """Theoretical detection SNR ``amplitude * sqrt(t^T Sigma^-1 t)``."""

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

    def _apply(self) -> float:
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
    """Gaussian false-alarm score threshold."""

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

    def _apply(self) -> float:
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
    """Fail-fast pass-through validator for matched-filter inputs."""

    def __init__(
        self,
        *,
        cov_op: NumpyLinearOperator | np.ndarray,
        target: np.ndarray,
    ) -> None:
        self.cov_op = cov_op
        self.target = target

    def _apply(self, value: Any = None) -> Any:
        validate_mf_inputs(cov_op=self.cov_op, target=self.target)
        return value

    def get_config(self) -> dict[str, Any]:
        return {
            "cov_op": _cov_config(self.cov_op),
            "target": np.asarray(self.target).tolist(),
        }


class EstimateMean(Operator):
    """Estimate a spectral background mean vector from a cube."""

    def __init__(
        self,
        *,
        method: MeanMethod = "mean",
        trim_proportion: float = 0.1,
        huber_c: float = 1.345,
        axis: int = 0,
    ) -> None:
        self.method = method
        self.trim_proportion = trim_proportion
        self.huber_c = huber_c
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> np.ndarray:
        return estimate_mean(
            np.asarray(gt),
            method=self.method,
            trim_proportion=self.trim_proportion,
            huber_c=self.huber_c,
            axis=self.axis,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "trim_proportion": self.trim_proportion,
            "huber_c": self.huber_c,
            "axis": self.axis,
        }


class EstimateCovEmpirical(Operator):
    """Estimate an empirical covariance operator."""

    def __init__(
        self, *, mean: np.ndarray | None = None, ridge: float = 0.0, axis: int = 0
    ) -> None:
        self.mean = mean
        self.ridge = ridge
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> NumpyLinearOperator:
        return estimate_cov_empirical(
            np.asarray(gt), mean=self.mean, ridge=self.ridge, axis=self.axis
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean is None else np.asarray(self.mean).tolist(),
            "ridge": self.ridge,
            "axis": self.axis,
        }


class EstimateCovShrunk(Operator):
    """Estimate a diagonal-target shrinkage covariance operator."""

    def __init__(
        self,
        *,
        mean: np.ndarray | None = None,
        method: CovShrinkageMethod = "ledoit_wolf",
        axis: int = 0,
    ) -> None:
        self.mean = mean
        self.method = method
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> NumpyLinearOperator:
        return estimate_cov_shrunk(
            np.asarray(gt), mean=self.mean, method=self.method, axis=self.axis
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean is None else np.asarray(self.mean).tolist(),
            "method": self.method,
            "axis": self.axis,
        }


class EstimateCovLowRank(Operator):
    """Estimate a low-rank-plus-Tikhonov covariance operator."""

    def __init__(
        self,
        *,
        mean: np.ndarray | None = None,
        rank: int = 10,
        tikhonov: float = 1e-3,
        random_state: int | None = 0,
        n_oversamples: int = 10,
        axis: int = 0,
    ) -> None:
        self.mean = mean
        self.rank = rank
        self.tikhonov = tikhonov
        self.random_state = random_state
        self.n_oversamples = n_oversamples
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> NumpyLinearOperator:
        return estimate_cov_lowrank(
            np.asarray(gt),
            mean=self.mean,
            rank=self.rank,
            tikhonov=self.tikhonov,
            random_state=self.random_state,
            n_oversamples=self.n_oversamples,
            axis=self.axis,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean is None else np.asarray(self.mean).tolist(),
            "rank": self.rank,
            "tikhonov": self.tikhonov,
            "random_state": self.random_state,
            "n_oversamples": self.n_oversamples,
            "axis": self.axis,
        }


class GMMClusterBackground(Operator):
    """Estimate clustered background statistics with a NumPy k-means backend."""

    def __init__(
        self,
        *,
        n_clusters: int,
        cov_estimator: Literal["empirical", "ledoit_wolf", "oas"] = "ledoit_wolf",
        random_state: int | None = 0,
        bayesian: bool = False,
        axis: int = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.cov_estimator = cov_estimator
        self.random_state = random_state
        self.bayesian = bayesian
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> ClusterBackground:
        return gmm_cluster_background(
            np.asarray(gt),
            n_clusters=self.n_clusters,
            cov_estimator=self.cov_estimator,
            random_state=self.random_state,
            bayesian=self.bayesian,
            axis=self.axis,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "cov_estimator": self.cov_estimator,
            "random_state": self.random_state,
            "bayesian": self.bayesian,
            "axis": self.axis,
        }


class AdaptiveWindowBackground(Operator):
    """Per-pixel local mean and diagonal variance over a square window."""

    def __init__(
        self, *, window_size: int = 7, pad_mode: str = "reflect", axis: int = 0
    ) -> None:
        self.window_size = window_size
        self.pad_mode = pad_mode
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> AdaptiveBackground:
        return adaptive_window_background(
            np.asarray(gt),
            window_size=self.window_size,
            pad_mode=self.pad_mode,
            axis=self.axis,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "pad_mode": self.pad_mode,
            "axis": self.axis,
        }


class ApplyClusterMF(Operator):
    """Apply a matched filter dispatched by per-pixel cluster label."""

    def __init__(
        self,
        *,
        cluster: ClusterBackground,
        target: np.ndarray,
        axis: int = 0,
    ) -> None:
        self.cluster = cluster
        self.target = target
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = apply_cluster_mf(
            np.asarray(gt), cluster=self.cluster, target=self.target, axis=self.axis
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"target": np.asarray(self.target).tolist(), "axis": self.axis}


class StreamingBackground(Operator):
    """Aggregate mean/covariance across an iterable of cubes."""

    def __init__(
        self,
        *,
        cubes: Iterable[GeoTensor],
        cov_kind: Literal["empirical", "shrunk"] = "shrunk",
        axis: int = 0,
    ) -> None:
        self.cubes = cubes
        self.cov_kind = cov_kind
        self.axis = axis

    def _apply(self) -> StreamingBackgroundResult:
        acc: WelfordAccumulator | None = None
        for cube in self.cubes:
            samples, _ = _cube_to_samples(np.asarray(cube), axis=self.axis)
            if acc is None:
                acc = WelfordAccumulator.empty(samples.shape[1])
            acc.update(samples)
        if acc is None:
            raise ValueError("cubes must contain at least one cube")
        cov = acc.covariance(ridge=1e-8)
        if self.cov_kind == "shrunk":
            cov = shrink_covariance(cov, method="ledoit_wolf", n_samples=acc.count)
        elif self.cov_kind != "empirical":
            raise ValueError(f"unknown cov_kind {self.cov_kind!r}")
        return StreamingBackgroundResult(mean=acc.mean, cov_op=NumpyLinearOperator(cov))

    def get_config(self) -> dict[str, Any]:
        return {"cov_kind": self.cov_kind, "axis": self.axis}


class LinearTargetFromObs(Operator):
    """Finite-difference tangent-linear target from a callable observation model."""

    def __init__(
        self,
        *,
        obs_model: Any,
        vmr_background: np.ndarray | None = None,
        pattern: str | np.ndarray = "uniform",
        pixel: tuple[int, int] | None = None,
        linear_forward: bool = False,
    ) -> None:
        self.obs_model = obs_model
        self.vmr_background = vmr_background
        self.pattern = pattern
        self.pixel = pixel
        self.linear_forward = linear_forward

    def _apply(self, gt: GeoTensor) -> np.ndarray:
        return (
            _target_from_obs(
                self.obs_model,
                gt,
                vmr_background=self.vmr_background,
                pattern=self.pattern,
                pixel=self.pixel,
                amplitude=TARGET_FINITE_DIFFERENCE_EPSILON,
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
            "pixel": self.pixel,
            "linear_forward": self.linear_forward,
        }


class NonlinearTargetFromObs(Operator):
    """Finite-amplitude target from a callable observation model."""

    def __init__(
        self,
        *,
        obs_model: Any,
        vmr_background: np.ndarray | None = None,
        amplitude: float = 1.0,
        pattern: str | np.ndarray = "uniform",
        pixel: tuple[int, int] | None = None,
    ) -> None:
        self.obs_model = obs_model
        self.vmr_background = vmr_background
        self.amplitude = amplitude
        self.pattern = pattern
        self.pixel = pixel

    def _apply(self, gt: GeoTensor) -> np.ndarray:
        return _target_from_obs(
            self.obs_model,
            gt,
            vmr_background=self.vmr_background,
            pattern=self.pattern,
            pixel=self.pixel,
            amplitude=self.amplitude,
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
            "pixel": self.pixel,
        }


class ColumnEnhancement(Operator):
    """Convenience trace-gas column-enhancement matched-filter composite."""

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
        axis: int = 0,
    ) -> None:
        self.gas = gas
        self.sensor = sensor
        self.obs_model = obs_model
        self.mean_method = mean_method
        self.cov_method = cov_method
        self.target_pattern = target_pattern
        self.rank = rank
        self.tikhonov = tikhonov
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        mean = EstimateMean(method=self.mean_method, axis=self.axis)(gt)
        if self.cov_method == "lowrank":
            cov_op = EstimateCovLowRank(
                mean=mean,
                rank=10 if self.rank is None else self.rank,
                tikhonov=1e-3 if self.tikhonov is None else self.tikhonov,
                axis=self.axis,
            )(gt)
        else:
            cov_op = EstimateCovShrunk(
                mean=mean, method=self.cov_method, axis=self.axis
            )(gt)
        if self.obs_model is None:
            target = np.ones_like(mean)
        else:
            target = LinearTargetFromObs(
                obs_model=self.obs_model, pattern=self.target_pattern
            )(gt)
        return MatchedFilter(mean=mean, cov_op=cov_op, target=target, axis=self.axis)(
            gt
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "gas": self.gas,
            "sensor": self.sensor,
            "mean_method": self.mean_method,
            "cov_method": self.cov_method,
            "target_pattern": self.target_pattern,
            "rank": self.rank,
            "tikhonov": self.tikhonov,
            "axis": self.axis,
        }


def _cov_config(cov_op: NumpyLinearOperator | np.ndarray | None) -> Any:
    if cov_op is None:
        return None
    if isinstance(cov_op, NumpyLinearOperator):
        return cov_op.matrix.tolist()
    return np.asarray(cov_op).tolist()


def _cube_to_samples(
    cube: np.ndarray, *, axis: int
) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(cube, dtype=float)
    moved = np.moveaxis(arr, axis, -1)
    return moved.reshape(-1, moved.shape[-1]), moved.shape[:-1]


def _target_from_obs(
    obs_model: Any,
    gt: GeoTensor,
    *,
    vmr_background: np.ndarray | None,
    pattern: str | np.ndarray,
    pixel: tuple[int, int] | None,
    amplitude: float,
) -> np.ndarray:
    if not callable(obs_model):
        raise TypeError("obs_model must be callable for the NumPy target wrappers")
    base = np.asarray(gt if vmr_background is None else vmr_background, dtype=float)
    perturb = _target_pattern(base, pattern=pattern, pixel=pixel) * amplitude
    y0 = np.asarray(obs_model(base), dtype=float)
    y1 = np.asarray(obs_model(base + perturb), dtype=float)
    target = y1 - y0
    if target.ndim > 1:
        samples, _ = _cube_to_samples(target, axis=0)
        return np.mean(samples, axis=0)
    return target.reshape(-1)


def _target_pattern(
    base: np.ndarray,
    *,
    pattern: str | np.ndarray,
    pixel: tuple[int, int] | None,
) -> np.ndarray:
    if not isinstance(pattern, str):
        return np.asarray(pattern, dtype=float)
    out = np.zeros_like(base, dtype=float)
    if pattern == "uniform":
        out[...] = 1.0
    elif pattern == "impulse":
        if pixel is None:
            flat = out.reshape(-1)
            flat[0] = 1.0
        else:
            out[(slice(None), *pixel)] = 1.0
    else:
        raise ValueError(f"unknown target pattern {pattern!r}")
    return out
