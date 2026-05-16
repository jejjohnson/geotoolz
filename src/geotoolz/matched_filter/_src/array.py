"""Pure NumPy matched-filter primitives for hyperspectral cubes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from scipy import stats


MeanMethod = Literal["mean", "median", "trimmed", "huber"]
CovShrinkageMethod = Literal["ledoit_wolf", "oas"]


@dataclass(frozen=True)
class NumpyLinearOperator:
    """Dense NumPy linear operator with a small ``solve`` interface.

    Args:
        matrix: Square covariance matrix.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        mat = np.asarray(self.matrix, dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("matrix must be a square 2-D array")
        object.__setattr__(self, "matrix", mat)

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape."""
        return self.matrix.shape

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ``matrix @ x = rhs``."""
        return np.linalg.solve(self.matrix, np.asarray(rhs, dtype=float))


@dataclass(frozen=True)
class ClusterBackground:
    """Cluster labels with per-cluster matched-filter background statistics."""

    labels: np.ndarray
    means: np.ndarray
    cov_ops: tuple[NumpyLinearOperator, ...]


@dataclass(frozen=True)
class AdaptiveBackground:
    """Local diagonal background statistics for each pixel."""

    mean: np.ndarray
    variance: np.ndarray


@dataclass
class WelfordAccumulator:
    """Streaming mean/covariance accumulator using Chan-Welford updates."""

    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls, n_features: int) -> WelfordAccumulator:
        """Create an empty accumulator for ``n_features`` spectral bands."""
        return cls(
            count=0,
            mean=np.zeros(n_features, dtype=float),
            m2=np.zeros((n_features, n_features), dtype=float),
        )

    def update(self, values: np.ndarray) -> None:
        """Update the accumulator with rows shaped ``(n_samples, n_features)``."""
        x = _as_2d_samples(values)
        if x.size == 0:
            return
        other = WelfordAccumulator.from_values(x)
        self.merge(other)

    def merge(self, other: WelfordAccumulator) -> None:
        """Merge another accumulator into this one."""
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.copy()
            self.m2 = other.m2.copy()
            return
        delta = other.mean - self.mean
        total = self.count + other.count
        self.m2 = (
            self.m2
            + other.m2
            + np.outer(delta, delta) * (self.count * other.count / total)
        )
        self.mean = self.mean + delta * (other.count / total)
        self.count = total

    @classmethod
    def from_values(cls, values: np.ndarray) -> WelfordAccumulator:
        """Create an accumulator from sample rows."""
        x = _as_2d_samples(values)
        if x.shape[0] == 0:
            return cls.empty(x.shape[1])
        mean = np.mean(x, axis=0)
        centered = x - mean
        return cls(count=x.shape[0], mean=mean, m2=centered.T @ centered)

    def covariance(self, *, ddof: int = 1, ridge: float = 0.0) -> np.ndarray:
        """Return the sample covariance matrix."""
        if self.count <= ddof:
            raise ValueError("at least two samples are required for covariance")
        cov = self.m2 / (self.count - ddof)
        if ridge:
            cov = cov + float(ridge) * np.eye(cov.shape[0], dtype=float)
        return cov


def cube_to_samples(
    cube: np.ndarray, *, axis: int = 0
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Move a spectral cube to ``(pixels, bands)`` samples."""
    arr = np.asarray(cube, dtype=float)
    if arr.ndim < 2:
        raise ValueError("matched-filter input must have at least two dimensions")
    moved = np.moveaxis(arr, axis, -1)
    spatial_shape = moved.shape[:-1]
    return moved.reshape(-1, moved.shape[-1]), spatial_shape


def estimate_mean(
    cube: np.ndarray,
    *,
    method: MeanMethod = "mean",
    trim_proportion: float = 0.1,
    huber_c: float = 1.345,
    axis: int = 0,
) -> np.ndarray:
    """Estimate a spectral background mean from a cube."""
    x, _ = cube_to_samples(cube, axis=axis)
    if method == "mean":
        return np.mean(x, axis=0)
    if method == "median":
        return np.median(x, axis=0)
    if method == "trimmed":
        if not 0 <= trim_proportion < 0.5:
            raise ValueError("trim_proportion must satisfy 0 <= p < 0.5")
        return stats.trim_mean(x, proportiontocut=trim_proportion, axis=0)
    if method == "huber":
        return _huber_mean(x, c=huber_c)
    raise ValueError(f"unknown mean method {method!r}")


def estimate_cov_empirical(
    cube: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    ridge: float = 0.0,
    axis: int = 0,
) -> NumpyLinearOperator:
    """Estimate an empirical covariance operator from a cube."""
    x, _ = cube_to_samples(cube, axis=axis)
    mu = np.mean(x, axis=0) if mean is None else _as_vector(mean, x.shape[1], "mean")
    centered = x - mu
    denom = max(x.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    if ridge:
        cov = cov + float(ridge) * np.eye(cov.shape[0], dtype=float)
    return NumpyLinearOperator(cov)


def estimate_cov_shrunk(
    cube: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    method: CovShrinkageMethod = "ledoit_wolf",
    axis: int = 0,
) -> NumpyLinearOperator:
    """Estimate a diagonal-target shrinkage covariance operator."""
    empirical = estimate_cov_empirical(cube, mean=mean, axis=axis).matrix
    n_features = empirical.shape[0]
    target = np.trace(empirical) / n_features * np.eye(n_features, dtype=float)
    if method == "oas":
        n_samples = cube_to_samples(cube, axis=axis)[0].shape[0]
        trace = np.trace(empirical)
        trace_sq = trace * trace
        fro_sq = float(np.sum(empirical * empirical))
        denom = (n_samples + 1) * (fro_sq - trace_sq / n_features)
        shrinkage = (
            1.0
            if denom <= 0
            else min(
                1.0,
                ((1 - 2 / n_features) * fro_sq + trace_sq) / denom,
            )
        )
    elif method == "ledoit_wolf":
        diag_energy = float(np.sum(np.diag(empirical) ** 2))
        total_energy = float(np.sum(empirical * empirical))
        shrinkage = 0.0 if total_energy == 0 else min(1.0, diag_energy / total_energy)
    else:
        raise ValueError(f"unknown covariance shrinkage method {method!r}")
    return NumpyLinearOperator((1.0 - shrinkage) * empirical + shrinkage * target)


def estimate_cov_lowrank(
    cube: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    rank: int = 10,
    tikhonov: float = 1e-3,
    axis: int = 0,
) -> NumpyLinearOperator:
    """Estimate a low-rank-plus-Tikhonov dense covariance operator."""
    empirical = estimate_cov_empirical(cube, mean=mean, axis=axis).matrix
    if rank < 1:
        raise ValueError("rank must be positive")
    u, s, _ = np.linalg.svd(empirical, full_matrices=False)
    k = min(rank, s.shape[0])
    cov = (u[:, :k] * s[:k]) @ u[:, :k].T
    cov = cov + float(tikhonov) * np.eye(cov.shape[0], dtype=float)
    return NumpyLinearOperator(cov)


def apply_pixel(
    pixel: np.ndarray,
    *,
    mean: np.ndarray,
    cov_op: NumpyLinearOperator | np.ndarray,
    target: np.ndarray,
) -> float:
    """Apply the scalar matched-filter score to one pixel."""
    mean_vec = _as_vector(mean, np.asarray(pixel).shape[0], "mean")
    target_vec = _as_vector(target, mean_vec.shape[0], "target")
    solved_target = solve(cov_op, target_vec)
    denom = float(target_vec @ solved_target)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("target/covariance produce a non-positive MF denominator")
    return float((np.asarray(pixel, dtype=float) - mean_vec) @ solved_target / denom)


def apply_image(
    cube: np.ndarray,
    *,
    mean: np.ndarray,
    cov_op: NumpyLinearOperator | np.ndarray,
    target: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Apply a matched filter over a hyperspectral image cube."""
    x, spatial_shape = cube_to_samples(cube, axis=axis)
    mean_vec = _as_vector(mean, x.shape[1], "mean")
    target_vec = _as_vector(target, x.shape[1], "target")
    solved_target = solve(cov_op, target_vec)
    denom = float(target_vec @ solved_target)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("target/covariance produce a non-positive MF denominator")
    scores = (x - mean_vec) @ solved_target / denom
    return scores.reshape(spatial_shape)


def matched_filter_snr(
    *, amplitude: float, cov_op: NumpyLinearOperator | np.ndarray, target: np.ndarray
) -> float:
    """Return theoretical SNR ``amplitude * sqrt(t^T Sigma^-1 t)``."""
    t = np.asarray(target, dtype=float).reshape(-1)
    gain = float(t @ solve(cov_op, t))
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("target/covariance produce a non-positive MF gain")
    return float(amplitude) * float(np.sqrt(gain))


def detection_threshold(
    *,
    false_alarm_rate: float,
    cov_op: NumpyLinearOperator | np.ndarray,
    target: np.ndarray,
) -> float:
    """Return a Gaussian false-alarm score threshold."""
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false_alarm_rate must be between 0 and 1")
    gain = matched_filter_snr(amplitude=1.0, cov_op=cov_op, target=target)
    return float(stats.norm.ppf(1.0 - false_alarm_rate) / gain)


def validate_mf_inputs(
    *, cov_op: NumpyLinearOperator | np.ndarray, target: np.ndarray
) -> None:
    """Raise ``ValueError`` for degenerate target/covariance pairs."""
    t = np.asarray(target, dtype=float).reshape(-1)
    if t.size == 0 or not np.any(t):
        raise ValueError("target must contain at least one non-zero value")
    try:
        gain = float(t @ solve(cov_op, t))
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance operator must be non-singular") from exc
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("target/covariance produce a non-positive MF gain")


def solve(cov_op: NumpyLinearOperator | np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a covariance system for either supported operator representation."""
    if isinstance(cov_op, NumpyLinearOperator):
        return cov_op.solve(rhs)
    return np.linalg.solve(
        np.asarray(cov_op, dtype=float), np.asarray(rhs, dtype=float)
    )


def gmm_cluster_background(
    cube: np.ndarray,
    *,
    n_clusters: int,
    cov_estimator: Literal["empirical", "ledoit_wolf", "oas"] = "ledoit_wolf",
    random_state: int | None = 0,
    axis: int = 0,
) -> ClusterBackground:
    """Estimate a deterministic NumPy k-means cluster background."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    x, spatial_shape = cube_to_samples(cube, axis=axis)
    labels = _kmeans_labels(x, n_clusters=n_clusters, random_state=random_state)
    means = np.empty((n_clusters, x.shape[1]), dtype=float)
    cov_ops: list[NumpyLinearOperator] = []
    for k in range(n_clusters):
        group = x[labels == k]
        if group.shape[0] == 0:
            group = x
        means[k] = np.mean(group, axis=0)
        if cov_estimator == "empirical":
            cov = _cov_from_samples(group, means[k], ridge=1e-8)
        elif cov_estimator in {"ledoit_wolf", "oas"}:
            cov = _shrunk_cov_from_samples(group, means[k], method=cov_estimator)
        else:
            raise ValueError(f"unknown covariance estimator {cov_estimator!r}")
        cov_ops.append(NumpyLinearOperator(cov))
    return ClusterBackground(
        labels=labels.reshape(spatial_shape), means=means, cov_ops=tuple(cov_ops)
    )


def apply_cluster_mf(
    cube: np.ndarray,
    *,
    cluster: ClusterBackground,
    target: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Apply matched filtering with per-cluster background statistics."""
    x, spatial_shape = cube_to_samples(cube, axis=axis)
    labels = np.asarray(cluster.labels).reshape(-1)
    if labels.shape[0] != x.shape[0]:
        raise ValueError("cluster labels must match cube spatial shape")
    scores = np.empty(x.shape[0], dtype=float)
    for k, cov_op in enumerate(cluster.cov_ops):
        mask = labels == k
        if np.any(mask):
            scores[mask] = [
                apply_pixel(pixel, mean=cluster.means[k], cov_op=cov_op, target=target)
                for pixel in x[mask]
            ]
    return scores.reshape(spatial_shape)


def adaptive_window_background(
    cube: np.ndarray,
    *,
    window_size: int = 7,
    pad_mode: str = "reflect",
    axis: int = 0,
) -> AdaptiveBackground:
    """Estimate local mean and diagonal variance over square windows."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    arr = np.moveaxis(np.asarray(cube, dtype=float), axis, -1)
    half = window_size // 2
    padded = np.pad(
        arr,
        [(half, half), (half, half), (0, 0)],
        mode=cast(Any, pad_mode),
    )
    mean = np.empty_like(arr, dtype=float)
    variance = np.empty_like(arr, dtype=float)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            window = padded[row : row + window_size, col : col + window_size, :]
            flat = window.reshape(-1, arr.shape[-1])
            mean[row, col] = np.mean(flat, axis=0)
            variance[row, col] = np.var(flat, axis=0, ddof=1)
    return AdaptiveBackground(
        mean=np.moveaxis(mean, -1, axis), variance=np.moveaxis(variance, -1, axis)
    )


def _as_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    vec = np.asarray(values, dtype=float).reshape(-1)
    if vec.shape[0] != size:
        raise ValueError(
            f"{name} length {vec.shape[0]} does not match band count {size}"
        )
    return vec


def _as_2d_samples(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("values must be a 1-D vector or 2-D sample matrix")
    return arr


def _huber_mean(
    values: np.ndarray, *, c: float, max_iter: int = 50, tol: float = 1e-6
) -> np.ndarray:
    if c <= 0:
        raise ValueError("huber_c must be positive")
    mu = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - mu), axis=0)
    scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
    for _ in range(max_iter):
        z = (values - mu) / scale
        weights = np.minimum(1.0, c / np.maximum(np.abs(z), np.finfo(float).eps))
        next_mu = np.sum(weights * values, axis=0) / np.sum(weights, axis=0)
        if np.max(np.abs(next_mu - mu)) < tol:
            return next_mu
        mu = next_mu
    return mu


def _cov_from_samples(
    values: np.ndarray, mean: np.ndarray, *, ridge: float = 0.0
) -> np.ndarray:
    centered = values - mean
    denom = max(values.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    if ridge:
        cov = cov + ridge * np.eye(cov.shape[0], dtype=float)
    return cov


def _shrunk_cov_from_samples(
    values: np.ndarray,
    mean: np.ndarray,
    *,
    method: Literal["ledoit_wolf", "oas"],
) -> np.ndarray:
    cov = _cov_from_samples(values, mean, ridge=1e-8)
    target = np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0], dtype=float)
    shrinkage = 0.2 if method == "ledoit_wolf" else 0.5
    return (1.0 - shrinkage) * cov + shrinkage * target


def _kmeans_labels(
    values: np.ndarray,
    *,
    n_clusters: int,
    random_state: int | None,
    max_iter: int = 50,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if values.shape[0] < n_clusters:
        raise ValueError("n_clusters cannot exceed number of pixels")
    centers = values[rng.choice(values.shape[0], size=n_clusters, replace=False)].copy()
    labels = np.zeros(values.shape[0], dtype=np.intp)
    for _ in range(max_iter):
        distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for k in range(n_clusters):
            if np.any(labels == k):
                centers[k] = np.mean(values[labels == k], axis=0)
    return labels
