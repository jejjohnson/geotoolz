"""Pure NumPy matched-filter primitives for hyperspectral cubes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage, stats


MeanMethod = Literal["mean", "median", "trimmed", "huber"]
CovShrinkageMethod = Literal["ledoit_wolf", "oas"]
CovMethod = Literal["empirical", "ledoit_wolf", "oas", "lowrank"]
MAD_NORMAL_SCALE = 1.4826
GMM_VARIANCE_RIDGE = 1e-6


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


@dataclass(frozen=True)
class StreamingBackgroundResult:
    """Mean and covariance operator estimated from streamed cubes."""

    mean: np.ndarray
    cov_op: NumpyLinearOperator


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
    # Vectorise the cube once and reuse the sample matrix for both the
    # empirical covariance and the sample count that shrink_covariance needs.
    x, _ = cube_to_samples(cube, axis=axis)
    mu = np.mean(x, axis=0) if mean is None else _as_vector(mean, x.shape[1], "mean")
    centered = x - mu
    denom = max(x.shape[0] - 1, 1)
    empirical = centered.T @ centered / denom
    return NumpyLinearOperator(
        shrink_covariance(empirical, method=method, n_samples=x.shape[0])
    )


def shrink_covariance(
    empirical: np.ndarray,
    *,
    method: CovShrinkageMethod,
    n_samples: int,
) -> np.ndarray:
    """Shrink an empirical covariance toward a scaled-identity target."""
    cov = np.asarray(empirical, dtype=float)
    n_features = cov.shape[0]
    target = np.trace(cov) / n_features * np.eye(n_features, dtype=float)
    if method == "oas":
        trace = np.trace(cov)
        trace_sq = trace * trace
        fro_sq = float(np.sum(cov * cov))
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
        # Analytical Ledoit-Wolf approximation. The proper LW estimator
        # (Ledoit & Wolf 2004) requires the sample matrix X to estimate
        # b^2 = variance of the empirical-covariance entries; with only
        # the empirical covariance and the sample count available we use
        # the asymptotic estimate b^2 ~= (1/n) * (tr(S^2) + tr(S)^2 / p)
        # under iid samples, with d^2 = ||S - mu*I||_F^2 as the distance
        # to the scaled-identity target. Behaves correctly: no shrinkage
        # when S is already scaled-identity (d^2 = 0 -> shrinkage = 0),
        # vanishing shrinkage as n_samples grows, and full shrinkage when
        # n_samples is tiny.
        mu = float(np.trace(cov)) / n_features
        diff = cov - mu * np.eye(n_features, dtype=float)
        d_sq = float(np.sum(diff * diff))
        trace_cov_sq = float(np.sum(cov * cov))
        trace_sq = float(np.trace(cov)) ** 2
        b_sq = (trace_cov_sq + trace_sq / n_features) / max(n_samples, 1)
        shrinkage = 0.0 if d_sq == 0 else min(1.0, b_sq / d_sq)
    else:
        raise ValueError(f"unknown covariance shrinkage method {method!r}")
    return (1.0 - shrinkage) * cov + shrinkage * target


def estimate_cov_lowrank(
    cube: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    rank: int = 10,
    tikhonov: float = 1e-3,
    random_state: int | None = 0,
    n_oversamples: int = 10,
    axis: int = 0,
) -> NumpyLinearOperator:
    """Estimate a low-rank-plus-Tikhonov dense covariance operator."""
    empirical = estimate_cov_empirical(cube, mean=mean, axis=axis).matrix
    if rank < 1:
        raise ValueError("rank must be positive")
    if n_oversamples < 0:
        raise ValueError("n_oversamples must be non-negative")
    n_features = empirical.shape[0]
    sample_rank = min(rank + n_oversamples, n_features)
    rng = np.random.default_rng(random_state)
    omega = rng.normal(size=(n_features, sample_rank))
    basis, _ = np.linalg.qr(empirical @ omega, mode="reduced")
    u_hat, s, _ = np.linalg.svd(basis.T @ empirical, full_matrices=False)
    u = basis @ u_hat
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
    bayesian: bool = False,
    axis: int = 0,
) -> ClusterBackground:
    """Estimate a deterministic diagonal-GMM background."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    x, spatial_shape = cube_to_samples(cube, axis=axis)
    labels = _gmm_labels(
        x, n_clusters=n_clusters, random_state=random_state, bayesian=bayesian
    )
    n_active = int(labels.max()) + 1
    means = np.empty((n_active, x.shape[1]), dtype=float)
    cov_ops: list[NumpyLinearOperator] = []
    for k in range(n_active):
        group = x[labels == k]
        if group.shape[0] == 0:
            raise ValueError("GMM produced an empty cluster")
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
    target_vec = _as_vector(target, x.shape[1], "target")
    scores = np.empty(x.shape[0], dtype=float)
    for k, cov_op in enumerate(cluster.cov_ops):
        mask = labels == k
        if np.any(mask):
            solved_target = solve(cov_op, target_vec)
            denom = float(target_vec @ solved_target)
            if not np.isfinite(denom) or denom <= 0:
                raise ValueError(
                    "target/covariance produce a non-positive MF denominator"
                )
            scores[mask] = (x[mask] - cluster.means[k]) @ solved_target / denom
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
    if arr.ndim != 3:
        raise ValueError("adaptive windows require a 3-D cube")
    size = (window_size, window_size, 1)
    mean = ndimage.uniform_filter(arr, size=size, mode=pad_mode)
    mean_sq = ndimage.uniform_filter(arr * arr, size=size, mode=pad_mode)
    n_window = window_size * window_size
    variance = np.maximum(mean_sq - mean * mean, 0.0) * n_window / max(n_window - 1, 1)
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
    """Return a per-band robust Huber mean.

    ``c`` is the Huber tuning constant in robust z-score units. Smaller
    values down-weight outliers more aggressively. If convergence is not
    reached within ``max_iter`` iterations, the last iterate is returned.
    """
    if c <= 0:
        raise ValueError("huber_c must be positive")
    mu = np.median(values, axis=0)
    scale = MAD_NORMAL_SCALE * np.median(np.abs(values - mu), axis=0)
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
    return shrink_covariance(cov, method=method, n_samples=values.shape[0])


def _kmeans_labels(
    values: np.ndarray,
    *,
    n_clusters: int,
    random_state: int | None,
    max_iter: int = 50,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if values.shape[0] < n_clusters:
        raise ValueError("n_clusters must be <= number of pixels")
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


def _gmm_labels(
    values: np.ndarray,
    *,
    n_clusters: int,
    random_state: int | None,
    bayesian: bool,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if values.shape[0] < n_clusters:
        raise ValueError("n_clusters must be <= number of pixels")
    labels = _kmeans_labels(
        values, n_clusters=n_clusters, random_state=random_state, max_iter=10
    )
    means = np.empty((n_clusters, values.shape[1]), dtype=float)
    variances = np.empty_like(means)
    global_variance = values.var(axis=0) + GMM_VARIANCE_RIDGE
    for k in range(n_clusters):
        group = values[labels == k]
        if group.shape[0] == 0:
            means[k] = values[rng.integers(values.shape[0])]
            variances[k] = global_variance
        else:
            means[k] = group.mean(axis=0)
            variances[k] = group.var(axis=0) + GMM_VARIANCE_RIDGE
    weights = np.bincount(labels, minlength=n_clusters).astype(float) / values.shape[0]
    previous_ll = -np.inf
    for _ in range(max_iter):
        log_prob = _diag_gmm_log_prob(values, means, variances, weights)
        log_norm = np.logaddexp.reduce(log_prob, axis=1)
        responsibilities = np.exp(log_prob - log_norm[:, None])
        counts = responsibilities.sum(axis=0)
        empty = counts <= np.finfo(float).eps
        if np.any(empty):
            repl = rng.choice(values.shape[0], size=int(empty.sum()), replace=False)
            means[empty] = values[repl]
            variances[empty] = values.var(axis=0) + GMM_VARIANCE_RIDGE
            counts[empty] = 1.0
        weights = counts / counts.sum()
        means = responsibilities.T @ values / counts[:, None]
        centered = values[:, None, :] - means[None, :, :]
        variances = (responsibilities[:, :, None] * centered * centered).sum(
            axis=0
        ) / counts[:, None]
        variances = np.maximum(variances, GMM_VARIANCE_RIDGE)
        ll = float(np.sum(log_norm))
        if abs(ll - previous_ll) / max(abs(ll), 1.0) < tol:
            break
        previous_ll = ll
    minimum_component_weight = 1.0 / values.shape[0]
    active = (
        weights > minimum_component_weight
        if bayesian
        else np.ones_like(weights, dtype=bool)
    )
    if not np.any(active):
        active[np.argmax(weights)] = True
    log_prob = _diag_gmm_log_prob(
        values, means[active], variances[active], weights[active]
    )
    return np.argmax(log_prob, axis=1)


def _diag_gmm_log_prob(
    values: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    centered = values[:, None, :] - means[None, :, :]
    mahal = np.sum(centered * centered / variances[None, :, :], axis=2)
    log_det = np.sum(np.log(variances), axis=1)
    log_weights = np.log(np.maximum(weights, np.finfo(float).tiny))
    return log_weights[None, :] - 0.5 * (
        values.shape[1] * np.log(2.0 * np.pi) + log_det[None, :] + mahal
    )
