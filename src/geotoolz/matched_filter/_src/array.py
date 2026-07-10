"""Pure NumPy matched-filter primitives for hyperspectral cubes.

Tier-A layer of the matched-filter module: everything here is plain
numpy (plus scipy for filtering and distributions) operating on spectral
cubes shaped ``(c, h, w)`` — band axis first by default, movable via the
``axis`` argument. The carrier-aware Operator wrappers live in
:mod:`geotoolz.matched_filter._src.operators`.

The classic matched filter scores each pixel spectrum :math:`x` against
a target signature :math:`t` under background statistics
:math:`(\\mu, \\Sigma)`:

.. math::

    \\alpha(x) \\;=\\; \\frac{(x - \\mu)^\\top \\Sigma^{-1} t}
                           {t^\\top \\Sigma^{-1} t}

which is the maximum-likelihood estimate of the target amplitude under
an additive-target Gaussian background model. The primitives split into
three groups:

- **background estimation** — robust means (`estimate_mean`); empirical,
  shrunk, and low-rank covariances (`estimate_cov_empirical`,
  `estimate_cov_shrunk`, `estimate_cov_lowrank`); clustered
  (`gmm_cluster_background`) and locally adaptive
  (`adaptive_window_background`) backgrounds; and the streaming
  `WelfordAccumulator`;
- **filtering** — `apply_pixel`, `apply_image`, `apply_cluster_mf`;
- **detection theory** — `matched_filter_snr`, `detection_threshold`,
  `validate_mf_inputs`.

Shape conventions (mirrored in the jaxtyping annotations): cubes are
``(c, h, w)``, vectorised sample matrices are ``(n, c)``, covariance
matrices are ``(c, c)``, and mean / target spectra are ``(c,)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float, Int
from scipy import ndimage, stats


MeanMethod = Literal["mean", "median", "trimmed", "huber"]
CovShrinkageMethod = Literal["ledoit_wolf", "oas"]
CovMethod = Literal["empirical", "ledoit_wolf", "oas", "lowrank"]
MAD_NORMAL_SCALE = 1.4826
GMM_VARIANCE_RIDGE = 1e-6


@dataclass(frozen=True)
class NumpyLinearOperator:
    """Dense NumPy linear operator with a small ``solve`` interface.

    Wraps a square (covariance) matrix behind the minimal interface the
    matched-filter kernels need — ``shape`` and ``solve`` — so callers
    don't depend on how the operator is represented. The matrix is
    coerced to a float ndarray at construction.

    Attributes:
        matrix: Square ``(c, c)`` float matrix.

    Raises:
        ValueError: If ``matrix`` is not a square 2-D array.
    """

    matrix: Float[np.ndarray, "c c"]

    def __post_init__(self) -> None:
        mat = np.asarray(self.matrix, dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("matrix must be a square 2-D array")
        object.__setattr__(self, "matrix", mat)

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape ``(c, c)``."""
        return self.matrix.shape

    def solve(self, rhs: Float[np.ndarray, " c"]) -> Float[np.ndarray, " c"]:
        """Solve ``matrix @ x = rhs`` for ``x``.

        Args:
            rhs: Right-hand-side vector with one entry per band.

        Returns:
            The solution vector ``x``.

        Raises:
            numpy.linalg.LinAlgError: If the matrix is singular.
        """
        return np.linalg.solve(self.matrix, np.asarray(rhs, dtype=float))


@dataclass(frozen=True)
class ClusterBackground:
    """Cluster labels with per-cluster matched-filter background statistics.

    Produced by `gmm_cluster_background` and consumed by
    `apply_cluster_mf`, which scores each pixel against the statistics
    of its own cluster.

    Attributes:
        labels: Integer cluster-label map over the cube's spatial shape;
            values index into ``means`` and ``cov_ops``.
        means: Per-cluster mean spectra, shaped ``(k, c)``.
        cov_ops: Per-cluster ``(c, c)`` covariance operators, one entry
            per cluster.
    """

    labels: Int[np.ndarray, "h w"]
    means: Float[np.ndarray, "k c"]
    cov_ops: tuple[NumpyLinearOperator, ...]


@dataclass(frozen=True)
class AdaptiveBackground:
    """Local per-pixel background statistics from square-window filtering.

    Produced by `adaptive_window_background`. The covariance model is
    diagonal: only per-band local variances are stored.

    Attributes:
        mean: Per-pixel local mean cube, shaped like the input
            ``(c, h, w)``.
        variance: Per-pixel local (diagonal, unbiased) variance cube,
            shaped like the input ``(c, h, w)``.
    """

    mean: Float[np.ndarray, "c h w"]
    variance: Float[np.ndarray, "c h w"]


@dataclass(frozen=True)
class StreamingBackgroundResult:
    """Mean and covariance operator estimated from streamed cubes.

    Attributes:
        mean: Background mean spectrum ``(c,)`` over all streamed pixels.
        cov_op: Background ``(c, c)`` covariance operator.
    """

    mean: Float[np.ndarray, " c"]
    cov_op: NumpyLinearOperator


@dataclass
class WelfordAccumulator:
    """Streaming mean/covariance accumulator using Chan-Welford updates.

    Accumulates first and second moments over batches of spectra without
    holding them all in memory, using the numerically stable pairwise
    (Chan et al.) merge of Welford statistics. Feed sample batches with
    `update` (or combine partial accumulators with `merge`) and read the
    result off ``mean`` / `covariance`.

    Attributes:
        count: Number of samples absorbed so far.
        mean: Running mean spectrum ``(c,)``.
        m2: Running centred sum-of-products matrix ``(c, c)``;
            ``m2 / (count - ddof)`` is the sample covariance.
    """

    count: int
    mean: Float[np.ndarray, " c"]
    m2: Float[np.ndarray, "c c"]

    @classmethod
    def empty(cls, n_features: int) -> WelfordAccumulator:
        """Create an empty accumulator for ``n_features`` spectral bands.

        Args:
            n_features: Number of bands ``c`` of the incoming samples.

        Returns:
            A zero-count accumulator ready for `update` / `merge`.
        """
        return cls(
            count=0,
            mean=np.zeros(n_features, dtype=float),
            m2=np.zeros((n_features, n_features), dtype=float),
        )

    def update(self, values: Float[np.ndarray, "n c"]) -> None:
        """Absorb a batch of samples into the running statistics.

        Args:
            values: Sample rows shaped ``(n, c)``; a single 1-D spectrum
                is treated as one row. Empty batches are a no-op.
        """
        x = _as_2d_samples(values)
        if x.size == 0:
            return
        other = WelfordAccumulator.from_values(x)
        self.merge(other)

    def merge(self, other: WelfordAccumulator) -> None:
        """Merge another accumulator into this one in place.

        Uses the pairwise Chan update, so merging accumulators built
        from disjoint batches is exact (equivalent to a single pass over
        the concatenated samples).

        Args:
            other: Accumulator over the same number of bands.
        """
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
    def from_values(cls, values: Float[np.ndarray, "n c"]) -> WelfordAccumulator:
        """Create an accumulator from a batch of sample rows.

        Args:
            values: Sample rows shaped ``(n, c)``; a single 1-D spectrum
                is treated as one row.

        Returns:
            An accumulator holding the batch's exact mean and centred
            sum-of-products.
        """
        x = _as_2d_samples(values)
        if x.shape[0] == 0:
            return cls.empty(x.shape[1])
        mean = np.mean(x, axis=0)
        centered = x - mean
        return cls(count=x.shape[0], mean=mean, m2=centered.T @ centered)

    def covariance(
        self, *, ddof: int = 1, ridge: float = 0.0
    ) -> Float[np.ndarray, "c c"]:
        """Return the sample covariance matrix of the absorbed samples.

        Args:
            ddof: Delta degrees of freedom; the default ``1`` gives the
                unbiased estimator.
            ridge: Optional Tikhonov ridge added to the diagonal to keep
                the matrix invertible.

        Returns:
            The ``(c, c)`` covariance matrix.

        Raises:
            ValueError: If fewer than ``ddof + 1`` samples were absorbed.
        """
        if self.count <= ddof:
            raise ValueError("at least two samples are required for covariance")
        cov = self.m2 / (self.count - ddof)
        if ridge:
            cov = cov + float(ridge) * np.eye(cov.shape[0], dtype=float)
        return cov


def cube_to_samples(
    cube: Float[np.ndarray, "c h w"], *, axis: int = 0
) -> tuple[Float[np.ndarray, "n c"], tuple[int, ...]]:
    """Vectorise a spectral cube into a ``(pixels, bands)`` sample matrix.

    Moves the spectral axis last and flattens the remaining (spatial)
    axes — the layout every estimator in this module works in.

    Args:
        cube: Spectral cube, canonically ``(c, h, w)``; any number of
            non-band dimensions (at least one) is supported. Array-likes
            are coerced to float.
        axis: Position of the spectral (band) axis. Default ``0``.

    Returns:
        A pair ``(samples, spatial_shape)`` where ``samples`` is the
        ``(n, c)`` float sample matrix and ``spatial_shape`` is the shape
        of the non-band axes, for reshaping per-pixel results back into
        maps.

    Raises:
        ValueError: If ``cube`` has fewer than two dimensions.
    """
    arr = np.asarray(cube, dtype=float)
    if arr.ndim < 2:
        raise ValueError("matched-filter input must have at least two dimensions")
    moved = np.moveaxis(arr, axis, -1)
    spatial_shape = moved.shape[:-1]
    return moved.reshape(-1, moved.shape[-1]), spatial_shape


def estimate_mean(
    cube: Float[np.ndarray, "c h w"],
    *,
    method: MeanMethod = "mean",
    trim_proportion: float = 0.1,
    huber_c: float = 1.345,
    axis: int = 0,
) -> Float[np.ndarray, " c"]:
    """Estimate a per-band background mean spectrum from a cube.

    Robustness matters for matched filtering: sparse bright targets
    (plumes, panels) bias the plain mean toward the target itself, so
    the ``median``, ``trimmed``, or ``huber`` estimators usually give a
    cleaner background.

    Args:
        cube: Spectral cube, canonically ``(c, h, w)``; any number of
            non-band dimensions is supported.
        method: Mean estimator — ``"mean"`` (arithmetic), ``"median"``,
            ``"trimmed"`` (symmetric trimmed mean), or ``"huber"``
            (iteratively reweighted Huber M-estimator).
        trim_proportion: Fraction cut from *each* tail for
            ``method="trimmed"``; must satisfy ``0 <= p < 0.5``.
        huber_c: Huber tuning constant in robust z-score units for
            ``method="huber"``; smaller values down-weight outliers more
            aggressively. Must be positive.
        axis: Position of the spectral axis. Default ``0``.

    Returns:
        Mean spectrum with one entry per band.

    Raises:
        ValueError: If ``method`` is unknown, ``trim_proportion`` is out
            of range, or ``huber_c`` is not positive.
    """
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
    cube: Float[np.ndarray, "c h w"],
    *,
    mean: Float[np.ndarray, " c"] | None = None,
    ridge: float = 0.0,
    axis: int = 0,
) -> NumpyLinearOperator:
    """Estimate an empirical covariance operator from a cube.

    Computes the standard ``ddof=1`` sample covariance of the vectorised
    pixel spectra, optionally centred on a caller-supplied (e.g. robust)
    mean and regularised with a diagonal ridge.

    Args:
        cube: Spectral cube, canonically ``(c, h, w)``.
        mean: Optional precomputed mean spectrum ``(c,)`` to centre on;
            when ``None`` the arithmetic sample mean is used.
        ridge: Optional Tikhonov ridge added to the diagonal to keep the
            matrix invertible for near-degenerate backgrounds.
        axis: Position of the spectral axis. Default ``0``.

    Returns:
        `NumpyLinearOperator` wrapping the ``(c, c)`` sample covariance.

    Raises:
        ValueError: If ``mean`` length does not match the band count.
    """
    x, _ = cube_to_samples(cube, axis=axis)
    mu = np.mean(x, axis=0) if mean is None else _as_vector(mean, x.shape[1], "mean")
    centered = x - mu
    denom = max(x.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    if ridge:
        cov = cov + float(ridge) * np.eye(cov.shape[0], dtype=float)
    return NumpyLinearOperator(cov)


def estimate_cov_shrunk(
    cube: Float[np.ndarray, "c h w"],
    *,
    mean: Float[np.ndarray, " c"] | None = None,
    method: CovShrinkageMethod = "ledoit_wolf",
    axis: int = 0,
) -> NumpyLinearOperator:
    """Estimate a diagonal-target shrinkage covariance operator.

    Forms the empirical covariance of the vectorised spectra and blends
    it toward the scaled-identity target ``tr(S)/c * I`` with a
    data-driven intensity (see `shrink_covariance`) — the standard fix
    when the pixel count is small relative to the band count.

    Args:
        cube: Spectral cube, canonically ``(c, h, w)``.
        mean: Optional precomputed mean spectrum ``(c,)`` to centre on;
            when ``None`` the arithmetic sample mean is used.
        method: Shrinkage-intensity estimator, ``"ledoit_wolf"`` or
            ``"oas"``.
        axis: Position of the spectral axis. Default ``0``.

    Returns:
        `NumpyLinearOperator` wrapping the shrunk ``(c, c)`` covariance.

    Raises:
        ValueError: If ``mean`` length does not match the band count or
            ``method`` is unknown.
    """
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
    empirical: Float[np.ndarray, "c c"],
    *,
    method: CovShrinkageMethod,
    n_samples: int,
) -> Float[np.ndarray, "c c"]:
    """Shrink an empirical covariance toward a scaled-identity target.

    Returns the convex combination ``(1 - s) * S + s * (tr(S)/c) * I``
    where the intensity ``s`` in ``[0, 1]`` is estimated from the data:

    - ``"ledoit_wolf"``: analytical Ledoit-Wolf approximation (see the
      derivation notes inline);
    - ``"oas"``: Oracle Approximating Shrinkage (Chen et al. 2010).

    Args:
        empirical: Empirical covariance matrix ``(c, c)``.
        method: Shrinkage-intensity estimator.
        n_samples: Number of samples used to form ``empirical``; more
            samples mean less shrinkage.

    Returns:
        The shrunk ``(c, c)`` covariance matrix.

    Raises:
        ValueError: If ``method`` is unknown.
    """
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
    cube: Float[np.ndarray, "c h w"],
    *,
    mean: Float[np.ndarray, " c"] | None = None,
    rank: int = 10,
    tikhonov: float = 1e-3,
    random_state: int | None = 0,
    n_oversamples: int = 10,
    axis: int = 0,
) -> NumpyLinearOperator:
    """Estimate a low-rank-plus-Tikhonov dense covariance operator.

    Approximates the empirical covariance by its top-``rank`` eigenpairs
    — found with a randomized range finder (Halko et al. 2011) followed
    by an SVD of the projected matrix — and adds a Tikhonov diagonal, so
    the result is the dense, well-conditioned matrix
    ``U_k S_k U_k^T + tikhonov * I``. Useful when the band count is
    large and the background is dominated by a few spectral modes.

    Args:
        cube: Spectral cube, canonically ``(c, h, w)``.
        mean: Optional precomputed mean spectrum ``(c,)`` to centre on;
            when ``None`` the arithmetic sample mean is used.
        rank: Number of leading eigenpairs kept. Must be positive;
            values above the band count are truncated to it.
        tikhonov: Diagonal regulariser added after the low-rank
            reconstruction; keeps the operator invertible even though
            the low-rank part is singular for ``rank < c``.
        random_state: Seed for the randomized range finder. The default
            ``0`` makes the estimate deterministic; ``None`` draws fresh
            OS entropy.
        n_oversamples: Extra random probe vectors beyond ``rank`` used
            by the range finder; improves subspace capture at negligible
            cost. Must be non-negative.
        axis: Position of the spectral axis. Default ``0``.

    Returns:
        `NumpyLinearOperator` wrapping the regularised ``(c, c)``
        low-rank covariance.

    Raises:
        ValueError: If ``rank`` is not positive, ``n_oversamples`` is
            negative, or ``mean`` length does not match the band count.
    """
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
    pixel: Float[np.ndarray, " c"],
    *,
    mean: Float[np.ndarray, " c"],
    cov_op: NumpyLinearOperator | Float[np.ndarray, "c c"],
    target: Float[np.ndarray, " c"],
) -> float:
    """Score one pixel spectrum with the matched filter.

    Computes the maximum-likelihood target-amplitude estimate

    .. math::

        \\alpha \\;=\\; \\frac{(x - \\mu)^\\top \\Sigma^{-1} t}
                             {t^\\top \\Sigma^{-1} t}

    for a single spectrum ``x``. See `apply_image` for whole cubes.

    Args:
        pixel: Pixel spectrum ``(c,)``.
        mean: Background mean spectrum ``(c,)``.
        cov_op: Background covariance as a `NumpyLinearOperator` or a
            raw ``(c, c)`` matrix.
        target: Target signature ``(c,)``.

    Returns:
        The scalar matched-filter score (estimated target amplitude).

    Raises:
        ValueError: If vector lengths disagree with the band count, or
            the target/covariance pair produces a non-positive filter
            denominator.
    """
    mean_vec = _as_vector(mean, np.asarray(pixel).shape[0], "mean")
    target_vec = _as_vector(target, mean_vec.shape[0], "target")
    solved_target = solve(cov_op, target_vec)
    denom = float(target_vec @ solved_target)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("target/covariance produce a non-positive MF denominator")
    return float((np.asarray(pixel, dtype=float) - mean_vec) @ solved_target / denom)


def apply_image(
    cube: Float[np.ndarray, "c h w"],
    *,
    mean: Float[np.ndarray, " c"],
    cov_op: NumpyLinearOperator | Float[np.ndarray, "c c"],
    target: Float[np.ndarray, " c"],
    axis: int = 0,
) -> Float[np.ndarray, "h w"]:
    """Apply a matched filter over a hyperspectral image cube.

    The covariance system is solved once for the target and reused for
    every pixel, so the per-pixel work is a single dot product. See
    `apply_pixel` for the scalar kernel and its formula.

    Args:
        cube: Spectral cube, canonically ``(c, h, w)``; any number of
            non-band dimensions is supported.
        mean: Background mean spectrum ``(c,)``.
        cov_op: Background covariance as a `NumpyLinearOperator` or a
            raw ``(c, c)`` matrix.
        target: Target signature ``(c,)``.
        axis: Position of the spectral axis. Default ``0``.

    Returns:
        Matched-filter score map with the cube's spatial shape (e.g.
        ``(h, w)`` for a ``(c, h, w)`` cube).

    Raises:
        ValueError: If vector lengths disagree with the band count, or
            the target/covariance pair produces a non-positive filter
            denominator.
    """
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
    *,
    amplitude: float,
    cov_op: NumpyLinearOperator | Float[np.ndarray, "c c"],
    target: Float[np.ndarray, " c"],
) -> float:
    """Return the theoretical matched-filter detection SNR.

    For a target of amplitude ``a`` in Gaussian background noise with
    covariance :math:`\\Sigma`, the matched-filter output SNR is
    :math:`a \\sqrt{t^\\top \\Sigma^{-1} t}`.

    Args:
        amplitude: Target amplitude ``a``.
        cov_op: Background covariance as a `NumpyLinearOperator` or a
            raw ``(c, c)`` matrix.
        target: Target signature ``(c,)``.

    Returns:
        The scalar SNR.

    Raises:
        ValueError: If the target/covariance pair produces a
            non-positive filter gain.
    """
    t = np.asarray(target, dtype=float).reshape(-1)
    gain = float(t @ solve(cov_op, t))
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("target/covariance produce a non-positive MF gain")
    return float(amplitude) * float(np.sqrt(gain))


def detection_threshold(
    *,
    false_alarm_rate: float,
    cov_op: NumpyLinearOperator | Float[np.ndarray, "c c"],
    target: Float[np.ndarray, " c"],
) -> float:
    """Return the score threshold for a Gaussian false-alarm rate.

    Under the no-target null hypothesis the matched-filter score is
    zero-mean Gaussian with standard deviation
    :math:`1 / \\sqrt{t^\\top \\Sigma^{-1} t}`, so thresholding scores at
    :math:`\\Phi^{-1}(1 - \\text{FAR}) / \\sqrt{t^\\top \\Sigma^{-1} t}`
    yields the requested false-alarm probability.

    Args:
        false_alarm_rate: Desired false-alarm probability, strictly
            between 0 and 1.
        cov_op: Background covariance as a `NumpyLinearOperator` or a
            raw ``(c, c)`` matrix.
        target: Target signature ``(c,)``.

    Returns:
        The scalar score threshold.

    Raises:
        ValueError: If ``false_alarm_rate`` is outside ``(0, 1)`` or the
            target/covariance pair produces a non-positive filter gain.
    """
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false_alarm_rate must be between 0 and 1")
    gain = matched_filter_snr(amplitude=1.0, cov_op=cov_op, target=target)
    return float(stats.norm.ppf(1.0 - false_alarm_rate) / gain)


def validate_mf_inputs(
    *,
    cov_op: NumpyLinearOperator | Float[np.ndarray, "c c"],
    target: Float[np.ndarray, " c"],
) -> None:
    """Raise ``ValueError`` for degenerate target/covariance pairs.

    Checks, in order: the target has at least one non-zero entry, the
    covariance system is solvable, and the resulting filter gain
    :math:`t^\\top \\Sigma^{-1} t` is finite and positive.

    Args:
        cov_op: Background covariance as a `NumpyLinearOperator` or a
            raw ``(c, c)`` matrix.
        target: Target signature ``(c,)``.

    Raises:
        ValueError: If the target is all-zero, the covariance is
            singular, or the filter gain is non-positive.
    """
    t = np.asarray(target, dtype=float).reshape(-1)
    if t.size == 0 or not np.any(t):
        raise ValueError("target must contain at least one non-zero value")
    try:
        gain = float(t @ solve(cov_op, t))
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance operator must be non-singular") from exc
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("target/covariance produce a non-positive MF gain")


def solve(
    cov_op: NumpyLinearOperator | Float[np.ndarray, "c c"],
    rhs: Float[np.ndarray, " c"],
) -> Float[np.ndarray, " c"]:
    """Solve a covariance system for either supported representation.

    Args:
        cov_op: Background covariance as a `NumpyLinearOperator` or a
            raw ``(c, c)`` matrix.
        rhs: Right-hand-side vector ``(c,)``.

    Returns:
        The solution ``x`` of ``cov @ x = rhs``.

    Raises:
        numpy.linalg.LinAlgError: If the covariance is singular.
    """
    if isinstance(cov_op, NumpyLinearOperator):
        return cov_op.solve(rhs)
    return np.linalg.solve(
        np.asarray(cov_op, dtype=float), np.asarray(rhs, dtype=float)
    )


def gmm_cluster_background(
    cube: Float[np.ndarray, "c h w"],
    *,
    n_clusters: int,
    cov_estimator: Literal["empirical", "ledoit_wolf", "oas"] = "ledoit_wolf",
    random_state: int | None = 0,
    bayesian: bool = False,
    axis: int = 0,
) -> ClusterBackground:
    """Estimate a clustered background with a deterministic diagonal GMM.

    Pixels are soft-clustered by a small pure-NumPy EM loop (k-means
    initialisation, diagonal component covariances) and hard-assigned to
    their most likely component; each cluster then receives a full mean
    and covariance for cluster-wise matched filtering via
    `apply_cluster_mf`.

    Args:
        cube: Spectral cube, canonically ``(c, h, w)``.
        n_clusters: Number of mixture components; must be positive and
            no larger than the pixel count.
        cov_estimator: Per-cluster covariance estimator —
            ``"empirical"`` (with a tiny stabilising ridge) or shrunk
            via ``"ledoit_wolf"`` / ``"oas"``.
        random_state: Seed for the k-means initialisation and
            empty-cluster restarts; a fixed seed makes the clustering
            reproducible.
        bayesian: If ``True``, drop components whose mixture weight
            falls below ``1 / n_pixels`` (a cheap sparsifying prune), so
            fewer than ``n_clusters`` clusters may be returned.
        axis: Position of the spectral axis. Default ``0``.

    Returns:
        `ClusterBackground` with the per-pixel label map and per-cluster
        means and covariance operators.

    Raises:
        ValueError: If ``n_clusters`` is not positive or exceeds the
            pixel count, a cluster ends up empty, or ``cov_estimator``
            is unknown.
    """
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
    cube: Float[np.ndarray, "c h w"],
    *,
    cluster: ClusterBackground,
    target: Float[np.ndarray, " c"],
    axis: int = 0,
) -> Float[np.ndarray, "h w"]:
    """Apply a matched filter with per-cluster background statistics.

    Each pixel is scored against the mean and covariance of its own
    cluster (from `gmm_cluster_background`), adapting the filter to
    scene heterogeneity that a single global background would smear.

    Args:
        cube: Spectral cube, canonically ``(c, h, w)``; its spatial
            shape must match ``cluster.labels``.
        cluster: Clustered background statistics for the cube.
        target: Target signature ``(c,)``.
        axis: Position of the spectral axis. Default ``0``.

    Returns:
        Matched-filter score map with the cube's spatial shape.

    Raises:
        ValueError: If the label map does not match the cube's spatial
            size, the target length disagrees with the band count, or a
            cluster produces a non-positive filter denominator.
    """
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
    cube: Float[np.ndarray, "c h w"],
    *,
    window_size: int = 7,
    pad_mode: str = "reflect",
    axis: int = 0,
) -> AdaptiveBackground:
    """Estimate local mean and diagonal variance over square windows.

    Per-band uniform filtering yields, for every pixel, the mean and
    unbiased variance of its ``window_size x window_size``
    neighbourhood — a cheap, locally adaptive background model for
    detection over non-stationary scenes.

    Args:
        cube: Spectral cube; must be exactly 3-D, canonically
            ``(c, h, w)``.
        window_size: Side length of the square window; must be a
            positive odd integer so windows are centred.
        pad_mode: Boundary mode forwarded to
            :func:`scipy.ndimage.uniform_filter` (e.g. ``"reflect"``,
            ``"nearest"``, ``"wrap"``).
        axis: Position of the spectral axis. Default ``0``.

    Returns:
        `AdaptiveBackground` with per-pixel ``mean`` and diagonal
        ``variance`` cubes shaped like the input.

    Raises:
        ValueError: If ``window_size`` is not a positive odd integer or
            the cube is not 3-D.
    """
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


def _as_vector(values: np.ndarray, size: int, name: str) -> Float[np.ndarray, " c"]:
    vec = np.asarray(values, dtype=float).reshape(-1)
    if vec.shape[0] != size:
        raise ValueError(
            f"{name} length {vec.shape[0]} does not match band count {size}"
        )
    return vec


def _as_2d_samples(values: np.ndarray) -> Float[np.ndarray, "n c"]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("values must be a 1-D vector or 2-D sample matrix")
    return arr


def _huber_mean(
    values: Float[np.ndarray, "n c"],
    *,
    c: float,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Float[np.ndarray, " c"]:
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
    values: Float[np.ndarray, "n c"],
    mean: Float[np.ndarray, " c"],
    *,
    ridge: float = 0.0,
) -> Float[np.ndarray, "c c"]:
    centered = values - mean
    denom = max(values.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    if ridge:
        cov = cov + ridge * np.eye(cov.shape[0], dtype=float)
    return cov


def _shrunk_cov_from_samples(
    values: Float[np.ndarray, "n c"],
    mean: Float[np.ndarray, " c"],
    *,
    method: Literal["ledoit_wolf", "oas"],
) -> Float[np.ndarray, "c c"]:
    cov = _cov_from_samples(values, mean, ridge=1e-8)
    return shrink_covariance(cov, method=method, n_samples=values.shape[0])


def _kmeans_labels(
    values: Float[np.ndarray, "n c"],
    *,
    n_clusters: int,
    random_state: int | None,
    max_iter: int = 50,
) -> Int[np.ndarray, " n"]:
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
    values: Float[np.ndarray, "n c"],
    *,
    n_clusters: int,
    random_state: int | None,
    bayesian: bool,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> Int[np.ndarray, " n"]:
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
    values: Float[np.ndarray, "n c"],
    means: Float[np.ndarray, "k c"],
    variances: Float[np.ndarray, "k c"],
    weights: Float[np.ndarray, " k"],
) -> Float[np.ndarray, "n k"]:
    centered = values[:, None, :] - means[None, :, :]
    mahal = np.sum(centered * centered / variances[None, :, :], axis=2)
    log_det = np.sum(np.log(variances), axis=1)
    log_weights = np.log(np.maximum(weights, np.finfo(float).tiny))
    return log_weights[None, :] - 0.5 * (
        values.shape[1] * np.log(2.0 * np.pi) + log_det[None, :] + mahal
    )
