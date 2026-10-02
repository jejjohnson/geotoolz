"""`SpatialWindow` — boundary treatment for the patch.

A `SpatialWindow` returns a weight array shaped like the geometry's patch. The
weights multiply into the patch data on the way in (for `SpatialOverlapAdd`-
style aggregations) and form the denominator on the way out so the
overlap-add reconstruction recovers the unweighted field.

Five windows: `SpatialBoxcar` (no taper), `SpatialHann`, `SpatialTukey`,
`SpatialGaussian`, and a `SpatialCustom` escape hatch for caller-supplied
weight functions.

Taper convention: `SpatialHann` and `SpatialTukey` are **periodic**
(DFT-even, ``scipy.signal.windows.*(n, sym=False)``), applied separably
along each axis. A periodic length-``N`` Hann sums to exactly ``1`` when
shifted by ``N / 2`` (constant overlap-add), so with even ``N`` and
``step = N / 2`` every interior seam is covered at full weight. The first
sample of each axis is ``0`` and the last is not, so the only cells left
with zero accumulated weight are the leading (top / left) row and column
of the domain — the "border ring" — which no chip covers with non-zero
weight because regular samplers start at anchor ``0``. Axes shorter than
3 samples cannot carry a taper and fall back to boxcar (all ones).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from geopatcher._src._serialize import config_from_fields
from geopatcher._src.spatial.geometry import SpatialGeometry, SpatialRectangular


class SpatialWindow:
    """Base for window functions.

    Subclasses implement `weights(geometry) -> Array`. The base class
    handles the trivial `SpatialBoxcar` default.
    """

    forbid_in_yaml: ClassVar[bool] = False

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass(eq=False)
class SpatialBoxcar(SpatialWindow):
    """Constant 1.0 — no edge taper."""

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        shape = geom_shape(geometry)
        return np.ones(shape, dtype=np.float64)


@dataclass(eq=False)
class SpatialHann(SpatialWindow):
    """Periodic Hann (raised-cosine) window — the standard overlap-add taper.

    Each axis of length ``n`` is ``scipy.signal.windows.hann(n, sym=False)``,
    i.e. ``0.5 - 0.5 * cos(2 * pi * k / n)`` for ``k = 0 .. n - 1``. The
    periodic form is exactly COLA at hop ``n / 2``: with an even patch size
    and ``step = size / 2`` the windows of neighbouring chips sum to ``1``,
    so even un-normalised overlap-add reproduces a constant field across
    interior seams.

    Endpoints: ``w[0] = 0`` and ``w[-1] > 0``. Merged with
    `SpatialOverlapAdd`, the domain's first row and column therefore get
    zero accumulated weight (filled with ``0.0``); see the "Window
    convention" section of the patching docs. With ``step == size`` every
    chip's first row/column is a zero-weight seam — overlap the chips or
    use `SpatialBoxcar` for exact tiling.

    Axes shorter than 3 samples fall back to boxcar (all ones): a 1- or
    2-sample taper is ``[1]`` / ``[0, 1]``, which only zeroes data.
    """

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        from scipy.signal.windows import hann

        shape = geom_shape(geometry)
        return _separable(shape, lambda n: _tapered(n, lambda m: hann(m, sym=False)))


@dataclass(eq=False)
class SpatialTukey(SpatialWindow):
    """Periodic Tukey (tapered-cosine) window.

    Each axis of length ``n`` is
    ``scipy.signal.windows.tukey(n, alpha, sym=False)`` — the same periodic
    (DFT-even) convention as `SpatialHann`, so ``alpha = 1.0`` is exactly
    `SpatialHann` and ``alpha = 0.0`` is exactly `SpatialBoxcar`. A periodic
    Tukey is COLA at hop ``n * (1 - alpha / 2)`` (``0.75 * n`` for the
    default ``alpha = 0.5``); other hops still reconstruct correctly under
    `SpatialOverlapAdd`'s normalisation, just not as a partition of unity.

    Like `SpatialHann`, ``w[0] = 0`` for ``alpha > 0`` (the leading
    border-ring caveat applies), and axes shorter than 3 samples fall back
    to boxcar.

    Args:
        alpha: Fraction of the window occupied by the cosine taper
            (``0.0`` is `SpatialBoxcar`, ``1.0`` is `SpatialHann`). Defaults
            to ``0.5``.
    """

    alpha: float = 0.5

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        from scipy.signal.windows import tukey

        shape = geom_shape(geometry)
        return _separable(
            shape,
            lambda n: _tapered(n, lambda m: tukey(m, alpha=self.alpha, sym=False)),
        )

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class SpatialGaussian(SpatialWindow):
    """Separable Gaussian envelope centred on the patch.

    ``sigma`` is a **fraction of the patch half-width**, not a pixel
    count: along an axis of length ``n`` the standard deviation is
    ``sigma * n / 2`` pixels, centred at ``(n - 1) / 2`` (symmetric). The
    same value therefore gives geometrically similar tapers on different
    patch shapes; the default ``0.5`` puts the patch edge at about two
    standard deviations (edge weight ``≈ exp(-2) ≈ 0.14`` for large patches).
    The weights never reach zero, so a Gaussian has no zero-weight border
    ring, but it is not a partition of unity — rely on
    `SpatialOverlapAdd`'s normalisation.

    Args:
        sigma: Standard deviation as a fraction of the half-width.
            Defaults to ``0.5``.
    """

    sigma: float = 0.5

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        shape = geom_shape(geometry)
        return _separable(shape, lambda n: _gaussian_1d(n, self.sigma))

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class SpatialCustom(SpatialWindow):
    """Caller-supplied weight function — the escape hatch.

    Args:
        fn: Callable ``(geometry) -> np.ndarray`` returning the weight
            array. Carries closures, so ``forbid_in_yaml = True``.
    """

    fn: Callable[[SpatialGeometry], np.ndarray]

    forbid_in_yaml: ClassVar[bool] = True

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        return self.fn(geometry)


def geom_shape(geometry: SpatialGeometry) -> tuple[int, ...]:
    """Shape of the weight array a `SpatialWindow` returns for ``geometry``.

    Public so third-party windows (e.g. ``geotoolz.patch_ops.SpatialTriangular``)
    size their weights exactly like the built-in ones.

    Args:
        geometry: A fixed-size geometry — `SpatialRectangular`, or any
            object exposing a ``size`` sequence.

    Returns:
        ``size`` with every entry coerced to ``int``.

    Raises:
        TypeError: if the geometry has no fixed size (ragged geometries).
    """
    if isinstance(geometry, SpatialRectangular):
        return tuple(int(s) for s in geometry.size)
    size = getattr(geometry, "size", None)
    if size is not None:
        return tuple(int(s) for s in size)
    raise TypeError(
        f"SpatialWindow weights aren't defined for {type(geometry).__name__} - "
        "only fixed-shape geometries (e.g. SpatialRectangular). Use "
        "SpatialBoxcar for ragged geometries like SpatialRadiusGraph / "
        "SpatialKNNGraph / SpatialPolygonIntersection."
    )


def _separable(
    shape: tuple[int, ...], one_d: Callable[[int], np.ndarray]
) -> np.ndarray:
    """Build an N-D window as the outer product of 1-D windows.

    Standard practice — keeps construction O(sum(shape)) and avoids the
    O(prod(shape)) cost of an explicit N-D formula.
    """
    if not shape:
        return np.array(1.0)
    axes = [one_d(int(n)).astype(np.float64) for n in shape]
    out = axes[0]
    for ax in axes[1:]:
        out = np.multiply.outer(out, ax)
    return out


def _tapered(n: int, taper: Callable[[int], np.ndarray]) -> np.ndarray:
    """``taper(n)``, or boxcar (all ones) when ``n < 3``.

    A periodic taper of length 1 or 2 is ``[1]`` / ``[0, 1]`` — no taper at
    all, or one that just throws half the samples away (``step == size == 2``
    would leave every other row with zero weight). Boxcar keeps every sample
    and stays per-axis, so a ``(1, 256)`` strip still tapers along its long
    axis.
    """
    if n < 3:
        return np.ones(n, dtype=np.float64)
    return np.asarray(taper(n), dtype=np.float64)


def _gaussian_1d(n: int, sigma_frac: float) -> np.ndarray:
    """Symmetric Gaussian of length ``n`` with sigma = ``sigma_frac * n / 2``."""
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    sigma = max(sigma_frac * (n / 2.0), 1e-12)
    return np.exp(-0.5 * (x / sigma) ** 2)
