"""`spatial.window.Window` — boundary treatment for the patch.

A `spatial.window.Window` returns a weight array shaped like the geometry's patch. The
weights multiply into the patch data on the way in (for
`spatial.aggregation.OverlapAdd`- style aggregations) and form the denominator on the
way out so the overlap-add reconstruction recovers the unweighted field.

Five windows: `spatial.window.Boxcar` (no taper), `spatial.window.Hann`,
`spatial.window.Tukey`, `spatial.window.Gaussian`, and a `spatial.window.Custom` escape
hatch for caller-supplied weight functions.

Taper convention: `spatial.window.Hann` and `spatial.window.Tukey` sample
a periodic taper of period ``N`` at the **pixel centres**, ``t = (k + 1/2) / N``
for ``k = 0 .. N - 1``, applied separably along each axis. A length-``N``
Hann then sums to exactly ``1`` when shifted by ``N / 2`` (constant
overlap-add), so with even ``N`` and ``step = N / 2`` every interior seam is
covered at full weight; and no sample is ``0``, so every pixel a chip covers
— the scene's outer edge included — keeps a non-zero weight. Axes shorter
than 3 samples fall back to boxcar (all ones).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from geopatcher._src._serialize import config_from_fields
from geopatcher._src.spatial.geometry import (
    Geometry as SpatialGeometry,
    Rectangular as SpatialRectangular,
)


class Window:
    """Base for window functions.

    Subclasses implement `weights(geometry) -> Array`. The base class
    handles the trivial `spatial.window.Boxcar` default.
    """

    forbid_in_yaml: ClassVar[bool] = False

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass(eq=False)
class Boxcar(Window):
    """Constant 1.0 — no edge taper."""

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        shape = geom_shape(geometry)
        return np.ones(shape, dtype=np.float64)


@dataclass(eq=False)
class Hann(Window):
    """Hann (raised-cosine) window — the standard overlap-add taper.

    Each axis of length ``n`` is ``sin²(π (k + 1/2) / n)`` for
    ``k = 0 .. n - 1``: a periodic Hann sampled at the pixel centres. It is
    exactly COLA at hop ``n / 2``: with an even patch size and
    ``step = size / 2`` the windows of neighbouring chips sum to ``1``, so
    even un-normalised overlap-add reproduces a constant field across
    interior seams.

    The window is symmetric and never ``0`` (the edge weight is
    ``sin²(π / 2n)``), so merged with `spatial.aggregation.OverlapAdd` every
    covered pixel keeps its value — the scene's first row and column
    included — and ``step == size`` tiles reproduce the field exactly.

    Example:
        >>> from geopatcher import spatial
        >>> w = spatial.window.Hann().weights(
        ...     spatial.geometry.Rectangular(size=(4, 4))
        ... )  # (4, 4) float64
        >>> bool((w > 0).all())
        True

    Axes shorter than 3 samples fall back to boxcar (all ones).
    """

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        shape = geom_shape(geometry)
        return _separable(shape, lambda n: _tapered(n, lambda m: _tukey_1d(m, 1.0)))


@dataclass(eq=False)
class Tukey(Window):
    """Tukey (tapered-cosine) window.

    Each axis of length ``n`` is a periodic Tukey of period ``n`` — cosine
    tapers over a fraction ``alpha`` of the window, flat in between —
    sampled at the pixel centres ``t = (k + 1/2) / n``, the same convention
    as `spatial.window.Hann`: ``alpha = 1.0`` is exactly `spatial.window.Hann`
    and ``alpha = 0.0`` is exactly `spatial.window.Boxcar`. It is COLA at hop
    ``n * (1 - alpha / 2)`` (``0.75 * n`` for the default ``alpha = 0.5``);
    other hops still reconstruct correctly under
    `spatial.aggregation.OverlapAdd`'s normalisation, just not as a partition of unity.

    Like `spatial.window.Hann` it is symmetric and never ``0``, and axes
    shorter than 3 samples fall back to boxcar.

    Args:
        alpha: Fraction of the window occupied by the cosine taper
            (``0.0`` is `spatial.window.Boxcar`, ``1.0`` is `spatial.window.Hann`).
            Defaults to ``0.5``.
    """

    alpha: float = 0.5

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        shape = geom_shape(geometry)
        return _separable(
            shape, lambda n: _tapered(n, lambda m: _tukey_1d(m, self.alpha))
        )

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class Gaussian(Window):
    """Separable Gaussian envelope centred on the patch.

    ``sigma`` is a **fraction of the patch half-width**, not a pixel
    count: along an axis of length ``n`` the standard deviation is
    ``sigma * n / 2`` pixels, centred at ``(n - 1) / 2`` (symmetric). The
    same value therefore gives geometrically similar tapers on different
    patch shapes; the default ``0.5`` puts the patch edge at about two
    standard deviations (edge weight ``≈ exp(-2) ≈ 0.14`` for large patches).
    The weights never reach zero, so a Gaussian has no zero-weight border
    ring, but it is not a partition of unity — rely on
    `spatial.aggregation.OverlapAdd`'s normalisation.

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
class Custom(Window):
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
    """Shape of the weight array a `spatial.window.Window` returns for ``geometry``.

    Public so third-party windows (e.g. ``geotoolz.patch_ops.TriangularWindow``)
    size their weights exactly like the built-in ones.

    Args:
        geometry: A fixed-size geometry — `spatial.geometry.Rectangular`, or any
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
        f"spatial.window.Window weights aren't defined for {type(geometry).__name__} - "
        "only fixed-shape geometries (e.g. spatial.geometry.Rectangular). Use "
        "spatial.window.Boxcar for ragged geometries like "
        "spatial.geometry.RadiusGraph / spatial.geometry.KNNGraph / "
        "spatial.geometry.PolygonIntersection."
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

    A taper of length 1 or 2 is flat anyway (``[1]`` / ``[0.5, 0.5]``);
    boxcar keeps unit weights and stays per-axis, so a ``(1, 256)`` strip
    still tapers along its long axis.
    """
    if n < 3:
        return np.ones(n, dtype=np.float64)
    return np.asarray(taper(n), dtype=np.float64)


def _tukey_1d(n: int, alpha: float) -> np.ndarray:
    """A periodic Tukey of period ``n`` sampled at the pixel centres.

    The odd samples of a length-``2n`` periodic Tukey sit at
    ``t = (k + 1/2) / n``: no sample lands on the taper's zero, and every
    COLA property of the periodic window holds (overlap-add is
    shift-invariant). ``alpha = 1`` is Hann, ``alpha = 0`` all ones.
    """
    from scipy.signal.windows import tukey

    return tukey(2 * n, alpha=alpha, sym=False)[1::2]


def _gaussian_1d(n: int, sigma_frac: float) -> np.ndarray:
    """Symmetric Gaussian of length ``n`` with sigma = ``sigma_frac * n / 2``."""
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    sigma = max(sigma_frac * (n / 2.0), 1e-12)
    return np.exp(-0.5 * (x / sigma) ** 2)
