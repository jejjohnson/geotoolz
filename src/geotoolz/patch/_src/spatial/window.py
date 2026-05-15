"""`SpatialWindow` — boundary treatment for the patch.

A `SpatialWindow` returns a weight array shaped like the geometry's patch. The
weights multiply into the patch data on the way in (for `SpatialOverlapAdd`-
style aggregations) and form the denominator on the way out so the
overlap-add reconstruction recovers the unweighted field.

Five windows: `SpatialBoxcar` (no taper), `SpatialHann`, `SpatialTukey`,
`SpatialGaussian`, and a `SpatialCustom` escape hatch for caller-supplied
weight functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from geotoolz.patch._src.spatial.geometry import SpatialGeometry, SpatialRectangular


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
        shape = _geom_shape(geometry)
        return np.ones(shape, dtype=np.float64)


@dataclass(eq=False)
class SpatialHann(SpatialWindow):
    """SpatialHann (raised-cosine) window — the standard overlap-add taper."""

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        shape = _geom_shape(geometry)
        return _separable(shape, lambda n: np.hanning(n))


@dataclass(eq=False)
class SpatialTukey(SpatialWindow):
    """SpatialTukey (tapered-cosine) window.

    Args:
        alpha: Fraction of the window occupied by the cosine taper
            (``0.0`` is SpatialBoxcar, ``1.0`` is SpatialHann). Defaults to ``0.5``.
    """

    alpha: float = 0.5

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        from scipy.signal.windows import tukey

        shape = _geom_shape(geometry)
        return _separable(shape, lambda n: tukey(n, alpha=self.alpha, sym=False))

    def get_config(self) -> dict[str, Any]:
        return {"alpha": self.alpha}


@dataclass(eq=False)
class SpatialGaussian(SpatialWindow):
    """SpatialGaussian envelope of standard deviation ``sigma`` (in patch units).

    Sigma is interpreted as a fraction of the patch half-width along each
    axis, so the same value gives geometrically-similar tapers on
    different patch shapes.
    """

    sigma: float = 0.5

    def weights(self, geometry: SpatialGeometry) -> np.ndarray:
        shape = _geom_shape(geometry)
        return _separable(shape, lambda n: _gaussian_1d(n, self.sigma))

    def get_config(self) -> dict[str, Any]:
        return {"sigma": self.sigma}


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


def _geom_shape(geometry: SpatialGeometry) -> tuple[int, ...]:
    """Best-effort shape extraction for a geometry's weight array."""
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


def _gaussian_1d(n: int, sigma_frac: float) -> np.ndarray:
    """Symmetric Gaussian of length ``n`` with sigma = ``sigma_frac * n / 2``."""
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    sigma = max(sigma_frac * (n / 2.0), 1e-12)
    return np.exp(-0.5 * (x / sigma) ** 2)
