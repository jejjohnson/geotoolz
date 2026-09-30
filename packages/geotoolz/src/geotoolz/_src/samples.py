"""Shared pixel flattening: ``(c, h, w)`` cubes <-> ``(n, c)`` sample matrices.

The one implementation of the layout change every per-pixel estimator
(``learn``, ``matched_filter``, ``restore``) needs. Rows are pixel-major
(C order over the sample axes) and columns are the flattened band axes,
i.e. the einx pattern ``"c h w -> (h w) c"`` generalised to any number
of sample and band axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import einx
import numpy as np
from jaxtyping import Shaped
from numpy.typing import DTypeLike


__all__ = ["SampleLayout", "cube_to_samples", "sample_layout", "samples_to_cube"]


@dataclass(frozen=True)
class SampleLayout:
    """How a cube was flattened by :func:`cube_to_samples`.

    Attributes:
        shape: Shape of the original cube.
        sample_axes: Axes (non-negative) flattened into rows, in row-major
            order.
        band_axes: Axes (non-negative) flattened into columns, in order;
            empty when the cube has no band axis (one column).
    """

    shape: tuple[int, ...]
    sample_axes: tuple[int, ...]
    band_axes: tuple[int, ...]

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Sizes of :attr:`sample_axes`, in row-major order."""
        return tuple(self.shape[axis] for axis in self.sample_axes)

    @property
    def band_shape(self) -> tuple[int, ...]:
        """Sizes of :attr:`band_axes`."""
        return tuple(self.shape[axis] for axis in self.band_axes)

    @property
    def n_samples(self) -> int:
        """Number of rows."""
        return prod(self.sample_shape)

    @property
    def n_bands(self) -> int:
        """Number of columns (``1`` when there is no band axis)."""
        return prod(self.band_shape)


def cube_to_samples(
    arr: Shaped[np.ndarray, "*dims"],
    *,
    band_axis: int | tuple[int, ...] = -3,
    sample_axes: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
) -> tuple[Shaped[np.ndarray, "n c"], SampleLayout]:
    """Flatten a cube into an ``(n_samples, n_bands)`` matrix.

    For a canonical ``(c, h, w)`` cube this is ``einx.id("c h w -> (h w)
    c")``; a ``(t, c, h, w)`` stack with the default ``band_axis=-3``
    yields ``t * h * w`` rows. Nodata is not dropped: NaN / fill pixels
    stay in their rows so :func:`samples_to_cube` can put per-row
    results back in place (filter rows with e.g.
    ``np.isfinite(samples).all(axis=1)``).

    Args:
        arr: Input array (any array-like), at least 1-D.
        band_axis: The band (feature) axis, or a tuple of axes flattened
            together into the columns in the given order. ``()`` means
            no band axis: every axis is a sample axis and the matrix has
            one column.
        sample_axes: Axes flattened into rows, in row-major order.
            Defaults to every non-band axis in ascending order.
        dtype: Optional dtype to coerce ``arr`` to first.

    Returns:
        ``(samples, layout)``: the ``(n, c)`` sample matrix and the
        :class:`SampleLayout` needed by :func:`samples_to_cube`.

    Raises:
        ValueError: If the axes are out of range, repeated, or do not
            cover every axis of ``arr`` exactly once.
    """
    values = np.asarray(arr) if dtype is None else np.asarray(arr, dtype=dtype)
    layout = sample_layout(values.shape, band_axis=band_axis, sample_axes=sample_axes)
    names = [f"a{axis}" for axis in range(values.ndim)]
    rows = " ".join(names[axis] for axis in layout.sample_axes)
    cols = " ".join(names[axis] for axis in layout.band_axes)
    samples = einx.id(f"{' '.join(names)} -> ({rows}) ({cols})", values)
    return np.asarray(samples), layout


def samples_to_cube(
    samples: Shaped[np.ndarray, " n"] | Shaped[np.ndarray, "n k"],
    layout: SampleLayout,
) -> np.ndarray:
    """Inverse of :func:`cube_to_samples` for per-row results.

    ``samples`` may have any number of columns ``k`` (e.g. scores,
    components, cluster probabilities), not only the original band
    count. The sample axes are restored in their original (ascending)
    order and the ``k`` columns become a single axis at the position of
    the first band axis -- the leading axis when the layout had no band
    axis. A 1-D ``samples`` restores just the sample axes. For a
    single-band-axis layout and ``k`` equal to the band count this is the
    exact inverse of :func:`cube_to_samples`.

    Args:
        samples: ``(n,)`` or ``(n, k)`` array with ``n ==
            layout.n_samples``.
        layout: The layout returned by :func:`cube_to_samples`.

    Returns:
        The reshaped array.

    Raises:
        ValueError: If ``samples`` is not 1-D / 2-D or its row count does
            not match the layout.
    """
    values = np.asarray(samples)
    if values.ndim not in (1, 2) or values.shape[0] != layout.n_samples:
        raise ValueError(
            f"samples must be (n,) or (n, k) with n={layout.n_samples}; "
            f"got shape {values.shape}"
        )
    names = {axis: f"a{axis}" for axis in layout.sample_axes}
    sizes: dict[str, Any] = {names[axis]: layout.shape[axis] for axis in names}
    rows = " ".join(names[axis] for axis in layout.sample_axes)
    order = [names[axis] for axis in sorted(layout.sample_axes)]
    if values.ndim == 1:
        return np.asarray(einx.id(f"({rows}) -> {' '.join(order)}", values, **sizes))
    out_axis = min(layout.band_axes) if layout.band_axes else 0
    position = sum(axis < out_axis for axis in layout.sample_axes)
    order.insert(position, "k")
    return np.asarray(einx.id(f"({rows}) k -> {' '.join(order)}", values, **sizes))


def sample_layout(
    shape: tuple[int, ...],
    *,
    band_axis: int | tuple[int, ...] = -3,
    sample_axes: tuple[int, ...] | None = None,
) -> SampleLayout:
    """The :class:`SampleLayout` :func:`cube_to_samples` uses for ``shape``.

    Lets a caller that only kept the cube's shape rebuild the layout for
    :func:`samples_to_cube`. Arguments as in :func:`cube_to_samples`.

    Raises:
        ValueError: If the axes are out of range, repeated, or do not
            cover every axis exactly once.
    """
    shape = tuple(int(size) for size in shape)
    ndim = len(shape)
    if ndim < 1:
        raise ValueError("cube_to_samples needs at least a 1-D array")
    bands = (band_axis,) if isinstance(band_axis, int) else tuple(band_axis)
    bands = tuple(_normalise(axis, ndim) for axis in bands)
    if sample_axes is None:
        samples = tuple(axis for axis in range(ndim) if axis not in bands)
    else:
        samples = tuple(_normalise(axis, ndim) for axis in sample_axes)
    if sorted(bands + samples) != list(range(ndim)):
        raise ValueError(
            f"band axes {bands} and sample axes {samples} must cover each of "
            f"the {ndim} axes exactly once"
        )
    return SampleLayout(shape=shape, sample_axes=samples, band_axes=bands)


def _normalise(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of range for a {ndim}-D array")
    return axis % ndim
