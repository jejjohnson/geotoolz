"""Tier-A primitives for :mod:`geotoolz.learn` -- axis bookkeeping, no sklearn.

Resolves how a ``(T, C, H, W)`` / ``(C, H, W)`` / ``(H, W)`` array is
split into scikit-learn's ``(n_samples, n_features)`` table and which
fill value an estimator output carries. The flattening itself is the
shared :func:`geotoolz._src.samples.cube_to_samples`; the estimator
marshalling lives in :mod:`geotoolz.learn._src.estimators`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np


ReshapeMode = Literal["pixel", "pixel_time", "spectral", "temporal", "patch", "custom"]


class ResolvedAxes:
    """Axis labels plus the sample / feature axes of one input."""

    def __init__(
        self,
        *,
        axis_order: tuple[str, ...],
        sample_axes: tuple[int, ...],
        feature_axes: tuple[int, ...],
    ) -> None:
        self.axis_order = axis_order
        self.sample_axes = sample_axes
        self.feature_axes = feature_axes


def resolve_axes(
    ndim: int,
    mode: ReshapeMode,
    sample_axes: tuple[str | int, ...] | None,
    feature_axes: tuple[str | int, ...] | None,
) -> ResolvedAxes:
    """Resolve a reshape ``mode`` into sample and feature axes.

    Axes are labelled ``(T, C, H, W)`` from the right (``(H, W)`` for
    2-D input; extra leading axes are ``X0, X1, ...``). ``mode`` picks
    the sample axes -- ``pixel`` ``(H, W)``, ``pixel_time``
    ``(T, H, W)``, ``spectral`` ``(C,)``, ``temporal`` ``(T,)``,
    ``patch`` the leading axis -- and every other axis is a feature
    axis; ``custom`` takes both lists (labels or integers) explicitly.

    Args:
        ndim: Input rank (``>= 2``).
        mode: Named reshape mode.
        sample_axes: Sample axes for ``mode="custom"``.
        feature_axes: Feature axes for ``mode="custom"``.

    Returns:
        The axis labels and the resolved sample / feature axes.

    Raises:
        ValueError: If the mode needs an axis the input lacks, or the
            custom axes overlap or do not cover the input.

    Examples:
        >>> resolve_axes(3, "pixel", None, None).sample_axes
        (1, 2)
    """
    if mode == "patch":
        return ResolvedAxes(
            axis_order=tuple(str(i) for i in range(ndim)),
            sample_axes=(0,),
            feature_axes=tuple(range(1, ndim)),
        )

    axis_order = _canonical_axis_order(ndim)
    if mode in {"pixel_time", "temporal"} and "T" not in axis_order:
        raise ValueError(
            f"mode={mode!r} needs a time axis, i.e. a 4-D (T, C, H, W) "
            f"input; got a {ndim}-D input"
        )
    if mode == "spectral" and "C" not in axis_order:
        raise ValueError(
            "mode='spectral' needs a band axis, i.e. a 3-D (C, H, W) or 4-D "
            f"(T, C, H, W) input; got a {ndim}-D input"
        )
    if mode == "custom":
        if sample_axes is None or feature_axes is None:
            raise ValueError(
                "sample_axes and feature_axes are required when mode='custom'"
            )
        samples = _axis_indices(sample_axes, axis_order, ndim)
        features = _axis_indices(feature_axes, axis_order, ndim)
    elif mode == "pixel":
        samples = _axis_indices(("H", "W"), axis_order, ndim)
        features = tuple(axis for axis in range(ndim) if axis not in samples)
    elif mode == "pixel_time":
        samples = _axis_indices(("T", "H", "W"), axis_order, ndim)
        features = tuple(axis for axis in range(ndim) if axis not in samples)
    elif mode == "spectral":
        samples = _axis_indices(("C",), axis_order, ndim)
        features = tuple(axis for axis in range(ndim) if axis not in samples)
    elif mode == "temporal":
        samples = _axis_indices(("T",), axis_order, ndim)
        features = tuple(axis for axis in range(ndim) if axis not in samples)
    else:
        raise ValueError(f"Unknown reshape mode: {mode!r}")

    if set(samples) & set(features):
        raise ValueError("sample_axes and feature_axes must be disjoint")
    if len(samples) + len(features) != ndim:
        raise ValueError("sample_axes and feature_axes must cover every input axis")
    return ResolvedAxes(
        axis_order=axis_order,
        sample_axes=samples,
        feature_axes=features,
    )


def _canonical_axis_order(ndim: int) -> tuple[str, ...]:
    if ndim < 2:
        raise ValueError(
            "GeoTensorEstimator expects at least 2 dimensions ((H, W), "
            f"(C, H, W) or (T, C, H, W)); got a {ndim}-D input"
        )
    if ndim == 2:
        return ("H", "W")
    labels = ("T", "C", "H", "W")
    if ndim <= 4:
        return labels[-ndim:]
    extra = tuple(f"X{i}" for i in range(ndim - 4))
    return extra + labels


def _axis_indices(
    axes: Iterable[str | int],
    axis_order: tuple[str, ...],
    ndim: int,
) -> tuple[int, ...]:
    indices: list[int] = []
    for axis in axes:
        if isinstance(axis, str):
            if axis not in axis_order:
                raise ValueError(f"Axis label {axis!r} is not present in this input")
            idx = axis_order.index(axis)
        else:
            idx = axis % ndim
        indices.append(idx)
    if len(set(indices)) != len(indices):
        raise ValueError("Axes must not contain duplicates")
    return tuple(indices)


def output_fill_value(dtype: np.dtype, label_fill_value: int) -> Any:
    """Fill value for an estimator output of ``dtype``.

    ``label_fill_value`` for integer outputs, ``False`` for booleans, and
    ``None`` -- meaning ``NaN`` in a float result -- for everything else.
    """
    if dtype.kind in "iu":
        return label_fill_value
    if dtype.kind == "b":
        return False
    return None


__all__ = ["ReshapeMode", "ResolvedAxes", "output_fill_value", "resolve_axes"]
