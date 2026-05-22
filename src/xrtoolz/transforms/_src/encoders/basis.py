"""Basis-function encoders — cyclical, Fourier, random Fourier, positional."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cyclical_encode(
    values: ArrayLike,
    period: float,
) -> tuple[NDArray, NDArray]:
    """Sin/cos embedding of a periodic variable.

    Args:
        values: Input values with period ``period``.
        period: Period length (e.g. 365.25 for day-of-year, 24 for hour
            of day, ``2 * np.pi`` for radians).

    Returns:
        ``(sin_component, cos_component)`` pair, each with the same
        shape as ``values``.
    """
    x = 2.0 * np.pi * np.asarray(values) / period
    return np.sin(x), np.cos(x)


def fourier_features(
    values: ArrayLike,
    num_freqs: int,
    scale: float = 1.0,
) -> NDArray:
    """Deterministic Fourier-feature encoding.

    Returns an array of shape ``(..., 2 * num_freqs)`` whose columns
    alternate ``sin(2^k * scale * x)`` and ``cos(2^k * scale * x)``.

    Args:
        values: Input array.
        num_freqs: Number of frequency octaves to use.
        scale: Base frequency.
    """
    values = np.asarray(values)
    freqs = (2.0 ** np.arange(num_freqs)) * scale
    angles = values[..., None] * freqs
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def random_fourier_features(
    values: ArrayLike,
    num_features: int,
    sigma: float = 1.0,
    seed: int | None = None,
) -> NDArray:
    """Random Fourier features in the style of Rahimi & Recht, 2007.

    Approximates the RBF kernel feature map. Returns an array of shape
    ``(..., num_features)``.

    Args:
        values: Input array with the feature dim as the last axis, or a
            1-D scalar coordinate.
        num_features: Output feature dimension (must be even).
        sigma: Bandwidth of the underlying RBF kernel.
        seed: Seed for the RNG used to draw the random frequencies.
    """
    if num_features % 2 != 0:
        raise ValueError("num_features must be even.")
    values = np.asarray(values)
    if values.ndim == 0:
        values = values[None]
    d = values.shape[-1] if values.ndim > 1 else 1
    x = values if values.ndim > 1 else values[..., None]

    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((d, num_features // 2)) / sigma
    projection = x @ omega  # (..., num_features // 2)
    return np.concatenate([np.sin(projection), np.cos(projection)], axis=-1)


def positional_encoding(
    values: ArrayLike,
    num_freqs: int,
    include_input: bool = True,
) -> NDArray:
    """NeRF-style positional encoding.

    Output shape is ``(..., (2 * num_freqs) + include_input)``.

    Args:
        values: Input array.
        num_freqs: Number of octave frequencies.
        include_input: If ``True``, concatenate the raw input as an
            additional column.
    """
    encoded = fourier_features(values, num_freqs=num_freqs, scale=np.pi)
    if include_input:
        values = np.asarray(values)[..., None]
        return np.concatenate([values, encoded], axis=-1)
    return encoded


__all__ = [
    "cyclical_encode",
    "fourier_features",
    "positional_encoding",
    "random_fourier_features",
]


# Silence "unused import" lints for types only touched in annotations.
_ANNOTATION_ONLY: tuple[Any, ...] = (ArrayLike, NDArray)
