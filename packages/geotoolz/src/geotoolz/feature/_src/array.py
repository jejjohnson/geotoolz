"""Tier-A primitives for :mod:`geotoolz.feature` -- no ``GeoTensor`` here.

Dense (raster-out) feature maps over a single-band ``(H, W)`` image,
returned channel-first so the operators can rewrap them like the input.
The sparse detectors (peaks, blobs, corners, Hough) are thin enough
over :mod:`skimage.feature` that their Tier-B operators call skimage
directly and only add the pixel → CRS point conversion.
"""

from __future__ import annotations

import einx
import numpy as np
from jaxtyping import Bool, Float, Num
from skimage.feature import (
    canny,
    multiscale_basic_features,
    structure_tensor,
    structure_tensor_eigenvalues,
)


def canny_edges(
    image: Num[np.ndarray, "h w"],
    *,
    sigma: float = 1.0,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
) -> Bool[np.ndarray, "h w"]:
    """Canny edge map of a single-band image (:func:`skimage.feature.canny`).

    The image is handed to skimage in its own dtype, so thresholds follow
    skimage's dtype-relative convention (input units for integers, with
    ``None`` defaults of 10 % / 20 % of the dtype maximum; absolute
    gradient magnitudes, defaults ``0.1`` / ``0.2``, for floats).
    ``int64`` / ``uint64``, which skimage rejects, is cast to ``float64``
    first and so follows the float convention.

    Args:
        image: ``(H, W)`` image.
        sigma: Width of the Gaussian smoothing kernel, in pixels.
        low_threshold: Lower hysteresis threshold; ``None`` uses skimage's.
        high_threshold: Upper hysteresis threshold; ``None`` uses skimage's.

    Returns:
        Boolean ``(H, W)`` edge map.

    Examples:
        >>> image = np.zeros((16, 16)); image[:, 8:] = 1.0
        >>> bool(canny_edges(image)[:, 7:9].any())
        True
    """
    arr = np.asarray(image)
    if arr.dtype in (np.int64, np.uint64):
        arr = arr.astype(np.float64)
    return canny(
        arr, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold
    )


def structure_tensor_eigvals(
    image: Num[np.ndarray, "h w"], *, sigma: float = 1.0
) -> Float[np.ndarray, "2 h w"]:
    """Eigenvalues of the local structure tensor, largest first.

        A = G_sigma * [[Iy^2, Iy*Ix], [Iy*Ix, Ix^2]],   l1 >= l2

    Args:
        image: ``(H, W)`` image (cast to float).
        sigma: Width of the Gaussian window averaging the gradient
            products, in pixels.

    Returns:
        ``(2, H, W)`` eigenvalue stack.

    Examples:
        >>> structure_tensor_eigvals(np.random.rand(8, 8)).shape
        (2, 8, 8)
    """
    tensor = structure_tensor(np.asarray(image, dtype=float), sigma=sigma)
    return np.asarray(structure_tensor_eigenvalues(tensor))


def multiscale_features(
    image: Num[np.ndarray, "h w"],
    *,
    intensity: bool = True,
    edges: bool = True,
    texture: bool = True,
    sigma_min: float = 0.5,
    sigma_max: float = 16.0,
) -> Float[np.ndarray, "f h w"]:
    """Channel-first :func:`skimage.feature.multiscale_basic_features` stack.

    Args:
        image: ``(H, W)`` image (cast to float).
        intensity: Include Gaussian-smoothed intensity features.
        edges: Include gradient-magnitude (edge) features.
        texture: Include Hessian-eigenvalue (texture) features.
        sigma_min: Smallest smoothing scale, in pixels.
        sigma_max: Largest smoothing scale, in pixels.

    Returns:
        ``(F, H, W)`` feature stack.

    Examples:
        >>> multiscale_features(np.random.rand(8, 8), edges=False,
        ...                     texture=False, sigma_max=2.0).shape[1:]
        (8, 8)
    """
    features = multiscale_basic_features(
        np.asarray(image, dtype=float),
        intensity=intensity,
        edges=edges,
        texture=texture,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        channel_axis=None,
    )
    return einx.id("y x f -> f y x", features)


__all__ = ["canny_edges", "multiscale_features", "structure_tensor_eigvals"]
