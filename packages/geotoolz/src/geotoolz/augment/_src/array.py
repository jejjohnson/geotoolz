"""Tier-A primitives for :mod:`geotoolz.augment` -- deterministic, no ``GeoTensor``.

The augmentation operators draw their random parameters (factors,
angles, noise fields) from a seeded generator and hand them to these
pure functions, so the physics of each augmentation is testable without
randomness:

* :func:`rot90_transform` -- the affine that goes with ``np.rot90``.
* :func:`rayleigh_weights` -- the λ⁻⁴ haze spectrum.
* :func:`sun_angle_scale` -- the TOA-reflectance factor of a solar-zenith
  change.
* :func:`cloud_alpha` -- a cloud opacity map from a smooth random field.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from jaxtyping import Float
from scipy.ndimage import gaussian_filter


DEFAULT_MIN_WAVELENGTH_NM = 450.0
DEFAULT_MAX_WAVELENGTH_NM = 850.0
CLOUD_ALPHA_EPSILON = 1e-12


def rot90_transform(transform: Affine, height: int, width: int, k: int) -> Affine:
    """Compose ``transform`` with the pixel map of ``np.rot90(..., k)``.

    The affine maps pixel *corners*, so the mirrored axes are anchored at
    the far edge (``width`` / ``height``), not the last pixel index; the
    rotated grid then covers exactly the input footprint.

    Args:
        transform: Affine of the input grid.
        height: Input height in pixels.
        width: Input width in pixels.
        k: Number of counter-clockwise quarter turns (any integer).

    Returns:
        The affine of ``np.rot90(arr, k, axes=(-2, -1))``.

    Examples:
        >>> rot90_transform(Affine.identity(), 4, 6, 2)
        Affine(-1.0, 0.0, 6.0,
               0.0, -1.0, 4.0)
    """
    k %= 4
    if k == 1:
        return transform * Affine(0, -1, width, 1, 0, 0)
    if k == 2:
        return transform * Affine.translation(width, height) * Affine.scale(-1, -1)
    if k == 3:
        return transform * Affine(0, 1, 0, -1, 0, height)
    return transform


def rayleigh_weights(
    wavelengths: Float[np.ndarray, " c"] | None, n_bands: int
) -> Float[np.ndarray, " c"]:
    """Per-band inverse-fourth-power haze weights, peak-normalised.

        w_c = λ_c⁻⁴ / max_c λ_c⁻⁴

    Wavelengths are nanometres; values below ``10`` are taken as
    micrometres and converted. ``None`` assumes a 450-850 nm linspace.

    Args:
        wavelengths: Band-centre wavelengths, or ``None``.
        n_bands: Number of bands.

    Returns:
        ``(n_bands,)`` weights in ``(0, 1]``; the shortest band is ``1``.

    Raises:
        ValueError: If ``wavelengths`` does not have ``n_bands`` values.

    Examples:
        >>> rayleigh_weights(np.array([0.5, 1.0]), 2)
        array([1.    , 0.0625])
    """
    if wavelengths is None:
        wavelengths = np.linspace(
            DEFAULT_MIN_WAVELENGTH_NM, DEFAULT_MAX_WAVELENGTH_NM, n_bands
        )
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if wavelengths.size != n_bands:
        raise ValueError("wavelength metadata must have one value per band.")
    if np.nanmax(wavelengths) < 10.0:
        # Convert micrometers to nanometers; RS visible/NIR values are never <10 nm.
        wavelengths = wavelengths * 1000.0
    weights = 1.0 / np.power(wavelengths, 4)
    return weights / np.nanmax(weights)


def sun_angle_scale(base_sza_deg: float, delta_deg: float) -> float:
    """TOA-reflectance factor for a solar-zenith change.

        s = cos(θ + Δθ) / cos(θ)

    Args:
        base_sza_deg: Solar zenith angle ``θ`` of the scene, degrees.
        delta_deg: Perturbation ``Δθ``, degrees.

    Returns:
        The multiplicative factor ``s``.

    Raises:
        ValueError: If ``θ`` is within floating-point tolerance of 90°.

    Examples:
        >>> round(sun_angle_scale(60.0, -60.0), 6)
        2.0
    """
    denom = np.cos(np.deg2rad(base_sza_deg))
    if np.isclose(denom, 0.0):
        raise ValueError(
            f"solar zenith angle ({base_sza_deg:.2f} deg) is too close to 90 degrees."
        )
    return float(np.cos(np.deg2rad(base_sza_deg + delta_deg)) / denom)


def cloud_alpha(
    field: Float[np.ndarray, "h w"], coverage: float, *, feather: float = 0.0
) -> Float[np.ndarray, "h w"]:
    """Cloud opacity map covering ``coverage`` of the pixels.

    The (random) ``field`` is Gaussian-smoothed with ``feather``,
    min-max normalised to ``f ∈ [0, 1]`` and thresholded at its
    ``1 − coverage`` quantile ``q``; opacity ramps linearly above it:

        α = clip((f − q) / (max f − q), 0, 1)

    Args:
        field: ``(H, W)`` noise field, e.g. ``rng.normal(size=(H, W))``.
        coverage: Fraction of pixels with ``α > 0``, in ``[0, 1]``.
        feather: Gaussian smoothing sigma in pixels; ``0`` disables it.

    Returns:
        ``(H, W)`` opacity in ``[0, 1]``.

    Examples:
        >>> alpha = cloud_alpha(np.random.default_rng(0).normal(size=(64, 64)), 0.25)
        >>> 0.2 < float((alpha > 0).mean()) < 0.3
        True
    """
    if feather:
        field = gaussian_filter(field, sigma=feather, mode="reflect")
    field = (field - field.min()) / (np.ptp(field) + np.finfo(np.float64).eps)
    threshold = np.quantile(field, 1.0 - coverage)
    return np.clip(
        (field - threshold) / (field.max() - threshold + CLOUD_ALPHA_EPSILON),
        0,
        1,
    )


__all__ = ["cloud_alpha", "rayleigh_weights", "rot90_transform", "sun_angle_scale"]
