"""Tier A — array kernels for Fourier-domain transforms.

Pure-array entry points for the canonical FFT and power-spectrum
operations. Tier B (xarray, ``dim=``) wrappers in
:mod:`xr_toolz.transforms._src.fourier` add coord/attr handling and the
``xrft`` integration; this module is the numpy-only computational core
that callers can drop down to when they have raw arrays.

Backend: numpy. JAX / CuPy variants are out of scope for the pilot.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


Axis = int | tuple[int, ...]
FFTNorm = Literal["backward", "ortho", "forward"] | None


def fft(
    x: ArrayLike,
    *,
    axis: Axis = -1,
    norm: FFTNorm = None,
) -> NDArray[np.complexfloating]:
    """N-dimensional discrete Fourier transform along ``axis``.

    Args:
        x: Input array.
        axis: Axis or axes to transform.
        norm: Normalization mode forwarded to :func:`numpy.fft.fftn`
            (``None``, ``"ortho"``, ``"forward"``, ``"backward"``).

    Returns:
        Complex-valued FFT with the same shape as ``x``.
    """
    arr = np.asarray(x)
    axes = [axis] if isinstance(axis, int) else list(axis)
    return np.fft.fftn(arr, axes=axes, norm=norm)


def ifft(
    x: ArrayLike,
    *,
    axis: Axis = -1,
    norm: FFTNorm = None,
) -> NDArray[np.complexfloating]:
    """Inverse N-dimensional FFT along ``axis``."""
    arr = np.asarray(x)
    axes = [axis] if isinstance(axis, int) else list(axis)
    return np.fft.ifftn(arr, axes=axes, norm=norm)


def power_spectrum(
    x: ArrayLike,
    *,
    axis: Axis = -1,
    d: float | tuple[float, ...] = 1.0,
    norm: FFTNorm = "ortho",
) -> tuple[NDArray[np.floating], tuple[NDArray[np.floating], ...]]:
    """Power spectrum of ``x`` along ``axis``.

    Computes the squared magnitude of the discrete Fourier transform,
    ``|FFT(x)|**2``. No windowing or detrending is applied — the Tier B
    wrapper handles those via ``xrft``. This minimal raw-array entry
    point also does not apply any additional density scaling by sample
    spacing beyond constructing the returned frequency coordinates;
    pass ``norm="ortho"`` (default) to keep units consistent across
    sample counts.

    Args:
        x: Input array.
        axis: Axis or axes to transform.
        d: Sample spacing along each transformed axis (scalar or
            per-axis tuple). Used to build the returned frequency
            coordinates.
        norm: FFT normalization mode (see :func:`fft`). ``"ortho"`` by
            default so the spectrum has consistent units across sample
            counts.

    Returns:
        Tuple ``(power, freqs)`` where ``power`` is the real-valued
        ``|FFT(x)|**2`` array and ``freqs`` is a tuple of 1-D frequency
        coordinate arrays — one per transformed axis, in the order of
        ``axis``.
    """
    arr = np.asarray(x)
    axes = [axis] if isinstance(axis, int) else list(axis)
    if isinstance(d, (int, float)):
        ds = (float(d),) * len(axes)
    else:
        ds = tuple(float(s) for s in d)
        if len(ds) != len(axes):
            raise ValueError(
                f"d tuple length ({len(ds)}) does not match number of axes "
                f"({len(axes)})."
            )
    spectrum = np.fft.fftn(arr, axes=axes, norm=norm)
    power = (spectrum.conj() * spectrum).real
    freqs = tuple(
        np.fft.fftfreq(arr.shape[ax], d=spacing)
        for ax, spacing in zip(axes, ds, strict=True)
    )
    return power, freqs


__all__ = [
    "fft",
    "ifft",
    "power_spectrum",
]
