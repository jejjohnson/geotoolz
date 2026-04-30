"""Tier A — array-tier entry points for :mod:`xr_toolz.transforms`.

Per design decision D11, every arithmetic submodule grows a duck-array
``axis=`` entry point under ``<module>/array.py``. This module re-exports
the pilot Fourier kernels (``fft``, ``ifft``, ``power_spectrum``) that
operate on raw numpy arrays without going through ``xrft``.

The Tier B wrappers in :mod:`xr_toolz.transforms._src.fourier` keep the
``xrft`` integration (windowing, detrending, frequency coordinates) for
xarray inputs; this module is the numpy-only computational core.
"""

from __future__ import annotations

from xr_toolz.transforms._src.array_fourier import (
    fft,
    ifft,
    power_spectrum,
)


__all__ = [
    "fft",
    "ifft",
    "power_spectrum",
]
