"""Wavelets — 1-D/2-D transforms, spectra, significance, along-track band-pass.

Layer-1 ``Operator`` wrappers live in :mod:`xrops.wavelets.operators`.
"""

from __future__ import annotations

from xrops.wavelets._src.along_track import (
    bandpass_wavelength,
    median_dx_km,
)
from xrops.wavelets._src.wavelet import (
    cwt2,
    morlet2_ft,
    wvlt_cross_spectrum,
    wvlt_power_spectrum,
)
from xrops.wavelets._src.wavelet1d import (
    cwt1d,
    dominant_period_map,
    icwt1d,
    wavelet_significance,
)
from xrops.wavelets._src.wavelet_utils import (
    build_coi_mask,
    geometric_scales,
    scale_to_wavenumber,
    wavenumber_to_scale,
)
from xrops.wavelets.operators import (
    BandpassWavelength,
    WaveletPowerSpectrum,
    WaveletScalogram,
    WaveletSignificance,
)


__all__ = [
    "BandpassWavelength",
    "WaveletPowerSpectrum",
    "WaveletScalogram",
    "WaveletSignificance",
    "bandpass_wavelength",
    "build_coi_mask",
    "cwt1d",
    "cwt2",
    "dominant_period_map",
    "geometric_scales",
    "icwt1d",
    "median_dx_km",
    "morlet2_ft",
    "scale_to_wavenumber",
    "wavelet_significance",
    "wavenumber_to_scale",
    "wvlt_cross_spectrum",
    "wvlt_power_spectrum",
]
