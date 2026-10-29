"""Generic transforms — Fourier, wavelet, DCT, and learned decompositions.

Pure functions live in :mod:`xrops.transforms._src` and are
``DataArray``-first: each takes ``da: xr.DataArray, dim, ...`` and
returns a ``DataArray``. Layer-1 ``Operator`` wrappers are in
:mod:`xrops.transforms.operators` and accept :class:`xr.Dataset`
inputs for pipeline composition.

Stateful estimators (PCA, EOF, ICA, NMF, KMeans) are returned as
:class:`xrops.learn.XarrayEstimator` instances — they expose
``fit / transform / fit_transform / inverse_transform`` directly,
matching the sklearn API on N-D xarray inputs, and are operators
themselves. Any other sklearn-style estimator goes through
:class:`xrops.learn.XarrayEstimator` (or `pipekit_array.Estimator`).
"""

from xrops.transforms._src.coord_remap import remap_axis, to_phase
from xrops.transforms._src.dct import dct, dst, idct, idst
from xrops.transforms._src.decompose import eof, ica, kmeans, nmf, pca
from xrops.transforms._src.fourier import (
    coherence,
    compensated_spectrum,
    cross_spectrum,
    drop_negative_frequencies,
    enstrophy_spectral_flux,
    fit_spectral_slope,
    integral_scale,
    isotropic_power_spectrum,
    ke_spectral_flux,
    power_spectrum,
    rotary_spectrum,
    stft,
)
from xrops.transforms._src.morphology import (
    binary_closing_2d,
    binary_opening_2d,
    clean_mask,
    remove_small_holes_2d,
    remove_small_objects_2d,
)
from xrops.transforms._src.wavelet import cwt, dwt
from xrops.transforms.operators import (
    CWT,
    DCT,
    DST,
    STFT,
    Coherence,
    CrossSpectrum,
    CyclicalEncode,
    EncodeTimeCyclical,
    EncodeTimeOrdinal,
    EnstrophySpectralFlux,
    FourierFeatures,
    KESpectralFlux,
    PositionalEncoding,
    PowerSpectrum,
    RandomFourierFeatures,
    RotarySpectrum,
    TimeRescale,
    TimeUnrescale,
)


__all__ = [
    "CWT",
    "DCT",
    "DST",
    "STFT",
    "Coherence",
    "CrossSpectrum",
    "CyclicalEncode",
    "EncodeTimeCyclical",
    "EncodeTimeOrdinal",
    "EnstrophySpectralFlux",
    "FourierFeatures",
    "KESpectralFlux",
    "PositionalEncoding",
    "PowerSpectrum",
    "RandomFourierFeatures",
    "RotarySpectrum",
    "TimeRescale",
    "TimeUnrescale",
    "binary_closing_2d",
    "binary_opening_2d",
    "clean_mask",
    "coherence",
    "compensated_spectrum",
    "cross_spectrum",
    "cwt",
    "dct",
    "drop_negative_frequencies",
    "dst",
    "dwt",
    "enstrophy_spectral_flux",
    "eof",
    "fit_spectral_slope",
    "ica",
    "idct",
    "idst",
    "integral_scale",
    "isotropic_power_spectrum",
    "ke_spectral_flux",
    "kmeans",
    "nmf",
    "pca",
    "power_spectrum",
    "remap_axis",
    "remove_small_holes_2d",
    "remove_small_objects_2d",
    "rotary_spectrum",
    "stft",
    "to_phase",
]
