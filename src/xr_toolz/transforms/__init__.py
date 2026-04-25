"""Generic transforms — Fourier, wavelet, DCT, and learned decompositions.

Pure functions live in :mod:`xr_toolz.transforms._src` and are
``DataArray``-first: each takes ``da: xr.DataArray, dim, ...`` and
returns a ``DataArray``. Layer-1 ``Operator`` wrappers are in
:mod:`xr_toolz.transforms.operators` and accept :class:`xr.Dataset`
inputs for pipeline composition.

Stateful estimators (PCA, EOF, ICA, NMF, KMeans) are returned as
:class:`xr_toolz.utils.XarrayEstimator` instances — they expose
``fit / transform / fit_transform / inverse_transform`` directly,
matching the sklearn API on N-D xarray inputs.
"""

from xr_toolz.transforms._src.dct import dct, dst, idct, idst
from xr_toolz.transforms._src.decompose import eof, ica, kmeans, nmf, pca
from xr_toolz.transforms._src.fourier import (
    coherence,
    cross_spectrum,
    drop_negative_frequencies,
    isotropic_power_spectrum,
    power_spectrum,
    stft,
)
from xr_toolz.transforms._src.wavelet import cwt, dwt


__all__ = [
    "coherence",
    "cross_spectrum",
    "cwt",
    "dct",
    "drop_negative_frequencies",
    "dst",
    "dwt",
    "eof",
    "ica",
    "idct",
    "idst",
    "isotropic_power_spectrum",
    "kmeans",
    "nmf",
    "pca",
    "power_spectrum",
    "stft",
]
