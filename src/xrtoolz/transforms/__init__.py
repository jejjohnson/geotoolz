"""Generic transforms — Fourier, wavelet, DCT, and learned decompositions.

Pure functions live in :mod:`xrtoolz.transforms._src` and are
``DataArray``-first: each takes ``da: xr.DataArray, dim, ...`` and
returns a ``DataArray``. Layer-1 ``Operator`` wrappers are in
:mod:`xrtoolz.transforms.operators` and accept :class:`xr.Dataset`
inputs for pipeline composition.

Stateful estimators (PCA, EOF, ICA, NMF, KMeans) are returned as
:class:`xrtoolz.utils.XarrayEstimator` instances — they expose
``fit / transform / fit_transform / inverse_transform`` directly,
matching the sklearn API on N-D xarray inputs. ``SklearnOp`` wraps any
sklearn-style estimator as a Layer-1 operator for ``Sequential`` chains.
"""

from xrtoolz.transforms._src.coord_remap import remap_axis, to_phase
from xrtoolz.transforms._src.dct import dct, dst, idct, idst
from xrtoolz.transforms._src.decompose import eof, ica, kmeans, nmf, pca
from xrtoolz.transforms._src.fourier import (
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
from xrtoolz.transforms._src.morphology import (
    binary_closing_2d,
    binary_opening_2d,
    clean_mask,
    remove_small_holes_2d,
    remove_small_objects_2d,
)
from xrtoolz.transforms._src.sklearn_op import SklearnOp
from xrtoolz.transforms._src.wavelet import cwt, dwt


__all__ = [
    "SklearnOp",
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
