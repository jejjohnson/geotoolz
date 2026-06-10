# Transforms

Spectral and signal transforms on xarray data — Fourier / wavelet / cosine
spectra, rotary and cross spectra, oceanographic spectral fluxes, matrix
decompositions (PCA / EOF / ICA / NMF / KMeans), and binary-mask
morphology. The numpy compute cores are jaxtyped; the xarray wrappers carry
coordinate and attribute handling.

## Spectra

::: xrtoolz.transforms.operators.PowerSpectrum

::: xrtoolz.transforms.operators.CrossSpectrum

::: xrtoolz.transforms.operators.Coherence

::: xrtoolz.transforms.operators.RotarySpectrum

::: xrtoolz.transforms.operators.STFT

::: xrtoolz.transforms.operators.DCT

::: xrtoolz.transforms.operators.DST

::: xrtoolz.transforms.operators.CWT

## Spectral flux

::: xrtoolz.transforms.operators.KESpectralFlux

::: xrtoolz.transforms.operators.EnstrophySpectralFlux

## Spectral functions

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.transforms.power_spectrum

::: xrtoolz.transforms.isotropic_power_spectrum

::: xrtoolz.transforms.cross_spectrum

::: xrtoolz.transforms.coherence

::: xrtoolz.transforms.rotary_spectrum

::: xrtoolz.transforms.compensated_spectrum

::: xrtoolz.transforms.stft

::: xrtoolz.transforms.dct

::: xrtoolz.transforms.idct

::: xrtoolz.transforms.dst

::: xrtoolz.transforms.idst

::: xrtoolz.transforms.cwt

::: xrtoolz.transforms.dwt

::: xrtoolz.transforms.drop_negative_frequencies

::: xrtoolz.transforms.to_phase

::: xrtoolz.transforms.fit_spectral_slope

::: xrtoolz.transforms.integral_scale

::: xrtoolz.transforms.ke_spectral_flux

::: xrtoolz.transforms.enstrophy_spectral_flux

## Matrix decompositions

These return `XarrayEstimator` instances (the scikit-learn ↔ xarray
bridge) configured for the named decomposition.

::: xrtoolz.transforms.pca

::: xrtoolz.transforms.eof

::: xrtoolz.transforms.ica

::: xrtoolz.transforms.nmf

::: xrtoolz.transforms.kmeans

::: xrtoolz.transforms.SklearnOp

## Mask morphology

::: xrtoolz.transforms.binary_opening_2d

::: xrtoolz.transforms.binary_closing_2d

::: xrtoolz.transforms.remove_small_holes_2d

::: xrtoolz.transforms.remove_small_objects_2d

::: xrtoolz.transforms.clean_mask

## Axis remapping

::: xrtoolz.transforms.remap_axis
