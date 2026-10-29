# Transforms

Spectral and signal transforms on xarray data — Fourier / wavelet / cosine
spectra, rotary and cross spectra, oceanographic spectral fluxes, matrix
decompositions (PCA / EOF / ICA / NMF / KMeans), and binary-mask
morphology. The numpy compute cores are jaxtyped; the xarray wrappers carry
coordinate and attribute handling.

## Spectra

::: xrops.transforms.operators.PowerSpectrum

::: xrops.transforms.operators.CrossSpectrum

::: xrops.transforms.operators.Coherence

::: xrops.transforms.operators.RotarySpectrum

::: xrops.transforms.operators.STFT

::: xrops.transforms.operators.DCT

::: xrops.transforms.operators.DST

::: xrops.transforms.operators.CWT

## Spectral flux

::: xrops.transforms.operators.KESpectralFlux

::: xrops.transforms.operators.EnstrophySpectralFlux

## Spectral functions

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.transforms.power_spectrum

::: xrops.transforms.isotropic_power_spectrum

::: xrops.transforms.cross_spectrum

::: xrops.transforms.coherence

::: xrops.transforms.rotary_spectrum

::: xrops.transforms.compensated_spectrum

::: xrops.transforms.stft

::: xrops.transforms.dct

::: xrops.transforms.idct

::: xrops.transforms.dst

::: xrops.transforms.idst

::: xrops.transforms.cwt

::: xrops.transforms.dwt

::: xrops.transforms.drop_negative_frequencies

::: xrops.transforms.to_phase

::: xrops.transforms.fit_spectral_slope

::: xrops.transforms.integral_scale

::: xrops.transforms.ke_spectral_flux

::: xrops.transforms.enstrophy_spectral_flux

## Matrix decompositions

These return `xrops.learn.XarrayEstimator` instances configured for the named decomposition (see [scikit-learn on xarray](learn.md)).

::: xrops.transforms.pca

::: xrops.transforms.eof

::: xrops.transforms.ica

::: xrops.transforms.nmf

::: xrops.transforms.kmeans

## Mask morphology

::: xrops.transforms.binary_opening_2d

::: xrops.transforms.binary_closing_2d

::: xrops.transforms.remove_small_holes_2d

::: xrops.transforms.remove_small_objects_2d

::: xrops.transforms.clean_mask

## Axis remapping

`remap_axis` interpolates one dimension of a DataArray onto a new axis of
target values. By default the source axis is the 1-D dimension coordinate
and targets outside its range become `NaN`. Pass `source_coords=` (a
coordinate name or a broadcastable DataArray) when the source levels vary
per column — terrain-following or hybrid vertical grids — and
`extrapolate="nearest"` to hold each column's end value beyond its own
range. See the *Vertical & axis remapping* section of the
[interpolation API](interpolate.md) for the operator-level presets.

::: xrops.transforms.remap_axis
