# Wavelet Analysis

Continuous wavelet transforms in 1-D (Torrence–Compo with Liu-rectified
power) and 2-D (Morlet), plus scalograms, the cone of influence, dominant-
period maps, and significance testing. The CWT of a signal $x(t)$ at scale
$s$ and translation $\tau$ is

$$
W_x(s, \tau) = \int x(t)\,\psi^{*}\!\left(\frac{t-\tau}{s}\right)\frac{dt}{\sqrt{s}}.
$$

## Operators

::: xrops.wavelets.operators.WaveletPowerSpectrum

::: xrops.wavelets.operators.WaveletScalogram

::: xrops.wavelets.operators.WaveletSignificance

::: xrops.wavelets.operators.BandpassWavelength

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.wavelets.cwt1d

::: xrops.wavelets.icwt1d

::: xrops.wavelets.cwt2

::: xrops.wavelets.morlet2_ft

::: xrops.wavelets.build_coi_mask

::: xrops.wavelets.dominant_period_map

::: xrops.wavelets.wvlt_power_spectrum

::: xrops.wavelets.wvlt_cross_spectrum

::: xrops.wavelets.wavelet_significance

::: xrops.wavelets.bandpass_wavelength
