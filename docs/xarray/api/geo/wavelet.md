# Wavelet Analysis

Continuous wavelet transforms in 1-D (Torrence–Compo with Liu-rectified
power) and 2-D (Morlet), plus scalograms, the cone of influence, dominant-
period maps, and significance testing. The CWT of a signal $x(t)$ at scale
$s$ and translation $\tau$ is

$$
W_x(s, \tau) = \int x(t)\,\psi^{*}\!\left(\frac{t-\tau}{s}\right)\frac{dt}{\sqrt{s}}.
$$

## Operators

::: xrops.geo.operators.WaveletPowerSpectrum

::: xrops.geo.operators.WaveletScalogram

::: xrops.geo.operators.WaveletSignificance

::: xrops.geo.operators.BandpassWavelength

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.geo.cwt1d

::: xrops.geo.icwt1d

::: xrops.geo.cwt2

::: xrops.geo.morlet2_ft

::: xrops.geo.build_coi_mask

::: xrops.geo.dominant_period_map

::: xrops.geo.wvlt_power_spectrum

::: xrops.geo.wvlt_cross_spectrum

::: xrops.geo.wavelet_significance

::: xrops.geo.bandpass_wavelength
