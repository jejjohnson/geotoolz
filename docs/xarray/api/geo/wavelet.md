# Wavelet Analysis

Continuous wavelet transforms in 1-D (Torrence–Compo with Liu-rectified
power) and 2-D (Morlet), plus scalograms, the cone of influence, dominant-
period maps, and significance testing. The CWT of a signal $x(t)$ at scale
$s$ and translation $\tau$ is

$$
W_x(s, \tau) = \int x(t)\,\psi^{*}\!\left(\frac{t-\tau}{s}\right)\frac{dt}{\sqrt{s}}.
$$

## Operators

::: xrtoolz.geo.operators.WaveletPowerSpectrum

::: xrtoolz.geo.operators.WaveletScalogram

::: xrtoolz.geo.operators.WaveletSignificance

::: xrtoolz.geo.operators.BandpassWavelength

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.geo.cwt1d

::: xrtoolz.geo.icwt1d

::: xrtoolz.geo.cwt2

::: xrtoolz.geo.morlet2_ft

::: xrtoolz.geo.build_coi_mask

::: xrtoolz.geo.dominant_period_map

::: xrtoolz.geo.wvlt_power_spectrum

::: xrtoolz.geo.wvlt_cross_spectrum

::: xrtoolz.geo.wavelet_significance

::: xrtoolz.geo.bandpass_wavelength
