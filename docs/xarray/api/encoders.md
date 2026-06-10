# Encoders

Feature encodings for coordinates and time — useful as inputs to learned
models. Cyclical time encoding maps a periodic coordinate onto the unit
circle, $\bigl(\sin(2\pi t/T),\cos(2\pi t/T)\bigr)$; Fourier features lift
coordinates into a higher-dimensional sinusoidal basis.

## Operators

::: xrtoolz.transforms.operators.EncodeTimeCyclical

::: xrtoolz.transforms.operators.EncodeTimeOrdinal

::: xrtoolz.transforms.operators.CyclicalEncode

::: xrtoolz.transforms.operators.FourierFeatures

::: xrtoolz.transforms.operators.RandomFourierFeatures

::: xrtoolz.transforms.operators.PositionalEncoding

::: xrtoolz.transforms.operators.TimeRescale

::: xrtoolz.transforms.operators.TimeUnrescale

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.transforms.encoders.encode_time_cyclical

::: xrtoolz.transforms.encoders.encode_time_ordinal

::: xrtoolz.transforms.encoders.cyclical_encode

::: xrtoolz.transforms.encoders.fourier_features

::: xrtoolz.transforms.encoders.random_fourier_features

::: xrtoolz.transforms.encoders.positional_encoding

::: xrtoolz.transforms.encoders.time_rescale

::: xrtoolz.transforms.encoders.time_unrescale

## Coordinate-range conversions

Convert longitude between $[-180, 180)$ and $[0, 360)$, and latitude
between the $[-90, 90]$ and $[0, 180]$ conventions.

::: xrtoolz.transforms.encoders.lon_180_to_360

::: xrtoolz.transforms.encoders.lon_360_to_180

::: xrtoolz.transforms.encoders.lat_90_to_180

::: xrtoolz.transforms.encoders.lat_180_to_90
