# Encoders

Feature encodings for coordinates and time — useful as inputs to learned
models. Cyclical time encoding maps a periodic coordinate onto the unit
circle, $\bigl(\sin(2\pi t/T),\cos(2\pi t/T)\bigr)$; Fourier features lift
coordinates into a higher-dimensional sinusoidal basis.

## Operators

::: xrops.transforms.operators.EncodeTimeCyclical

::: xrops.transforms.operators.EncodeTimeOrdinal

::: xrops.transforms.operators.CyclicalEncode

::: xrops.transforms.operators.FourierFeatures

::: xrops.transforms.operators.RandomFourierFeatures

::: xrops.transforms.operators.PositionalEncoding

::: xrops.transforms.operators.TimeRescale

::: xrops.transforms.operators.TimeUnrescale

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.transforms.encoders.encode_time_cyclical

::: xrops.transforms.encoders.encode_time_ordinal

::: xrops.transforms.encoders.cyclical_encode

::: xrops.transforms.encoders.fourier_features

::: xrops.transforms.encoders.random_fourier_features

::: xrops.transforms.encoders.positional_encoding

::: xrops.transforms.encoders.time_rescale

::: xrops.transforms.encoders.time_unrescale

## Coordinate-range conversions

Convert longitude between $[-180, 180)$ and $[0, 360)$, and latitude
between the $[-90, 90]$ and $[0, 180]$ conventions.

::: xrops.transforms.encoders.lon_180_to_360

::: xrops.transforms.encoders.lon_360_to_180

::: xrops.transforms.encoders.lat_90_to_180

::: xrops.transforms.encoders.lat_180_to_90
