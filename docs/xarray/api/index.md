# API Reference

The public API is organised by **subject**, mirroring the three-layer
architecture (primitives → operators → graph). Every page pairs Layer-0
pure functions — `(xr.Dataset, …) → xr.Dataset` — with the Layer-1
`Operator` wrappers that compose them into `Sequential` chains and
`Graph` networks.

!!! tip "How to read these pages"
    Each entry is generated from the source docstring, so the signature,
    argument types, and shapes you see here are exactly what ships in the
    package. Layer-0 functions take a `dim:` (a coordinate name); the
    private numpy kernels behind them take an `axis:` (an integer) and are
    jaxtyped — see the [array-typing convention](../design/conventions/array-typing.md).

## Composition core

| Page | Scope |
|------|-------|
| [Composition](composition.md) | `Operator`, `Sequential`, `Graph`, `Input`, `Node`, `Tap`, `Augment`, `ApplyToEach`, `Signature` |

## Geoprocessing — `geo`

Domain-agnostic xarray geoprocessing. Anything that is *not* domain
physics lives here.

| Page | Scope |
|------|-------|
| [Validation & Coordinates](geo/coords.md) | Coordinate validation, CF-time decoding, CF standard-name renaming, grid resolution |
| [Subsetting, Regions & Masks](geo/subset.md) | Bounding-box / time / region subsetting, the region registry, land/ocean/country masks |
| [CRS & Reprojection](geo/crs.md) | CRS assignment, reprojection, lon/lat ↔ projected coordinates |
| [Climatology & Anomalies](geo/climatology.md) | Climatologies, seasonal cycles, anomalies, detrending |
| [Wavelet Analysis](geo/wavelet.md) | 1-D / 2-D continuous wavelet transforms, scalograms, significance |
| [Extremes](geo/extremes.md) | Block extrema and peaks-over-threshold diagnostics |

## Numerics

| Page | Scope |
|------|-------|
| [Calculus](calc.md) | Finite-difference `gradient`, `divergence`, `curl`, `laplacian` on cartesian / spherical grids |
| [Transforms](transforms.md) | Fourier / wavelet / DCT spectra, spectral fluxes, matrix decompositions, morphology |
| [Encoders](encoders.md) | Coordinate-space, cyclical-time, and Fourier-feature encodings |
| [Interpolation](interpolate.md) | Regridding, coarsening, gap-filling, binning, smoothing, point sampling |

## Domain physics

| Page | Scope |
|------|-------|
| [Oceanography](ocn.md) | Geostrophy, vorticity, kinetic energy, stratification, SSH diagnostics |
| [Atmosphere](atm.md) | Atmospheric and trace-gas physics (planned) |
| [Remote Sensing](rs.md) | Radiometry and spectral indices (planned) |

## Diagnostics & evaluation

| Page | Scope |
|------|-------|
| [Metrics](metrics.md) | Pixel, spectral, multiscale, physical, masked, distributional metrics |
| [Budgets](budgets.md) | Conservation-budget residuals (heat, salt, volume, KE) |

## Bridges & integration

| Page | Scope |
|------|-------|
| [Named-tensor algebra (einx)](einx.md) | `einsum` / `rearrange` / `reduce` / `repeat`, `matmul` / `outer`, `pack`/`unpack` Dataset |
| [Inference](inference.md) | `ModelOp` wrappers for sklearn / JAX / framework-agnostic models |
| [Visualization](viz.md) | Cartopy axes, colormap registry, and the V1–V5 validation panels |
| [Utilities](utils.md) | The scikit-learn ↔ xarray estimator bridge |

## Whole-package dump

For a flat, searchable dump of every public symbol:

::: xrtoolz
    options:
      show_root_heading: false
      show_source: false
      members: []
