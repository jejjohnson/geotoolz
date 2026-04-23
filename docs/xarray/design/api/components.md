---
status: draft
version: 0.1.0
---

!!! note "Imports in this page are from the original `geo_toolz` layout"
    These design docs were adapted from the `geo_toolz` design study.
    Code snippets use the original feature-based import paths
    (`geo_toolz.<module>`). In `xr_toolz`, the domain-agnostic
    operations live under `xr_toolz.geo.<module>`; physics-specific
    operations live under `xr_toolz.ocn` / `xr_toolz.atm` /
    `xr_toolz.rs`. See `xr_toolz/__init__.py` and
    `xr_toolz/geo/__init__.py` for the current export surface.

# Components — Layer 1 Operators

## `core` — Base Infrastructure

```python
class Operator:        # Base class (see architecture.md §Operator) with dual-mode __call__ (eager + symbolic)
class Sequential:      # Linear pipeline (see architecture.md §Sequential)
class Identity:        # No-op operator
class Lambda:          # Wrap an arbitrary Callable as an Operator
    def __init__(self, fn: Callable, name: str = "lambda"): ...
```

`Lambda` is the escape hatch: any Layer 0 function (or any user function) becomes an operator via `Lambda(partial(my_fn, param=value))`.

---

## `validation` — Data Harmonization

Standardizes coordinate names, ranges, ordering, and metadata across heterogeneous data sources. This is almost always the first step in any pipeline.

```python
class ValidateCoords:
    """Validate and harmonize all spatial and temporal coordinates.
    Renames common variants (longitude→lon, latitude→lat),
    normalizes ranges, sorts, and sets CF-compliant attributes."""
    def __init__(self, lon_range="-180_to_180", sort=True): ...

class RenameCoords:
    """Rename dimensions and coordinates via a mapping dict."""
    def __init__(self, mapping: dict): ...

class SortCoords:
    """Sort dataset along specified dimensions."""
    def __init__(self, dims: list[str] | None = None): ...
```

---

## `crs` — Coordinate Reference Systems

Embedding and transforming CRS metadata. Wraps `rioxarray` and `pyproj`.

```python
class AssignCRS:
    def __init__(self, crs="EPSG:4326"): ...

class Reproject:
    def __init__(self, target_crs, resolution=None): ...
```

---

## `subset` — Spatial and Temporal Selection

Extract regions of interest by bounding box, geometry, or time period.

```python
class SubsetBBox:
    def __init__(self, lon_bnds: tuple, lat_bnds: tuple): ...

class SubsetTime:
    def __init__(self, time_min: str, time_max: str): ...

class SubsetWhere:
    """Apply an arbitrary boolean mask."""
    def __init__(self, mask: xr.DataArray, drop: bool = False): ...

class SelectVariables:
    def __init__(self, variables: list[str]): ...
```

---

## `masks` — Spatial Masks

Add land, ocean, country, or custom masks as coordinate variables. Wraps `regionmask`.

```python
class AddLandMask:
    def __init__(self): ...

class AddOceanMask:
    def __init__(self, ocean: str = "global"): ...

class AddCountryMask:
    def __init__(self, country: str): ...
```

---

## `regrid` — Grid Transformations

Regridding between different spatial resolutions and grid types. Uses scipy and sklearn instead of xesmf.

```python
class Regrid:
    """Regrid to a target grid using scipy interpolation.
    Accepts target coords as arrays or as an xr.Dataset to match."""
    def __init__(self, target_lon, target_lat, method="linear"): ...

class Coarsen:
    """Reduce resolution by integer factor with aggregation."""
    def __init__(self, factor: dict, method: str = "mean"): ...

class Refine:
    """Increase resolution by integer factor with interpolation."""
    def __init__(self, factor: dict, method: str = "linear"): ...
```

---

## `interpolation` — Gap Filling and Resampling

Fill missing data spatially or temporally. Resample time series to different frequencies.

```python
class FillNaN:
    """Fill NaN values using spatial or temporal interpolation."""
    def __init__(self, dim="spatial", method="linear", max_gap=None): ...

class FillNaNRBF:
    """Fill NaN values using radial basis function interpolation."""
    def __init__(self, kernel="thin_plate_spline", neighbors=None): ...

class Resample:
    """Resample along the time dimension."""
    def __init__(self, freq: str = "1D", method: str = "mean"): ...
```

---

## `detrend` — Climatology, Anomalies, and Filtering

Remove temporal trends and seasonal cycles. The stateful pattern (see architecture.md §Split-Object Pattern) applies here: compute climatology from training data, then apply as a stateless operator.

```python
class CalculateClimatology:
    """Compute climatology from a dataset. Returns the climatology, not the anomalies.
    This is the 'learning' operator."""
    def __init__(self, freq="day", smoothing: int | None = 60): ...
    def __call__(self, ds) -> xr.Dataset:  # returns climatology

class RemoveClimatology:
    """Subtract a pre-computed climatology. This is the 'applying' operator."""
    def __init__(self, climatology: xr.Dataset): ...

class AddClimatology:
    """Add a climatology back (inverse of RemoveClimatology)."""
    def __init__(self, climatology: xr.Dataset): ...

class LowpassFilter:
    """Butterworth lowpass filter along a dimension."""
    def __init__(self, dim="time", cutoff=30, order=3): ...
```

---

## `encoders` — Coordinate Encodings

Transform coordinates into feature representations useful for ML models and analysis. Covers spatial coordinate transforms and temporal encodings, including NeRF-style positional encodings.

```python
class LonLatToCartesian:
    """Convert lon/lat coordinates to 3D Cartesian (x, y, z) on the unit sphere."""
    def __init__(self): ...

class CyclicalTimeEncoding:
    """Add sin/cos cyclical features for temporal components."""
    def __init__(self, components: tuple = ("dayofyear", "hour")): ...

class FourierFeatures:
    """Add Fourier feature encodings to specified coordinate dimensions.
    NeRF-style positional encoding: [sin(2πσ⁰x), cos(2πσ⁰x), ..., sin(2πσ^(L-1)x), cos(2πσ^(L-1)x)]"""
    def __init__(self, coords: list[str], num_freqs: int = 10, scale: float = 1.0): ...

class RandomFourierFeatures:
    """Random Fourier feature mapping (Rahimi & Recht 2007).
    Approximates RBF kernels for downstream linear models."""
    def __init__(self, coords: list[str], num_features: int = 64, sigma: float = 1.0, seed: int = 0): ...
```

---

## `discretize` — Binning and Gridding

Convert unstructured observations to gridded products, or coarsen existing grids via binning.

```python
class Bin2D:
    """Bin unstructured 2D observations onto a regular grid."""
    def __init__(self, grid: Grid, statistic: str = "mean"): ...

class Bin2DTime:
    """Bin unstructured observations onto a regular spatiotemporal grid."""
    def __init__(self, grid: SpaceTimeGrid, statistic: str = "mean"): ...

class PointsToGrid:
    """Interpolate scattered point observations to a regular grid."""
    def __init__(self, grid: Grid, method: str = "nearest"): ...
```

---

## `extremes` — Extreme Value Analysis

Block maxima, peaks over threshold, and point process approaches for characterizing distributional tails.

```python
class BlockMaxima:
    def __init__(self, block_size: int = 365, side: str = "center"): ...

class BlockMinima:
    def __init__(self, block_size: int = 365, side: str = "center"): ...

class PeakOverThreshold:
    def __init__(self, quantile: float = 0.98, decluster_freq: int | None = None): ...

class PointProcessCounts:
    def __init__(self, threshold: float, block_size: int = 365): ...

class PointProcessStats:
    def __init__(self, threshold: float, block_size: int = 365, statistic: str = "mean"): ...
```

---

## `spectral` — Spectral Analysis

Power spectral density and spectral transformations for space and time. Wraps `xrft`.

```python
class PSD:
    """Power spectral density along specified dimensions."""
    def __init__(self, variable: str, dims: list[str], scaling="density", detrend="linear"): ...

class IsotropicPSD:
    """Isotropic power spectral density."""
    def __init__(self, variable: str, dims: list[str], scaling="density"): ...

class CrossSpectrum:
    """Cross-spectral density between two variables (multi-input)."""
    def __init__(self, var_a: str, var_b: str, dims: list[str]): ...
```

---

## `metrics` — Evaluation Metrics

Pixel-level, spectral, and multiscale metrics. These are multi-input operators: `(prediction, reference) → Dataset | scalar`.

```python
class RMSE:
    """Root mean squared error. Multi-input operator."""
    def __init__(self, variable: str, dims: list[str]): ...
    def __call__(self, prediction, reference) -> xr.DataArray: ...

class NRMSE:
    def __init__(self, variable: str, dims: list[str]): ...
    def __call__(self, prediction, reference) -> xr.DataArray: ...

class MAE:
    def __init__(self, variable: str, dims: list[str]): ...

class Bias:
    def __init__(self, variable: str, dims: list[str]): ...

class Correlation:
    def __init__(self, variable: str, dims: list[str]): ...

class PSDScore:
    """Spectral coherence-based score."""
    def __init__(self, variable: str, dims: list[str]): ...
    def __call__(self, prediction, reference) -> xr.Dataset: ...

class ResolvedScale:
    """Minimum resolved spatial scale at a given PSD threshold."""
    def __init__(self, variable: str, dims: list[str], threshold: float = 0.5): ...
```

---

## `kinematics` — Physical Quantities

Domain-specific physical transformations. Starting with remote sensing, methane, and oceanography. Uses `metpy` where available, with pure numpy/scipy fallbacks.

```python
# Oceanography
class Streamfunction:
    def __init__(self, variable="ssh", f0=None, g=None): ...

class GeostrophicVelocities:
    def __init__(self, variable="ssh"): ...

class RelativeVorticity:
    def __init__(self, u_var="u", v_var="v"): ...

class KineticEnergy:
    def __init__(self, u_var="u", v_var="v"): ...

class OkuboWeiss:
    def __init__(self, u_var="u", v_var="v"): ...

# Remote sensing
class NormalizedDifference:
    def __init__(self, var_a: str, var_b: str, name: str = "ndvi"): ...

class RadianceToReflectance:
    def __init__(self, solar_zenith_var: str, solar_irradiance: float): ...

# Methane
class ColumnAveragingKernel:
    def __init__(self, pressure_var: str, kernel_var: str): ...

# Atmospheric
class WindSpeed:
    def __init__(self, u_var="u10", v_var="v10"): ...

class PotentialTemperature:
    def __init__(self, temp_var: str, pressure_var: str): ...
```

---

## `sklearn` — scikit-learn Interop

A lightweight wrapper for applying sklearn estimators to xarray objects. This is a utility, not an architectural centerpiece.

```python
class SklearnOp(Operator):
    """Wrap any sklearn estimator as a geo_toolz Operator.
    Handles the xarray ↔ numpy marshalling.

    When xarray_sklearn is installed, delegates to XarrayEstimator for
    NaN policy support and richer metadata round-tripping.
    """
    def __init__(self, estimator, sample_dim, new_feature_dim="component", nan_policy="propagate"): ...
```

`SklearnOp` works in three modes depending on available packages:

| Installed | Behaviour |
|---|---|
| Neither `xarray_sklearn` nor `xrpatcher` | Built-in `to_2d`/`from_2d` marshalling. NaN policy limited to `"propagate"`. |
| `xarray_sklearn` | Delegates to `XarrayEstimator` — full NaN policies (`"propagate"`, `"raise"`, `"mask"`), Dataset column-concat, Pipeline/GridSearchCV compat. |
| `xarray_sklearn` + `xrpatcher` | Same as above, plus users can wrap `SklearnOp` in a patch loop via `XRDAPatcher` for per-region fitting and memory-bounded inference. |

See [../examples/](../examples/) for usage patterns covering all three modes.
