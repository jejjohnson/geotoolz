---
status: draft
version: 0.1.0
---

!!! note "These design docs cover the planned operator surface for `xr_toolz`"
    Code snippets use class names directly. In the implementation, the
    submodule layout is:

    - **`xr_toolz.geo`** — domain-agnostic geoprocessing (CRS, validation,
      subset, masks, regrid, discretize, detrend)
    - **`xr_toolz.transforms`** — signal transforms / decompositions /
      encoders (D8)
    - **`xr_toolz.metrics`** — skill scores (D7)
    - **`xr_toolz.kinematics`** — domain-specific physical quantities,
      sub-organized by domain (D9)
    - **`xr_toolz.viz`** — plotting operators (D10)

    See `xr_toolz/__init__.py` for the current export surface.

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

## `transforms` — Signal Transforms, Decompositions, and Encoders

Mathematical transforms over data values and over coordinates. **All encoders, basis expansions, signal transforms, and statistical decompositions live here** — see [decisions.md §D8](../decisions.md) for why this is one module instead of split between `geo` and `transforms`.

Organized by sub-category:

### `transforms.fourier` — Fourier-domain transforms

Wraps `xrft`. Layer 0 functions + Layer 1 Operator wrappers.

```python
# Layer 0
def power_spectrum(da, dim, *, isotropic=False, **kwargs) -> xr.DataArray: ...
def cross_spectrum(da_a, da_b, dim, **kwargs) -> xr.DataArray: ...
def coherence(da_a, da_b, dim, **kwargs) -> xr.DataArray: ...
def stft(da, dim, *, window_size, hop, **kwargs) -> xr.DataArray: ...
def drop_negative_frequencies[T](da: T, dims, *, drop=True) -> T: ...

# Layer 1
class PowerSpectrum(Operator): ...
class CrossSpectrum(Operator): ...
class Coherence(Operator): ...
class STFT(Operator): ...
```

### `transforms.dct` — Cosine / sine transforms

Wraps `scipy.fft`. Layer 0: `dct`, `idct`, `dst`, `idst`. Layer 1: `DCT`, `DST`.

### `transforms.wavelet` — Wavelet transforms

Optional dep `PyWavelets`. Layer 0: `cwt`, `dwt`. Layer 1: `CWT` (`DWT` is dict-returning, kept function-only).

### `transforms.decompose` — Statistical decompositions

Thin presets over `XarrayEstimator` (sklearn-bridge). Stateful (need `.fit()` first) — they are intentionally **not** plain `Dataset → Dataset` operators. EOF uses `mode` axis name; PCA/ICA/NMF/KMeans use `component`.

```python
def pca(n_components, sample_dim, ...) -> XarrayEstimator: ...
def eof(n_components, sample_dim, ...) -> XarrayEstimator: ...   # mode axis
def ica(n_components, sample_dim, ...) -> XarrayEstimator: ...
def nmf(n_components, sample_dim, ...) -> XarrayEstimator: ...
def kmeans(n_clusters, sample_dim, ...) -> XarrayEstimator: ...
```

### `transforms.encoders` — Coordinate and basis encoders

Transform coordinates or values into feature representations. Sub-organized by what they encode:

```python
# transforms/encoders/coord_space.py — spatial coordinate transforms
class LonLatToCartesian(Operator):
    """Convert lon/lat coordinates to 3D Cartesian (x, y, z) on the unit sphere."""
    def __init__(self): ...

class GeocentricToENU(Operator):
    """Geocentric (ECEF) → local East-North-Up at a reference point."""
    def __init__(self, ref_lon: float, ref_lat: float): ...

# transforms/encoders/coord_time.py — temporal coordinate transforms
class CyclicalTimeEncoding(Operator):
    """Add sin/cos cyclical features for temporal components."""
    def __init__(self, components: tuple = ("dayofyear", "hour")): ...

class JulianDate(Operator):
    """Continuous Julian / decimal-year encoding."""
    def __init__(self): ...

# transforms/encoders/basis.py — basis / feature expansions
class FourierFeatures(Operator):
    """NeRF-style positional encoding: [sin(2πσ⁰x), cos(2πσ⁰x), …, sin(2πσ^(L-1)x), cos(2πσ^(L-1)x)]."""
    def __init__(self, coords: list[str], num_freqs: int = 10, scale: float = 1.0): ...

class RandomFourierFeatures(Operator):
    """Random Fourier features (Rahimi & Recht 2007). Approximates RBF kernels for downstream linear models."""
    def __init__(self, coords: list[str], num_features: int = 64, sigma: float = 1.0, seed: int = 0): ...

class PolynomialFeatures(Operator):
    """Polynomial basis expansion over selected coords/variables."""
    def __init__(self, coords: list[str], degree: int = 2): ...
```

All encoders are coordinate-aware *or* value-aware Operators with the standard `Dataset → Dataset` shape — they add new variables / coords carrying the encoded features. Stateless (no `fit` step required), so they slot into `Sequential` directly.

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

## `extremes` — *Deferred*

Extreme-value statistics (block maxima/minima, peaks over threshold, point process counts/stats) live in the standalone **xtremax** package (master_plan Layer 3). `xr_toolz` does not own the implementation.

If a thin xarray wrapper / Operator surface is needed later, it would be added as `xr_toolz.extremes` (parallel to how `xr_toolz.assimilate` wraps filterX), but no work is planned in v0.x.

Until then: use `xtremax` directly, or hand-author a `Lambda(...)` operator over an xtremax function for a one-off pipeline.

---

## `metrics` — Evaluation Metrics

Pixel-level, spectral, multiscale, and distributional skill scores. **Owned implementation, no `xskillscore` dependency** (see [decisions.md §D7](../decisions.md)).

Two-layer module:

- **Layer 0** — pure functions in `xr_toolz/metrics/_src/<family>.py`. Standard signature: `(prediction, reference, *, dim, **kwargs) → xr.DataArray | xr.Dataset | float`. Importable directly for use in scripts, notebooks, and inside other operators.
- **Layer 1** — `Operator` wrappers around the Layer 0 functions. Multi-input: `__call__(prediction, reference) → result`. Each wrapper is one call into its Layer 0 function with config carried on the operator.

Custom skill score: write a Layer 0 function, then either reuse the generic `MetricOp(fn, **config)` wrapper or hand-author an `Operator` subclass.

### Layer 0 — pure functions

```python
# xr_toolz/metrics/_src/pixel.py
def rmse(prediction, reference, *, dim) -> xr.DataArray: ...
def nrmse(prediction, reference, *, dim, normalize="std") -> xr.DataArray: ...
def mae(prediction, reference, *, dim) -> xr.DataArray: ...
def bias(prediction, reference, *, dim) -> xr.DataArray: ...
def correlation(prediction, reference, *, dim) -> xr.DataArray: ...
def murphy_score(prediction, reference, *, dim) -> xr.DataArray: ...
def nash_sutcliffe(prediction, reference, *, dim) -> xr.DataArray: ...
def crps(prediction, reference, *, dim) -> xr.DataArray: ...

# xr_toolz/metrics/_src/spectral.py
def psd_score(prediction, reference, *, dim, **kwargs) -> xr.Dataset: ...
def resolved_scale(prediction, reference, *, dim, threshold=0.5, **kwargs) -> xr.DataArray: ...
def coherence_skill(prediction, reference, *, dim, **kwargs) -> xr.DataArray: ...

# xr_toolz/metrics/_src/multiscale.py
def per_scale_rmse(prediction, reference, *, dim, scales) -> xr.DataArray: ...
def wavelet_rmse(prediction, reference, *, dim, wavelet="db4", level=4) -> xr.DataArray: ...

# xr_toolz/metrics/_src/distributional.py
def ks_statistic(prediction, reference, *, dim) -> xr.DataArray: ...
def wasserstein_1d(prediction, reference, *, dim) -> xr.DataArray: ...
def energy_distance(prediction, reference, *, dim) -> xr.DataArray: ...

# xr_toolz/metrics/_src/masked.py
def masked_rmse(prediction, reference, *, dim, mask) -> xr.DataArray: ...
# ... mask-aware variants of the others
```

### Layer 1 — Operator wrappers

```python
class RMSE(Operator):
    """Root mean squared error. Multi-input operator."""
    def __init__(self, variable: str, dims: list[str]): ...
    def __call__(self, prediction, reference) -> xr.DataArray: ...

class NRMSE(Operator): ...
class MAE(Operator): ...
class Bias(Operator): ...
class Correlation(Operator): ...
class MurphyScore(Operator): ...
class NashSutcliffe(Operator): ...
class CRPS(Operator): ...

class PSDScore(Operator):
    """Spectral coherence-based score."""
    def __init__(self, variable: str, dims: list[str], **kwargs): ...
    def __call__(self, prediction, reference) -> xr.Dataset: ...

class ResolvedScale(Operator):
    """Minimum resolved spatial scale at a given PSD threshold."""
    def __init__(self, variable: str, dims: list[str], threshold: float = 0.5): ...

class CoherenceSkill(Operator): ...
class PerScaleRMSE(Operator): ...
class WaveletRMSE(Operator): ...
class KSStatistic(Operator): ...
class Wasserstein1D(Operator): ...
class EnergyDistance(Operator): ...

class MetricOp(Operator):
    """Generic wrapper: turns any Layer 0 metric function into an Operator."""
    def __init__(self, fn, variable: str, dims: list[str], **kwargs): ...
    def __call__(self, prediction, reference): ...
```

### Adding a custom skill score

```python
# 1. Write a Layer 0 function
def my_score(prediction, reference, *, dim, alpha=1.0):
    return ((prediction - reference) ** alpha).mean(dim=dim)

# 2. Use directly, or wrap as an Operator
op = MetricOp(my_score, variable="ssh", dims=["time"], alpha=2.0)
op(pred, ref)
```

---

## `viz` — Plotting Operators

First-class `Operator` subclasses that return `matplotlib.Figure` / `Axes`. **Documented exception to the `Dataset → Dataset` invariant** — see [decisions.md §D10](../decisions.md). They compose inside `Sequential` as the terminal step, and inside `Graph` as one of N output nodes (the motivating pattern: an evaluation graph that emits scores *and* figures from one symbolic computation).

Sub-organized by what they plot:

```
xr_toolz/viz/_src/
    maps.py         # PlotMap, PlotMapDiff, PlotMapPanel
    series.py       # PlotTimeseries, PlotHovmoller, PlotProfile
    spectral.py     # PlotSpectrum, PlotResolvedScale, PlotCoherence
    eval.py         # PlotMetricsTable, QuicklookPanel
```

Two-layer pattern as elsewhere — pure plotting functions at Layer 0 + `Operator` wrappers at Layer 1.

```python
# Layer 0
def plot_map(da, *, ax=None, projection=None, cmap=None, **kwargs) -> matplotlib.axes.Axes: ...
def plot_timeseries(da, *, ax=None, **kwargs) -> matplotlib.axes.Axes: ...
def plot_spectrum(da, *, ax=None, log=True, **kwargs) -> matplotlib.axes.Axes: ...

# Layer 1
class PlotMap(Operator):
    def __init__(self, variable: str, *, projection=None, cmap=None, figsize=(8, 6)): ...
    def __call__(self, ds) -> matplotlib.figure.Figure: ...

class PlotTimeseries(Operator): ...
class PlotHovmoller(Operator): ...
class PlotSpectrum(Operator): ...
class PlotResolvedScale(Operator): ...
class QuicklookPanel(Operator):
    """Multi-panel quick diagnostic plot — map + timeseries + spectrum."""
    def __init__(self, variable: str, *, dims=None): ...
```

End-to-end pattern (the motivating use case):

```python
preprocess = Sequential([Validate(), Regrid(grid), RemoveClimatology(clim)])

evaluate = Graph(
    inputs={"pred": Input(), "ref": Input()},
    outputs={
        "rmse": RMSE("ssh", dims=["time"])(pred, ref),
        "psd_score": PSDScore("ssh", dims=["lon", "lat"])(pred, ref),
        "fig_map": PlotMap("ssh")(pred),
        "fig_psd": PlotSpectrum("ssh", dims=["lon", "lat"])(pred),
    },
)

results = evaluate(pred=preprocess(raw_pred), ref=preprocess(raw_ref))
# results["rmse"] → xr.DataArray
# results["fig_psd"] → matplotlib.Figure
```

`Sequential` validates that non-`Dataset` returns only appear at the final step; otherwise raises a clear error.

---

## `kinematics` — Domain-Specific Physical Quantities

**Single home for derived physical-quantity operators across all geophysical domains** (atmosphere, ocean, ice, remote sensing). Replaces the earlier-considered split into `xr_toolz.atm/`, `xr_toolz.ocn/`, `xr_toolz.ice/`, `xr_toolz.rs/`. See [decisions.md §D9](../decisions.md).

Sub-organized by domain in one-file-per-domain layout:

```
xr_toolz/kinematics/_src/
    ocean.py        # Streamfunction, GeostrophicVelocities, RelativeVorticity,
                    # KineticEnergy, OkuboWeiss, MixedLayerDepth, …
    atmosphere.py   # WindSpeed, PotentialTemperature, BruntVaisala,
                    # Vorticity, Divergence, …
    ice.py          # SeaIceConcentration, IceAge, IceDrift, …
    remote.py       # NormalizedDifference (NDVI/NDWI/…), RadianceToReflectance,
                    # ColumnAveragingKernel, …
```

Each sub-file ships **Layer 0** pure functions and **Layer 1** Operator wrappers, mirroring the metrics pattern.

```python
# transforms/kinematics/_src/ocean.py
def streamfunction(ds, *, variable="ssh", f0=None, g=None) -> xr.DataArray: ...
def geostrophic_velocities(ds, *, variable="ssh") -> xr.Dataset: ...
def relative_vorticity(ds, *, u_var="u", v_var="v") -> xr.DataArray: ...
def kinetic_energy(ds, *, u_var="u", v_var="v") -> xr.DataArray: ...
def okubo_weiss(ds, *, u_var="u", v_var="v") -> xr.DataArray: ...

class Streamfunction(Operator): ...
class GeostrophicVelocities(Operator): ...
class RelativeVorticity(Operator): ...
class KineticEnergy(Operator): ...
class OkuboWeiss(Operator): ...

# transforms/kinematics/_src/atmosphere.py
def wind_speed(ds, *, u_var="u10", v_var="v10") -> xr.DataArray: ...
def potential_temperature(ds, *, temp_var, pressure_var) -> xr.DataArray: ...

class WindSpeed(Operator): ...
class PotentialTemperature(Operator): ...

# transforms/kinematics/_src/remote.py
def normalized_difference(ds, *, var_a, var_b, name="ndvi") -> xr.DataArray: ...
def radiance_to_reflectance(ds, *, solar_zenith_var, solar_irradiance) -> xr.DataArray: ...

class NormalizedDifference(Operator): ...
class RadianceToReflectance(Operator): ...
```

Uses `metpy` where available, with pure numpy/scipy fallbacks. No hard dependency on `metpy`.

**Disambiguation rule when an operator could fit multiple domains:**
The variable being *operated on* decides the home, not the variable being *produced*. `WindSpeed(u10, v10)` lives in `atmosphere.py` (atmospheric inputs) even when used in an ocean-forcing context.

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
