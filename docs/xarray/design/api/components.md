---
status: draft
version: 0.1.0
---

!!! note "These design docs cover the planned operator surface for `xr_toolz`"
    Code snippets use class names directly. In the implementation, the
    submodule layout is:

    - **`xr_toolz.geo`** — domain-agnostic geoprocessing (CRS, validation,
      subset, masks, detrend)
    - **`xr_toolz.interpolate`** — value resampling: regrid, gap-fill,
      bin, coord-axis remap, time resample, smooth, learned downscale
      (D12)
    - **`xr_toolz.transforms`** — signal transforms / decompositions /
      encoders (D8)
    - **`xr_toolz.metrics`** — skill scores (D7)
    - **`xr_toolz.kinematics`** — domain-specific physical quantities,
      sub-organized by domain (D9)
    - **`xr_toolz.viz`** — plotting operators (D10)

    See `xr_toolz/__init__.py` for the current export surface.

!!! note "Type contract — three tiers (D11)"
    Every module with array-meaningful math exposes three tiers:

    - **Tier A — `<module>.array`** — array functions (numpy / JAX / numba / optionally CuPy). Take/return arrays. Use `axis=`.
    - **Tier B — Layer 0 xarray** — `xr.DataArray` for single-variable, `xr.Dataset` + variable selectors for multi-variable. Use `dim=`.
    - **Tier C — Layer 1 Operator** — input is `xr.Dataset` (or two for multi-input). Output is usually `xr.Dataset`, may narrow to `xr.DataArray` or scalar for reductions (e.g., metrics), or `matplotlib.Figure / Axes` for terminal viz (D10). The only tier `Sequential` and `Graph` see.

    Modules whose math is inherently coord/attr-manipulation (`validation`, `crs`, `subset`, `masks`) skip Tier A — Tier B takes `xr.Dataset` directly. Value-resampling functionality lives under `xr_toolz.interpolate` (D12), not separate `regrid` / `interpolation` / `discretize` modules. See [architecture.md §Type Contract](../architecture.md) and [decisions.md §D11](../decisions.md).

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

## `interpolate` — Resampling, Aggregation, Smoothing

Unified home for **value resampling onto new coordinate locations**: regridding, gap-filling, binning, coord-axis remapping, time resampling, smoothing, and learned super-resolution. Subsumes what the v0.1 design split across `regrid`, `interpolation`, and `discretize`. Three tiers per D11; Tier A is rich here — most algorithms are pure array math (linear / cubic / RBF / kriging / FFT-based filters).

Sub-organized by source/target structure:

```
xr_toolz/interpolate/
    array.py
    _src/
        grid_to_grid.py    # gridded → gridded (same domain, different grid)
        grid_to_points.py  # gridded → unstructured (extract at points / tracks)
        points_to_grid.py  # unstructured → gridded (interpolation)
        binning.py         # unstructured → gridded (aggregation)
        gap_fill.py        # in-place hole filling (same grid, NaN → value)
        coord_remap.py     # remap along any coord axis (vertical canonical, also temporal phase, …)
        resample.py        # time-axis resampling
        smooth.py          # along-axis denoising
        downscale.py       # learned super-resolution / aggregation via ModelOp (Downscale, Upscale)
```

Modules outside `interpolate` that handle adjacent concerns:

- `crs.Reproject` — CRS-aware regridding (calls into `interpolate.Regrid` internally).
- `transforms.encoders.coord_space` / `coord_time` — coord *relabeling* (`LonLatToCartesian`, `JulianDate`), not value resampling.
- `assimilate.fusion` — multi-source data fusion (OI, weighted, kriged) — uses prior/obs-error machinery from `assimilate`.

### `interpolate.grid_to_grid` — Grid-to-grid resampling

Deterministic, same-domain regridding. Wraps scipy / sklearn interpolators (no xesmf hard dep).

```python
# Tier A — array
def regrid(values: ArrayLike, *, src_coords, tgt_coords, method: str = "linear", fill_value=np.nan) -> Array: ...
def coarsen(values: ArrayLike, *, factor: dict[int, int], reduction: str = "mean") -> Array: ...
def refine(values: ArrayLike, *, factor: dict[int, int], method: str = "linear") -> Array: ...

# Tier B — Layer 0 xarray
def regrid(da: xr.DataArray, *, target: xr.Dataset | Grid, method: str = "linear",
           fill_value=np.nan, extrap: bool = False) -> xr.DataArray: ...
def coarsen(da: xr.DataArray, *, factor: dict[str, int], reduction: str = "mean") -> xr.DataArray: ...
def refine(da: xr.DataArray, *, factor: dict[str, int], method: str = "linear") -> xr.DataArray: ...

# Tier C — Operator
class Regrid(Operator):
    def __init__(self, target: "Grid | xr.Dataset", *, method: str = "linear",
                 fill_value=np.nan, extrap: bool = False): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...

class Coarsen(Operator):
    """fine → coarse via deterministic aggregation."""
    def __init__(self, factor: dict[str, int], *, reduction: str = "mean"): ...

class Refine(Operator):
    """coarse → fine via deterministic interpolation (bilinear, bicubic, Lanczos)."""
    def __init__(self, factor: dict[str, int], *, method: str = "linear"): ...
```

### `interpolate.grid_to_points` — Extract at scattered locations

Sample a gridded field at scattered (lon, lat[, time, depth]) locations. Useful for satellite/glider/ship match-ups against a model field.

```python
# Tier C
class SampleAtPoints(Operator):
    def __init__(self, points: "xr.Dataset | gpd.GeoDataFrame",
                 *, method: str = "linear", time_match: str = "nearest"): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...

class AlongTrack(SampleAtPoints):
    """Specialization: extract along a moving track (satellite, ship, glider)."""
```

### `interpolate.points_to_grid` — Unstructured → grid (interpolation)

Project scattered observations onto a regular grid using a smooth interpolant (RBF / NN / IDW / kriging).

```python
# Tier A
def scatter_to_grid(values: ArrayLike, *, src_xy: ArrayLike, tgt_grid: ArrayLike,
                    method: str = "rbf", **kwargs) -> Array: ...
def kriging(values: ArrayLike, *, src_xy: ArrayLike, tgt_grid: ArrayLike,
            variogram: str = "matern", **kwargs) -> Array: ...

# Tier C
class ScatterToGrid(Operator):
    def __init__(self, target: "Grid", *, method: str = "rbf", **kwargs): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...

class Kriging(ScatterToGrid):
    def __init__(self, target: "Grid", *, variogram: str = "matern", **kwargs): ...
```

### `interpolate.binning` — Unstructured → grid (aggregation)

Project scattered observations onto a regular grid using bin-statistic aggregation. Different from `points_to_grid` in that it's a deterministic reduction, not an interpolation.

```python
# Tier C
class Bin2D(Operator):
    def __init__(self, target: "Grid", *, statistic: str = "mean",
                 min_count: int | None = None): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...

class BinND(Operator): ...
class Bin2DTime(Operator):
    """Bin unstructured observations onto a regular spatiotemporal grid."""
```

### `interpolate.gap_fill` — In-place hole filling

Same grid, fill NaN holes via interpolation. (Distinguished from `points_to_grid` because the source structure is the *same* grid, just with missing values.)

```python
# Tier C
class FillNaN(Operator):
    def __init__(self, *, dim: str = "spatial", method: str = "linear",
                 max_gap: float | None = None): ...

class FillNaNRBF(Operator):
    def __init__(self, *, kernel: str = "thin_plate_spline",
                 neighbors: int | None = None): ...

class FillNaNKriging(Operator):
    def __init__(self, *, variogram: str = "matern"): ...
```

### `interpolate.coord_remap` — Remap along a coordinate axis

Generic operation: remap a field defined on coordinate axis A onto a new axis B, where B may itself depend on the data. **Vertical coord remapping (depth ↔ σ ↔ isopycnal ↔ pressure-level) is the canonical usage**, but the same primitive handles temporal phase remapping (e.g., to-diurnal-cycle phase), curvilinear-orthogonal coord transforms, and Lagrangian ↔ Eulerian rebinning. The generic `RemapAxis` is the workhorse; named subclasses are convenience presets.

```python
# Tier A
def remap_axis(values: ArrayLike, *, src_axis: ArrayLike, tgt_axis: ArrayLike,
               method: str = "linear") -> Array: ...

# Tier B
def remap_axis(da: xr.DataArray, *, src_axis: xr.DataArray, tgt_axis: xr.DataArray,
               method: str = "linear") -> xr.DataArray: ...

# Tier C — generic
class RemapAxis(Operator):
    """Remap along an arbitrary coord axis. The target axis may depend on the data
    (e.g., σ-coords are a function of SSH and bathymetry)."""
    def __init__(self, *, src_dim: str, tgt_dim: str,
                 tgt_axis: "xr.DataArray | Callable[[xr.Dataset], xr.DataArray]",
                 method: str = "linear"): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...

# Tier C — vertical specializations (canonical)
class ToSigma(RemapAxis):
    """Depth → σ for ocean models  (σ = (z + ssh) / (h + ssh))."""
    def __init__(self, *, ssh_var: str = "ssh", depth_var: str = "depth",
                 bathy_var: str = "h", sigma_levels: np.ndarray,
                 target_dim: str = "sigma"): ...

class FromSigma(RemapAxis): ...

class ToIsopycnal(RemapAxis):
    """Depth → density-following (isopycnal) coords."""
    def __init__(self, *, density_var: str, target_density: np.ndarray,
                 target_dim: str = "isopycnal"): ...

class ToPressureLevels(RemapAxis):
    """Native vertical → standard pressure levels for atmospheric data."""
    def __init__(self, *, pressure_var: str, target_pressure: np.ndarray,
                 target_dim: str = "plev"): ...

class ToHeight(RemapAxis):
    """Pressure → geopotential height for atmospheric data."""

# Tier C — temporal specialization
class ToPhase(RemapAxis):
    """Time → cycle phase (diurnal, annual, ENSO, …) — folds a time series onto
    a reference cycle. Distinct from time encoders (which add features)."""
    def __init__(self, *, time_var: str = "time", period: str = "1D",
                 phase_bins: int = 24, target_dim: str = "phase"): ...
```

### `interpolate.resample` — Time-axis resampling

```python
# Tier C
class Resample(Operator):
    """Down-sample a time series via aggregation (e.g., hourly → daily mean)."""
    def __init__(self, freq: str, *, reduction: str = "mean"): ...

class Upsample(Operator):
    """Up-sample a time series via interpolation (e.g., daily → hourly linear)."""
    def __init__(self, freq: str, *, method: str = "linear"): ...
```

### `interpolate.smooth` — Along-axis denoising

Sequential along a dimension (typically time, but applies to any axis). Note: the previous design's `detrend.LowpassFilter` is the same primitive; canonical home is here. `detrend` retains only the climatology/anomaly tools.

```python
# Tier A
def moving_average(values: ArrayLike, *, axis: int, window: int) -> Array: ...
def gaussian_smooth(values: ArrayLike, *, axis: int, sigma: float) -> Array: ...
def lowpass(values: ArrayLike, *, axis: int, cutoff: float, order: int = 3) -> Array: ...

# Tier C
class MovingAverage(Operator):
    def __init__(self, *, dim: str, window: int): ...

class GaussianSmooth(Operator):
    def __init__(self, *, dim: str, sigma: float): ...

class LowpassFilter(Operator):
    """Butterworth lowpass filter along a dimension."""
    def __init__(self, *, dim: str = "time", cutoff: float = 30.0, order: int = 3): ...

class KalmanSmoother(Operator):
    """Forward-backward (Rauch-Tung-Striebel) smoother given a state-space model.
    The state-space model itself is owned by `assimilate`."""
    def __init__(self, *, state_space, dim: str = "time"): ...
```

### `interpolate.downscale` — Learned super-resolution

Coarse → fine via a *learned* model (CNN, BCSD regression, GAN, diffusion, …). Wraps a `ModelOp` (see [architecture.md §Inference](../architecture.md)) — the user supplies the fitted model; `Downscale` does the xarray ↔ array marshalling, optional patch tiling, and reconstruction. **Deterministic upsampling is `Refine`; this is the learned counterpart.**

```python
# Tier C
class Downscale(Operator):
    """Learned coarse → fine super-resolution. Wraps a ModelOp."""
    def __init__(self, model_op: ModelOp, *, target: "Grid",
                 patch_size: tuple[int, int] | None = None,
                 overlap: int = 0): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...

class Upscale(Operator):
    """Learned fine → coarse aggregation (e.g., subgrid-scale parameterization
    surrogates). Symmetric to Downscale; less common but supported."""
    def __init__(self, model_op: ModelOp, *, target: "Grid"): ...
```

`Downscale` deliberately requires a fitted `ModelOp` — there is no "default" super-resolution and the user brings their own model (sklearn, JAX/Equinox, PyTorch). For deterministic interpolation, use `Refine`.

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

Three-tier per D11. Tier A wraps `numpy.fft` / `scipy.fft` (numpy default; `jax.numpy.fft` variant added per-function where useful). Tier B wraps `xrft` for label preservation. Tier C wraps Tier B.

```python
# Tier A — array (numpy default; jax variant per-function)
def power_spectrum(x: ArrayLike, *, axis: int, isotropic: bool = False, **kwargs) -> Array: ...
def cross_spectrum(x: ArrayLike, y: ArrayLike, *, axis: int, **kwargs) -> Array: ...
def coherence(x: ArrayLike, y: ArrayLike, *, axis: int, **kwargs) -> Array: ...
def stft(x: ArrayLike, *, axis: int, window_size: int, hop: int, **kwargs) -> Array: ...

# Tier B — Layer 0 xarray (DataArray in, DataArray out)
def power_spectrum(da: xr.DataArray, *, dim: str, isotropic: bool = False, **kwargs) -> xr.DataArray: ...
def cross_spectrum(da_a: xr.DataArray, da_b: xr.DataArray, *, dim: str, **kwargs) -> xr.DataArray: ...
def coherence(da_a: xr.DataArray, da_b: xr.DataArray, *, dim: str, **kwargs) -> xr.DataArray: ...
def stft(da: xr.DataArray, *, dim: str, window_size: int, hop: int, **kwargs) -> xr.DataArray: ...
def drop_negative_frequencies[T: xr.DataArray | xr.Dataset](da: T, *, dims, drop: bool = True) -> T: ...

# Tier C — Operator (Dataset in, Dataset out)
class PowerSpectrum(Operator):
    def __init__(self, variable: str, *, dims: list[str], isotropic: bool = False, **kwargs): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...

class CrossSpectrum(Operator):
    def __init__(self, variable: str, *, dims: list[str], **kwargs): ...
    def __call__(self, ds_a: xr.Dataset, ds_b: xr.Dataset) -> xr.Dataset: ...

class Coherence(Operator): ...
class STFT(Operator): ...
```

### `transforms.dct` — Cosine / sine transforms

Three-tier per D11. Tier A wraps `scipy.fft.dct` / `idct` / `dst` / `idst` (numpy default; numba-jitted small-N variants where Python overhead matters).

```python
# Tier A — array
def dct(x: ArrayLike, *, axis: int = -1, type: int = 2, norm: str | None = None) -> Array: ...
def idct(x: ArrayLike, *, axis: int = -1, type: int = 2, norm: str | None = None) -> Array: ...
def dst(x: ArrayLike, *, axis: int = -1, type: int = 2, norm: str | None = None) -> Array: ...
def idst(x: ArrayLike, *, axis: int = -1, type: int = 2, norm: str | None = None) -> Array: ...

# Tier B — Layer 0 xarray
def dct(da: xr.DataArray, *, dim: str, type: int = 2, norm: str | None = None) -> xr.DataArray: ...
# (idct, dst, idst — identical shape)

# Tier C — Operator
class DCT(Operator):
    def __init__(self, variable: str, *, dim: str, type: int = 2): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...

class DST(Operator): ...
```

### `transforms.wavelet` — Wavelet transforms

Three-tier per D11. Tier A wraps `pywt` (optional dep `PyWavelets`).

```python
# Tier A — array
def cwt(x: ArrayLike, *, scales: ArrayLike, wavelet: str = "morl", axis: int = -1) -> Array: ...
def dwt(x: ArrayLike, *, wavelet: str = "db4", level: int | None = None, axis: int = -1) -> dict[str, Array]: ...

# Tier B — Layer 0 xarray
def cwt(da: xr.DataArray, *, scales, wavelet: str = "morl", dim: str) -> xr.DataArray: ...
def dwt(da: xr.DataArray, *, wavelet: str = "db4", level: int | None = None, dim: str) -> dict[str, xr.DataArray]: ...

# Tier C — Operator
class CWT(Operator):
    def __init__(self, variable: str, *, scales, wavelet: str = "morl", dim: str): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...
# `DWT` returns a dict and is kept function-only — wrap with `Lambda(...)` if needed in a Sequential.
```

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

Transform coordinates or values into feature representations. Sub-organized by what they encode. Three-tier per D11 — basis expansions have an array tier (the math is pure arithmetic over coordinate arrays); coordinate-system transforms (`LonLatToCartesian`, `GeocentricToENU`) similarly. Time encoders (`CyclicalTimeEncoding`, `JulianDate`) skip Tier A — the math depends on calendar semantics that live on `xr.DataArray`-of-datetime.

```python
# transforms/encoders/coord_space.py
# Tier A — array
def lonlat_to_cartesian(lon: ArrayLike, lat: ArrayLike, *, radians: bool = False) -> tuple[Array, Array, Array]: ...
def geocentric_to_enu(x: ArrayLike, y: ArrayLike, z: ArrayLike, *, ref_lon: float, ref_lat: float) -> tuple[Array, Array, Array]: ...

# Tier B — Layer 0 xarray
def lonlat_to_cartesian(ds: xr.Dataset, *, lon_var: str = "lon", lat_var: str = "lat") -> xr.Dataset: ...
def geocentric_to_enu(ds: xr.Dataset, *, x_var, y_var, z_var, ref_lon, ref_lat) -> xr.Dataset: ...

# Tier C — Operator
class LonLatToCartesian(Operator):
    def __init__(self, *, lon_var: str = "lon", lat_var: str = "lat"): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...

class GeocentricToENU(Operator):
    def __init__(self, ref_lon: float, ref_lat: float, *, x_var, y_var, z_var): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...
```

```python
# transforms/encoders/coord_time.py — Tier A skipped; calendar-aware
# Tier B
def cyclical_time_encoding(da_time: xr.DataArray, *, components=("dayofyear", "hour")) -> xr.Dataset: ...
def julian_date(da_time: xr.DataArray) -> xr.DataArray: ...

# Tier C
class CyclicalTimeEncoding(Operator):
    def __init__(self, *, components: tuple = ("dayofyear", "hour"), time_dim: str = "time"): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...

class JulianDate(Operator):
    def __init__(self, *, time_dim: str = "time"): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...
```

```python
# transforms/encoders/basis.py — basis / feature expansions
# Tier A — array (numpy default; jax variant useful for downstream JAX models)
def fourier_features(x: ArrayLike, *, num_freqs: int = 10, scale: float = 1.0) -> Array: ...
def random_fourier_features(x: ArrayLike, *, num_features: int = 64, sigma: float = 1.0, seed: int = 0) -> Array: ...
def polynomial_features(x: ArrayLike, *, degree: int = 2) -> Array: ...

# Tier B — Layer 0 xarray
def fourier_features(ds: xr.Dataset, *, coords: list[str], num_freqs: int = 10, scale: float = 1.0) -> xr.Dataset: ...
def random_fourier_features(ds: xr.Dataset, *, coords: list[str], num_features: int = 64, sigma: float = 1.0, seed: int = 0) -> xr.Dataset: ...
def polynomial_features(ds: xr.Dataset, *, coords: list[str], degree: int = 2) -> xr.Dataset: ...

# Tier C — Operator
class FourierFeatures(Operator):
    """NeRF-style positional encoding: [sin(2πσ⁰x), cos(2πσ⁰x), …, sin(2πσ^(L-1)x), cos(2πσ^(L-1)x)]."""
    def __init__(self, coords: list[str], num_freqs: int = 10, scale: float = 1.0): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...

class RandomFourierFeatures(Operator):
    """Random Fourier features (Rahimi & Recht 2007). Approximates RBF kernels for downstream linear models."""
    def __init__(self, coords: list[str], num_features: int = 64, sigma: float = 1.0, seed: int = 0): ...

class PolynomialFeatures(Operator):
    def __init__(self, coords: list[str], degree: int = 2): ...
```

All encoder Operators have the standard `Dataset → Dataset` shape — they add new variables / coords carrying the encoded features. Stateless (no `fit` step required), so they slot into `Sequential` directly.

---

## `extremes` — *Deferred*

Extreme-value statistics (block maxima/minima, peaks over threshold, point process counts/stats) live in the standalone **xtremax** package (master_plan Layer 3). `xr_toolz` does not own the implementation.

If a thin xarray wrapper / Operator surface is needed later, it would be added as `xr_toolz.extremes` (parallel to how `xr_toolz.assimilate` wraps filterX), but no work is planned in v0.x.

Until then: use `xtremax` directly, or hand-author a `Lambda(...)` operator over an xtremax function for a one-off pipeline.

---

## `metrics` — Evaluation Metrics

Pixel-level, spectral, multiscale, and distributional skill scores. **Owned implementation, no `xskillscore` dependency** (see [decisions.md §D7](../decisions.md)).

Three-tier per D11:

- **Tier A — `xr_toolz.metrics.array`** — array kernels in `metrics/array.py` and `metrics/_src/array_<family>.py`. Standard signature: `(prediction: ArrayLike, reference: ArrayLike, *, axis: int | tuple[int, ...], **kwargs) → Array | float`. Numpy default; JAX-friendly variants where the metric is used inside training loops or differentiable losses.
- **Tier B — Layer 0 xarray** — pure functions in `xr_toolz/metrics/_src/<family>.py`. Signature: `(prediction: xr.DataArray, reference: xr.DataArray, *, dim, **kwargs) → xr.DataArray | xr.Dataset | float`. Delegate to Tier A.
- **Tier C — Operator wrappers** in `xr_toolz/metrics/operators.py`. Multi-input: `__call__(prediction: xr.Dataset, reference: xr.Dataset) → xr.DataArray | xr.Dataset | float`. Selects a variable via constructor arg, then delegates to Tier B.

Custom skill score: write a Tier A array kernel (numpy is enough), wrap it in a Tier B function for label preservation, optionally wrap once more with the generic `MetricOp(fn, **config)` Tier C wrapper.

### Tier A — array kernels

```python
# xr_toolz/metrics/_src/array_pixel.py (re-exported via xr_toolz.metrics.array)
def rmse(prediction: ArrayLike, reference: ArrayLike, *, axis: int | tuple[int, ...] = -1) -> Array: ...
def nrmse(prediction: ArrayLike, reference: ArrayLike, *, axis=-1, normalize: str = "std") -> Array: ...
def mae(prediction: ArrayLike, reference: ArrayLike, *, axis=-1) -> Array: ...
def bias(prediction: ArrayLike, reference: ArrayLike, *, axis=-1) -> Array: ...
def correlation(prediction: ArrayLike, reference: ArrayLike, *, axis=-1) -> Array: ...
def murphy_score(prediction: ArrayLike, reference: ArrayLike, *, axis=-1) -> Array: ...
def nash_sutcliffe(prediction: ArrayLike, reference: ArrayLike, *, axis=-1) -> Array: ...
def crps(prediction: ArrayLike, reference: ArrayLike, *, axis=-1) -> Array: ...

# xr_toolz/metrics/_src/array_spectral.py
def psd_score(prediction: ArrayLike, reference: ArrayLike, *, axis, **kwargs) -> Array: ...
def resolved_scale(prediction: ArrayLike, reference: ArrayLike, *, axis, threshold: float = 0.5, **kwargs) -> Array: ...
def coherence_skill(prediction: ArrayLike, reference: ArrayLike, *, axis, **kwargs) -> Array: ...

# xr_toolz/metrics/_src/array_multiscale.py
def per_scale_rmse(prediction: ArrayLike, reference: ArrayLike, *, axis, scales) -> Array: ...
def wavelet_rmse(prediction: ArrayLike, reference: ArrayLike, *, axis, wavelet: str = "db4", level: int = 4) -> Array: ...

# xr_toolz/metrics/_src/array_distributional.py
def ks_statistic(prediction: ArrayLike, reference: ArrayLike, *, axis=-1) -> Array: ...
def wasserstein_1d(prediction: ArrayLike, reference: ArrayLike, *, axis=-1) -> Array: ...
def energy_distance(prediction: ArrayLike, reference: ArrayLike, *, axis=-1) -> Array: ...
```

### Tier B — Layer 0 xarray (DataArray in)

```python
# xr_toolz/metrics/_src/pixel.py
def rmse(prediction: xr.DataArray, reference: xr.DataArray, *, dim) -> xr.DataArray: ...
def nrmse(prediction: xr.DataArray, reference: xr.DataArray, *, dim, normalize: str = "std") -> xr.DataArray: ...
def mae(prediction: xr.DataArray, reference: xr.DataArray, *, dim) -> xr.DataArray: ...
def bias(prediction: xr.DataArray, reference: xr.DataArray, *, dim) -> xr.DataArray: ...
def correlation(prediction: xr.DataArray, reference: xr.DataArray, *, dim) -> xr.DataArray: ...
def murphy_score(prediction: xr.DataArray, reference: xr.DataArray, *, dim) -> xr.DataArray: ...
def nash_sutcliffe(prediction: xr.DataArray, reference: xr.DataArray, *, dim) -> xr.DataArray: ...
def crps(prediction: xr.DataArray, reference: xr.DataArray, *, dim) -> xr.DataArray: ...

# xr_toolz/metrics/_src/spectral.py
def psd_score(prediction: xr.DataArray, reference: xr.DataArray, *, dim, **kwargs) -> xr.Dataset: ...
def resolved_scale(prediction: xr.DataArray, reference: xr.DataArray, *, dim, threshold: float = 0.5, **kwargs) -> xr.DataArray: ...
def coherence_skill(prediction: xr.DataArray, reference: xr.DataArray, *, dim, **kwargs) -> xr.DataArray: ...

# xr_toolz/metrics/_src/multiscale.py
def per_scale_rmse(prediction: xr.DataArray, reference: xr.DataArray, *, dim, scales) -> xr.DataArray: ...
def wavelet_rmse(prediction: xr.DataArray, reference: xr.DataArray, *, dim, wavelet: str = "db4", level: int = 4) -> xr.DataArray: ...

# xr_toolz/metrics/_src/distributional.py
def ks_statistic(prediction: xr.DataArray, reference: xr.DataArray, *, dim) -> xr.DataArray: ...
def wasserstein_1d(prediction: xr.DataArray, reference: xr.DataArray, *, dim) -> xr.DataArray: ...
def energy_distance(prediction: xr.DataArray, reference: xr.DataArray, *, dim) -> xr.DataArray: ...

# xr_toolz/metrics/_src/masked.py
def masked_rmse(prediction: xr.DataArray, reference: xr.DataArray, *, dim, mask: xr.DataArray) -> xr.DataArray: ...
# ... mask-aware variants of the others
```

### Tier C — Operator wrappers

```python
class RMSE(Operator):
    """Root mean squared error. Multi-input operator."""
    def __init__(self, variable: str, dims: list[str]): ...
    def __call__(self, prediction: xr.Dataset, reference: xr.Dataset) -> xr.DataArray: ...

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
    def __call__(self, prediction: xr.Dataset, reference: xr.Dataset) -> xr.Dataset: ...

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
    """Generic wrapper: turns any Tier B metric function into a Tier C Operator."""
    def __init__(self, fn, variable: str, dims: list[str], **kwargs): ...
    def __call__(self, prediction: xr.Dataset, reference: xr.Dataset): ...
```

### Adding a custom skill score

```python
# 1. (optional) Tier A — array kernel
def my_score_array(prediction, reference, *, axis=-1, alpha=1.0):
    xp = array_namespace(prediction, reference)
    return xp.mean((prediction - reference) ** alpha, axis=axis)

# 2. Tier B — Layer 0 xarray
def my_score(prediction: xr.DataArray, reference: xr.DataArray, *, dim, alpha=1.0) -> xr.DataArray:
    return xr.apply_ufunc(
        my_score_array,
        prediction, reference,
        input_core_dims=[[dim], [dim]],
        kwargs={"axis": -1, "alpha": alpha},
    )

# 3. Tier C — wrap once with the generic MetricOp (or hand-author an Operator subclass)
op = MetricOp(my_score, variable="ssh", dims=["time"], alpha=2.0)
op(pred_ds, ref_ds)
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

Three-tier per D11. The array tier is intentionally narrow — most useful viz inputs are already `xr.DataArray` (coords carry the axis labels) — but a few low-level helpers (e.g., array → log-binned spectrum panel) are exposed at Tier A so they can be called from non-xarray code.

```python
# Tier A — array (matplotlib + numpy; no xarray dependency)
def plot_map_array(field: ArrayLike, *, lon: ArrayLike, lat: ArrayLike, ax=None, projection=None, cmap=None, **kwargs) -> matplotlib.axes.Axes: ...
def plot_timeseries_array(values: ArrayLike, *, time: ArrayLike, ax=None, **kwargs) -> matplotlib.axes.Axes: ...
def plot_spectrum_array(power: ArrayLike, *, freqs: ArrayLike, ax=None, log: bool = True, **kwargs) -> matplotlib.axes.Axes: ...

# Tier B — Layer 0 xarray (DataArray in)
def plot_map(da: xr.DataArray, *, ax=None, projection=None, cmap=None, **kwargs) -> matplotlib.axes.Axes: ...
def plot_timeseries(da: xr.DataArray, *, ax=None, **kwargs) -> matplotlib.axes.Axes: ...
def plot_spectrum(da: xr.DataArray, *, ax=None, log: bool = True, **kwargs) -> matplotlib.axes.Axes: ...

# Tier C — Operator (Dataset in, Figure out — D10 exception to Dataset → Dataset)
class PlotMap(Operator):
    def __init__(self, variable: str, *, projection=None, cmap=None, figsize: tuple = (8, 6)): ...
    def __call__(self, ds: xr.Dataset) -> matplotlib.figure.Figure: ...

class PlotTimeseries(Operator): ...
class PlotHovmoller(Operator): ...
class PlotSpectrum(Operator): ...
class PlotResolvedScale(Operator): ...
class QuicklookPanel(Operator):
    """Multi-panel quick diagnostic plot — map + timeseries + spectrum."""
    def __init__(self, variable: str, *, dims=None): ...
    def __call__(self, ds: xr.Dataset) -> matplotlib.figure.Figure: ...
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
xr_toolz/kinematics/
    array.py                  # Tier A re-exports across all domains
    _src/
        ocean.py
        array_ocean.py
        atmosphere.py
        array_atmosphere.py
        ice.py
        array_ice.py
        remote.py
        array_remote.py
```

Each domain ships all three D11 tiers: array kernels (Tier A), Layer 0 xarray functions (Tier B), Operator wrappers (Tier C). The kinematics math is mostly arithmetic on field arrays — finite differences, dot products, ratios — so the array tier is meaningful for every public function.

```python
# Tier A — array (numpy default; jax variants for diff-friendly kernels)
def streamfunction_array(ssh: ArrayLike, *, dx: float, dy: float, f0: float, g: float) -> Array: ...
def geostrophic_velocities_array(ssh: ArrayLike, *, dx: float, dy: float, f0: float, g: float) -> tuple[Array, Array]: ...
def relative_vorticity_array(u: ArrayLike, v: ArrayLike, *, dx: float, dy: float) -> Array: ...
def kinetic_energy_array(u: ArrayLike, v: ArrayLike) -> Array: ...
def okubo_weiss_array(u: ArrayLike, v: ArrayLike, *, dx: float, dy: float) -> Array: ...

def wind_speed_array(u: ArrayLike, v: ArrayLike) -> Array: ...
def potential_temperature_array(temp: ArrayLike, pressure: ArrayLike, *, p0: float = 1e5) -> Array: ...

def normalized_difference_array(a: ArrayLike, b: ArrayLike) -> Array: ...
def radiance_to_reflectance_array(radiance: ArrayLike, solar_zenith: ArrayLike, solar_irradiance: float) -> Array: ...

# Tier B — Layer 0 xarray (multi-variable → Dataset in, DataArray/Dataset out)
# xr_toolz/kinematics/_src/ocean.py
def streamfunction(ds: xr.Dataset, *, variable: str = "ssh", f0: float | None = None, g: float | None = None) -> xr.DataArray: ...
def geostrophic_velocities(ds: xr.Dataset, *, variable: str = "ssh") -> xr.Dataset: ...
def relative_vorticity(ds: xr.Dataset, *, u_var: str = "u", v_var: str = "v") -> xr.DataArray: ...
def kinetic_energy(ds: xr.Dataset, *, u_var: str = "u", v_var: str = "v") -> xr.DataArray: ...
def okubo_weiss(ds: xr.Dataset, *, u_var: str = "u", v_var: str = "v") -> xr.DataArray: ...

# xr_toolz/kinematics/_src/atmosphere.py
def wind_speed(ds: xr.Dataset, *, u_var: str = "u10", v_var: str = "v10") -> xr.DataArray: ...
def potential_temperature(ds: xr.Dataset, *, temp_var: str, pressure_var: str) -> xr.DataArray: ...

# xr_toolz/kinematics/_src/remote.py
def normalized_difference(ds: xr.Dataset, *, var_a: str, var_b: str, name: str = "ndvi") -> xr.DataArray: ...
def radiance_to_reflectance(ds: xr.Dataset, *, solar_zenith_var: str, solar_irradiance: float) -> xr.DataArray: ...

# Tier C — Operator (Dataset in, Dataset out)
class Streamfunction(Operator):
    def __init__(self, *, variable: str = "ssh", f0: float | None = None, g: float | None = None): ...
    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...

class GeostrophicVelocities(Operator): ...
class RelativeVorticity(Operator): ...
class KineticEnergy(Operator): ...
class OkuboWeiss(Operator): ...

class WindSpeed(Operator): ...
class PotentialTemperature(Operator): ...

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
