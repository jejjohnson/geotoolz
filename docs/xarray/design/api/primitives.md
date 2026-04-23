---
status: draft
version: 0.1.0
---

# Primitives — Layer 0 Pure Functions

## `core` — Base Infrastructure

```python
def compose(*operators: Callable) -> Callable:
    """Compose multiple callables left-to-right."""

def identity(ds: xr.Dataset) -> xr.Dataset:
    """Pass-through operator. Useful as a default/placeholder."""
```

---

## `validation` — Data Harmonization

```python
def validate_longitude(ds, target_range="-180_to_180") -> xr.Dataset
def validate_latitude(ds) -> xr.Dataset
def validate_time(ds, calendar="standard") -> xr.Dataset
def rename_coords(ds, mapping: dict) -> xr.Dataset
def sort_coords(ds, dims: list[str] | None = None) -> xr.Dataset
def set_coord_attrs(ds, coord, units, standard_name, long_name) -> xr.Dataset
```

---

## `crs` — Coordinate Reference Systems

```python
def assign_crs(ds, crs="EPSG:4326") -> xr.Dataset
def reproject(ds, target_crs, resolution=None) -> xr.Dataset
def get_crs(ds) -> pyproj.CRS | None
```

---

## `subset` — Spatial and Temporal Selection

```python
def subset_bbox(ds, lon_bnds, lat_bnds) -> xr.Dataset
def subset_time(ds, time_min, time_max) -> xr.Dataset
def subset_where(ds, mask: xr.DataArray, drop=False) -> xr.Dataset
def subset_variables(ds, variables: list[str]) -> xr.Dataset
```

---

## `masks` — Spatial Masks

```python
def add_land_mask(ds) -> xr.Dataset
def add_ocean_mask(ds, ocean="global") -> xr.Dataset
def add_country_mask(ds, country="spain") -> xr.Dataset
def add_region_mask(ds, region_name, regions=None) -> xr.Dataset
```

---

## `regrid` — Grid Transformations

```python
def regrid_linear(ds, target_lon, target_lat) -> xr.Dataset
def regrid_nearest(ds, target_lon, target_lat) -> xr.Dataset
def regrid_cubic(ds, target_lon, target_lat) -> xr.Dataset
def coarsen(ds, factor: dict, method="mean") -> xr.Dataset
def refine(ds, factor: dict, method="linear") -> xr.Dataset
```

Implementation strategy: extract lat/lon/values as numpy arrays, use `scipy.interpolate.RegularGridInterpolator` for rectilinear grids and `scipy.interpolate.griddata` for irregular grids. For nearest-neighbor and radius-based queries, use `sklearn.neighbors.BallTree` or `KDTree` which handle haversine distance natively.

---

## `interpolation` — Gap Filling and Resampling

```python
def fillnan_spatial(ds, method="linear", max_gap=None) -> xr.Dataset
def fillnan_temporal(ds, method="linear", max_gap=None) -> xr.Dataset
def fillnan_rbf(ds, kernel="thin_plate_spline", neighbors=None) -> xr.Dataset
def resample_time(ds, freq="1D", method="mean") -> xr.Dataset
```

Uses `scipy.interpolate.griddata` for spatial, `scipy.interpolate.RBFInterpolator` for RBF-based filling, and xarray's built-in `.resample()` for temporal.

---

## `detrend` — Climatology, Anomalies, and Filtering

```python
def calculate_climatology(ds, freq="day") -> xr.Dataset
def calculate_climatology_smoothed(ds, freq="day", window=60) -> xr.Dataset
def remove_climatology(ds, climatology) -> xr.Dataset
def add_climatology(ds, climatology) -> xr.Dataset
def calculate_anomalies(ds, freq="day", smoothing=None) -> xr.Dataset
def lowpass_filter(ds, dim="time", cutoff=30, order=3) -> xr.Dataset
```

---

## `encoders` — Coordinate Encodings

```python
# Spatial coordinate transforms
def lonlat_to_cartesian(ds) -> xr.Dataset
def cartesian_to_lonlat(ds) -> xr.Dataset
def lon_360_to_180(coord: np.ndarray) -> np.ndarray
def lon_180_to_360(coord: np.ndarray) -> np.ndarray

# Cyclical / positional encodings
def cyclical_encode(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]
def fourier_features(values: np.ndarray, num_freqs: int, scale: float = 1.0) -> np.ndarray
def random_fourier_features(values: np.ndarray, num_features: int, sigma: float = 1.0) -> np.ndarray
def positional_encoding(values: np.ndarray, num_freqs: int, include_input: bool = True) -> np.ndarray

# Temporal encodings
def encode_time_cyclical(ds, components=("dayofyear", "hour")) -> xr.Dataset
def encode_time_ordinal(ds, reference_date=None) -> xr.Dataset
```

---

## `discretize` — Binning and Gridding

```python
def bin_2d(da, target_lon, target_lat, statistic="mean") -> xr.DataArray
def bin_2d_time(da, target_lon, target_lat, target_time, statistic="mean") -> xr.DataArray
def histogram_2d(da, bins_x, bins_y) -> xr.DataArray
def points_to_grid(lons, lats, values, target_lon, target_lat, method="nearest") -> xr.DataArray
```

Implementation replaces `pyinterp.Binning2D` with `scipy.stats.binned_statistic_2d` for the core binning and `sklearn.neighbors.BallTree` for nearest-neighbor assignment. Grid definition uses dataclasses instead of `odc-geo`:

```python
@dataclass
class Grid:
    lon: np.ndarray
    lat: np.ndarray

    @classmethod
    def from_bounds(cls, lon_bnds, lat_bnds, resolution): ...

    @classmethod
    def from_dataset(cls, ds): ...

@dataclass
class SpaceTimeGrid(Grid):
    time: np.ndarray | pd.DatetimeIndex

    @classmethod
    def from_bounds(cls, lon_bnds, lat_bnds, resolution, time_min, time_max, freq): ...
```

---

## `extremes` — Extreme Value Analysis

```python
def block_maxima(da, block_size=365, side="center") -> xr.DataArray
def block_minima(da, block_size=365, side="center") -> xr.DataArray
def pot_threshold(da, quantile=0.98) -> float
def pot_exceedances(da, quantile=0.98, decluster_freq=None) -> xr.DataArray
def pp_counts(da, threshold, block_size=365) -> xr.DataArray
def pp_stats(da, threshold, block_size=365, statistic="mean") -> xr.DataArray
```

---

## `spectral` — Spectral Analysis

```python
def psd(ds, variable, dims, **kwargs) -> xr.Dataset
def psd_isotropic(ds, variable, dims, **kwargs) -> xr.Dataset
def cross_spectrum(ds, var_a, var_b, dims) -> xr.Dataset
def coherence(ds, var_a, var_b, dims) -> xr.Dataset
```

---

## `metrics` — Evaluation Metrics

```python
# Pixel metrics
def rmse(ds_pred, ds_ref, variable, dims) -> xr.DataArray
def nrmse(ds_pred, ds_ref, variable, dims) -> xr.DataArray
def mae(ds_pred, ds_ref, variable, dims) -> xr.DataArray
def bias(ds_pred, ds_ref, variable, dims) -> xr.DataArray
def correlation(ds_pred, ds_ref, variable, dims) -> xr.DataArray
def mse(ds_pred, ds_ref, variable, dims) -> xr.DataArray
def r2_score(ds_pred, ds_ref, variable, dims) -> xr.DataArray

# Spectral metrics
def psd_score(ds_pred, ds_ref, variable, dims) -> xr.Dataset
def resolved_scale(ds_pred, ds_ref, variable, dims, threshold=0.5) -> float

# Multiscale metrics
def spatial_rmse_at_scale(ds_pred, ds_ref, variable, scales) -> xr.Dataset
```

Wraps `xskillscore` where available, with pure numpy/scipy fallbacks.

---

## `kinematics` — Physical Quantities

Domain-specific physical transformations. Starting with remote sensing, methane, and oceanography. Uses `metpy` where available, with pure numpy/scipy fallbacks.

```python
# Oceanography
def coriolis_parameter(lat) -> xr.DataArray
def streamfunction(ds, variable="ssh", f0=None, g=None) -> xr.Dataset
def geostrophic_velocities(ds, variable="ssh") -> xr.Dataset
def relative_vorticity(ds, u_var="u", v_var="v") -> xr.Dataset
def kinetic_energy(ds, u_var="u", v_var="v") -> xr.Dataset
def enstrophy(ds, vorticity_var="vorticity") -> xr.Dataset
def strain_rate(ds, u_var="u", v_var="v") -> xr.Dataset
def okubo_weiss(ds, u_var="u", v_var="v") -> xr.Dataset
def mixed_layer_depth(ds, temp_var="temperature", depth_var="depth", threshold=0.2) -> xr.Dataset
def brunt_vaisala(ds, temp_var="temperature", salt_var="salinity", depth_var="depth") -> xr.Dataset

# Remote sensing / radiometry
def radiance_to_reflectance(ds, solar_zenith_var, solar_irradiance) -> xr.Dataset
def brightness_temperature(ds, variable, wavelength) -> xr.Dataset
def normalized_difference(ds, var_a, var_b, name="ndvi") -> xr.Dataset

# Methane-specific
def column_averaging_kernel(ds, pressure_var, kernel_var) -> xr.Dataset
def dry_air_column(ds, surface_pressure_var, humidity_var) -> xr.Dataset
def mixing_ratio_to_column(ds, variable, pressure_levels) -> xr.Dataset

# Atmospheric
def potential_temperature(ds, temp_var, pressure_var) -> xr.Dataset
def wind_speed(ds, u_var="u10", v_var="v10") -> xr.Dataset
def wind_direction(ds, u_var="u10", v_var="v10") -> xr.Dataset
```

---

## `sklearn` — scikit-learn Interop

```python
def to_2d(da, sample_dim) -> tuple[np.ndarray, dict]
def from_2d(arr, meta, new_feature_dim="component") -> xr.DataArray
```

When `xarray_sklearn` is installed, the `to_2d` / `from_2d` helpers are still available for low-level use, but the recommended path is `XarrayEstimator` (via `SklearnOp` at Layer 1) which handles NaN policies, shape-changing transforms, and metadata preservation automatically.

For large-domain workflows, combine with `xrpatcher` to apply these primitives per spatial patch rather than over the full grid — see [../examples/integration.md](../examples/integration.md).
