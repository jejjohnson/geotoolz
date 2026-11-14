# ODC-1.4 — Binned residual maps, region/regime scoring, Diebold–Mariano test

**Source survey item:** [ocean-data-challenges-survey.md §1.4](ocean-data-challenges-survey.md)
**Status:** proposed
**Maps to upstream:** `mod_stat.py` from `2024_DC_SSH_mapping_SWOT_OSE` (and equivalents in the 2023 / 2024c repos).

---

## 1. Motivation

After the segmented PSD score (ODC-1.3) gives a single λ<sub>x</sub> per
reconstruction, the next questions every notebook in the three repos
asks are:

1. **Where, geographically, are the errors?** Bin along-track residuals
   into a regular lat/lon grid → spatial RMSE map.
2. **Are the errors stratified by ocean regime?** Stratify the same
   residuals by *coastal vs open-ocean*, *equatorial vs extra-tropical*,
   *high-vs-low eddy variance* — or by named ocean basins / IPCC
   regions / user-defined polygons.
3. **Is method A *significantly* better than method B?** Diebold–Mariano
   test on the paired loss differential between two competing
   reconstructions.

xrtoolz already exposes the basic pixel metrics (RMSE, bias, MAE,
correlation, R²) but lacks the spatial-binning, region-stratification,
and significance-testing primitives that turn those into the standard
SSH-mapping evaluation outputs. This issue adds them.

The design pivots on **`regionmask`** (already a dependency): instead of
hand-rolling four hard-coded regimes, we expose a generic
`scores_by_region(...)` that consumes any `regionmask.Regions` object or
a categorical DataArray. The "canonical four regimes" become thin
convenience constructors on top.

## 2. User stories

### 2.1 Spatial RMSE map from along-track residuals (primary)

> *I have a colocated SWOT track with `ssha` and `ssh_interp`. I want a
> 1°×1° map of mean residual, std, count, and RMSE per cell.*

```python
import numpy as np
import xarray as xr
from xrtoolz.metrics import bin_residuals_2d

ds_map = bin_residuals_2d(
    ds_track,
    var_ref="ssha", var_pred="ssh_interp",
    lon_bins=np.arange(0, 361, 1),
    lat_bins=np.arange(-80, 81, 1),
    statistics=("mean", "std", "count", "rmse"),
)
# ds_map has dims (lat_bin, lon_bin) with one data_var per stat
```

### 2.2 Scores by ocean basin via regionmask

> *I want RMSE, bias, and explained variance per ocean basin (Atlantic /
> Pacific / Indian / Arctic / Southern) for each method.*

```python
import regionmask
from xrtoolz.metrics import scores_by_region

basins = regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50

ds_scores = scores_by_region(
    ds_track,
    var_ref="ssha", var_pred="ssh_interp",
    regions=basins,
    metrics=("rmse", "bias", "explained_variance"),
)
# ds_scores has dim (region,) with one data_var per metric
```

### 2.3 Canonical regime stratification

> *I want the upstream "coastal / equatorial / open-ocean high-eddy /
> open-ocean low-eddy" partition.*

```python
from xrtoolz.geo.regimes import coastal_regions, equatorial_regions, eddy_regions

coastal  = coastal_regions(distance_km=200)         # Regions
equator  = equatorial_regions(lat_threshold=5.0)    # Regions
eddy     = eddy_regions(ds_ref_grid, var="ssh")     # categorical DataArray

ds_scores = scores_by_region(ds_track,
                             var_ref="ssha", var_pred="ssh_interp",
                             regions=coastal)
```

### 2.4 Method-A vs method-B significance test

> *I have squared losses from two competing reconstructions on the same
> along-track points. Is one significantly better?*

```python
from xrtoolz.metrics import dm_test

loss_a = (ds_a["ssh_interp"] - ds_a["ssha"]) ** 2
loss_b = (ds_b["ssh_interp"] - ds_b["ssha"]) ** 2

dm_stat, pvalue = dm_test(loss_a, loss_b, h=1, alternative="two-sided")
```

### 2.5 As Layer-1 Operators inside a Sequential

```python
from xrtoolz.metrics import BinnedResiduals2D, RegionScores

pipeline = Sequential([
    SampleAtPoints(...),                            # ODC-1.2
    BinnedResiduals2D(var_ref="ssha", var_pred="ssh_interp",
                      lon_bins=..., lat_bins=...),
])
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| Pixel metrics (RMSE/bias/MAE/correlation/R²) | [`metrics/_src/pixel.py`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/metrics/_src/pixel.py) | reuse |
| Region polygons + mask machinery | `regionmask` (already a dep) | leverage |
| `geo/_src/masks.py` | exists | reuse / extend |
| 2D binning of point residuals | — | **add** `bin_residuals_2d` |
| Generic per-region scoring | — | **add** `scores_by_region` |
| Coastal / equatorial / eddy regimes | — | **add** convenience constructors |
| Diebold–Mariano test | — | **add** `dm_test` |
| Operator wrappers | — | **add** `BinnedResiduals2D`, `RegionScores` |

## 4. Design

### 4.1 Layer 0 — 2D residual binning

```python
# src/xrtoolz/metrics/_src/binned.py
def bin_residuals_2d(
    ds_track: xr.Dataset, *,
    var_ref: str, var_pred: str,
    lon: str = "longitude", lat: str = "latitude",
    lon_bins: ArrayLike, lat_bins: ArrayLike,
    statistics: Sequence[str] = ("mean", "std", "count", "rmse"),
) -> xr.Dataset:
    """2D lat/lon binning of along-track residuals."""
```

Single `scipy.stats.binned_statistic_2d` call per statistic. RMSE is
implemented as `sqrt(mean((pred - ref)**2))` via `statistic='mean'` on
the squared residual. ~30 LOC.

### 4.2 Generic per-region scoring (regionmask-centric)

```python
def scores_by_region(
    ds_track: xr.Dataset, *,
    var_ref: str, var_pred: str,
    regions: regionmask.Regions | xr.DataArray,
    lon: str = "longitude", lat: str = "latitude",
    metrics: Sequence[str] = ("rmse", "bias", "correlation",
                              "explained_variance"),
    region_dim: str = "region",
) -> xr.Dataset:
    """Pixel metrics stratified by region.

    `regions` is either a regionmask.Regions object (rasterised onto the
    track points via .mask()) or a DataArray of categorical labels
    aligned to the track's main dim.
    """
```

Internals:

1. If `regions` is a `regionmask.Regions`: call `regions.mask(lon, lat)`
   on the track coords to get a categorical DataArray of region indices.
2. `groupby` the residual + reference + prediction by the categorical
   DataArray.
3. Apply each metric as a Layer 0 reduction within the groupby.

This single function replaces the four hard-coded regime branches in
the upstream. The existing pixel-metric kernels in
[`metrics/_src/pixel.py`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/metrics/_src/pixel.py) are
reused inside the groupby.

### 4.3 Canonical regime constructors

```python
# src/xrtoolz/geo/_src/regimes.py
def coastal_regions(
    *, distance_km: float = 200.0,
    resolution: str = "110",       # land_110, land_50, land_10
) -> regionmask.Regions:
    """Two-region partition: coastal (within distance_km of land) vs open ocean.

    Internally rasterises ``regionmask.defined_regions.natural_earth_v5_0_0
    .land_<resolution>`` onto a global lat/lon grid, applies
    ``scipy.ndimage.distance_transform_edt``, thresholds at
    ``distance_km``, and converts the resulting binary masks back to
    polygons via the inverse rasterisation.
    """

def equatorial_regions(
    *, lat_threshold: float = 5.0,
) -> regionmask.Regions:
    """Two-region partition: equatorial (|lat| < threshold) vs extra-tropical."""

def eddy_regions(
    ds: xr.Dataset, *,
    var: str,
    threshold: float | None = None,    # None → median of local variance
    window: tuple[int, int] = (5, 5),
    lon: str = "longitude", lat: str = "latitude",
) -> xr.DataArray:
    """Two-region categorical mask: high-vs-low local-variance regime.

    Returns a DataArray of {0, 1} aligned to ``ds`` (data-driven, not a
    Regions object — variance regimes depend on the field, not on
    geography).
    """
```

Three constructors, three different return shapes intentional:

- `coastal_regions` / `equatorial_regions` → `regionmask.Regions`
  (geographic, reusable across datasets).
- `eddy_regions` → `xr.DataArray` (data-driven; depends on the SSH
  field, can't be a static `Regions` object).

`scores_by_region` accepts either, dispatching internally.

### 4.4 Diebold–Mariano test

```python
# src/xrtoolz/metrics/_src/dm_test.py
def dm_test(
    loss_a: ArrayLike, loss_b: ArrayLike, *,
    h: int = 1,                            # forecast horizon → HAC lag
    alternative: str = "two-sided",        # | "less" | "greater"
    power: float = 2.0,                    # 2 → squared loss; 1 → absolute
    hln_correction: bool = True,           # Harvey-Leybourne-Newbold small-sample
) -> tuple[float, float]:
    """Diebold–Mariano test for equal predictive accuracy.

    Newey-West HAC variance estimator on the loss differential
    `d_t = L(e_a,t) - L(e_b,t)`, then `DM = mean(d) / sqrt(var_HAC(d) / N)`.
    Optionally apply Harvey–Leybourne–Newbold small-sample t-correction.
    """
```

~30 LOC, no `statsmodels` dep. Returns scalar (DM stat, p-value).
Optionally validated against `statsmodels` in a marked test (skip if
unavailable).

### 4.5 Layer-1 Operators

```python
# src/xrtoolz/metrics/operators.py
class BinnedResiduals2D(Operator):
    def __init__(self, *, var_ref, var_pred, lon_bins, lat_bins,
                 lon="longitude", lat="latitude",
                 statistics=("mean", "std", "count", "rmse")): ...

class RegionScores(Operator):
    def __init__(self, *, var_ref, var_pred, regions,
                 lon="longitude", lat="latitude",
                 metrics=("rmse", "bias", "correlation",
                          "explained_variance"),
                 region_dim="region"): ...
```

Standard pattern. **No `DMTest` Operator** — it returns scalars, doesn't
fit the `Dataset → Dataset` pipeline shape.

## 5. Library leverage

| Need | Library |
|---|---|
| 2D lat/lon binning (mean/std/count/median) | `scipy.stats.binned_statistic_2d` |
| Region polygons + mask machinery | `regionmask` (already a dep) |
| Pre-built regions (land, ocean basins, IPCC) | `regionmask.defined_regions.*` |
| Distance-to-land | `scipy.ndimage.distance_transform_edt` (on rasterised landmask) |
| Per-region groupby | `xarray.Dataset.groupby` |
| HAC variance / Newey-West | hand-implemented (~15 LOC) — no `statsmodels` |
| Existing pixel metrics | [`metrics/_src/pixel.py`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/metrics/_src/pixel.py) |

No new top-level dependencies.

## 6. Public API surface

```python
# Layer 0 primitives
xrtoolz.metrics.bin_residuals_2d(ds_track, *, var_ref, var_pred,
                                  lon, lat, lon_bins, lat_bins, statistics)
xrtoolz.metrics.scores_by_region(ds_track, *, var_ref, var_pred,
                                  regions, lon, lat, metrics, region_dim)
xrtoolz.metrics.dm_test(loss_a, loss_b, *, h, alternative, power, hln_correction)

# Regime constructors
xrtoolz.geo.regimes.coastal_regions(*, distance_km, resolution)
xrtoolz.geo.regimes.equatorial_regions(*, lat_threshold)
xrtoolz.geo.regimes.eddy_regions(ds, *, var, threshold, window, lon, lat)

# Operators
xrtoolz.metrics.BinnedResiduals2D(...)
xrtoolz.metrics.RegionScores(...)
```

## 7. Tests

| Test | Asserts |
|---|---|
| `bin_residuals_2d` mean/std/count match per-bin numpy reduction | exact |
| `bin_residuals_2d` empty bins | NaN / count=0 |
| `bin_residuals_2d` RMSE | `sqrt(mean(err²))` matches direct calc |
| `scores_by_region` with a 2-region DataArray | per-region RMSE matches manual subset |
| `scores_by_region` with `regionmask.Regions` | rasterisation + groupby produces correct partition |
| `coastal_regions` rasterised | a coastal point lies in coastal region; offshore point in open-ocean |
| `equatorial_regions` | exact `|lat| < threshold` partition |
| `eddy_regions` | thresholding on local rolling variance produces a 2-class DataArray |
| `dm_test` identical losses | DM stat ≈ 0, p ≈ 1 |
| `dm_test` skewed losses | DM stat sign matches expectation |
| `dm_test` HLN small-sample correction | t-distribution p-value differs from z-distribution as expected |
| `dm_test` vs statsmodels (marked, skip if absent) | match within 1e-6 |
| Operator round-trips via `get_config` | reconstructed operators produce identical output |

Target: ~13 cases.

## 8. Out of scope

- **Bootstrap CIs for RMSE** — `xskillscore` does this; separate item.
- **Custom regime definitions beyond the canonical three** — users
  build a `regionmask.Regions` and pass it to `scores_by_region`.
- **Multivariate / multi-method DM test** — single-pair only.
- **Plotting** (regime bar plots, score-map panels) — that's ODC-1.5.
- **`statsmodels` dependency** — DM test is implemented inline.

## 9. Effort

≈120 LOC implementation + ≈110 LOC tests. Single PR.

| Slice | LOC |
|---|---|
| `bin_residuals_2d` | 30 |
| `scores_by_region` (incl. regionmask dispatch) | 35 |
| `coastal_regions` / `equatorial_regions` / `eddy_regions` | 35 |
| `dm_test` (HAC + HLN correction) | 30 |
| `BinnedResiduals2D`, `RegionScores` operators | 25 |
| Tests | ~110 |
| Docs / re-exports | 10 |

## 10. Risks / open questions

1. **`coastal_regions` performance.** Rasterising land at 0.1° + EDT
   + threshold + inverse-rasterise to polygons is expensive on first
   call. Cache the resulting `Regions` object (lru_cache on
   `(distance_km, resolution)` keys).
2. **`coastal_regions` fidelity.** EDT distances are pixel-space; we
   convert via the grid's mean Δkm. Document accuracy as
   "±half a grid cell" — adequate for 200-km coastal definition,
   not for sub-10-km fidelity.
3. **`scores_by_region` with `regionmask.Regions`.** `regions.mask(lon,
   lat)` works on coordinate arrays, returns NaN for points outside any
   region; we drop those before the groupby. Document the drop.
4. **`eddy_regions` returning a DataArray instead of `Regions`.** The
   API split is intentional but mildly surprising. Clear docstring +
   examples.
5. **DM test against `statsmodels` reference.** Validate in a marked
   test (`importorskip`) — gives confidence without adding the dep.
6. **Where things live:**
   - `metrics/_src/binned.py` (new) → `bin_residuals_2d`,
     `scores_by_region`.
   - `metrics/_src/dm_test.py` (new) → `dm_test`.
   - `geo/_src/regimes.py` (new) → three regime constructors.
   - `metrics/operators.py` → `BinnedResiduals2D`, `RegionScores`.
