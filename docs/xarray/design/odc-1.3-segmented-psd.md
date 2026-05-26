# ODC-1.3 — Segmented along-track PSD score + effective resolution λx

**Source survey item:** [ocean-data-challenges-survey.md §1.3](ocean-data-challenges-survey.md)
**Status:** proposed
**Maps to upstream:** `mod_spectral.py` from `2024_DC_SSH_mapping_SWOT_OSE` (and equivalents in the 2023 / 2024c repos).

---

## 1. Motivation

The canonical SSH-mapping skill metric is the wavelength λ<sub>x</sub> at
which the **PSD score** `1 − PSD(err) / PSD(ref)` first crosses 0.5 — the
shortest wavelength the reconstruction "resolves". Every notebook in the
three ocean-data-challenges repos computes this, and every leaderboard
ranks reconstructions by it.

The metric is computed on along-track satellite altimetry, not on the
gridded reconstruction. Along-track time series have:

- **Gaps** (no obs over land, between satellite passes).
- **Approximately uniform spacing within a contiguous chunk** (~7 km
  for SWOT, ~13 km for nadir altimeters), so each contiguous chunk can
  be FFT'd with a fixed Δx.
- **Highly variable lengths** between gaps — from a handful of samples
  to tens of thousands.

The upstream pipeline is: gap-split → fixed-length sliding windows
within each chunk → per-window scipy Welch PSD → average → λx.
xrtoolz already exposes a gridded `psd_score` (xrft-backed) and a
`resolved_scale` λ-extraction helper, but lacks the **gap-tolerant
1-D segmented Welch PSD** that bridges them.

This issue adds that bridge. After ODC-1.1 (bandpass) and ODC-1.2
(colocation) populate a single along-track Dataset with `ssha` and
`ssh_interp`, ODC-1.3 reduces it to `psd_score(wavenumber)` and emits
λ<sub>x</sub>.

## 2. User stories

### 2.1 Compute λx on a colocated SWOT track (primary)

> *I have a SWOT along-track Dataset with `ssha` (reference) and
> `ssh_interp` (my reconstruction sampled onto the track). I want the
> PSD score curve and the effective resolution λx in km.*

```python
import xarray as xr
from xrtoolz.metrics import along_track_psd_score, resolved_scale

ds_track = xr.open_dataset("swot_colocated.nc")        # has ssha + ssh_interp
ds_psd = along_track_psd_score(
    ds_track,
    var_ref="ssha", var_pred="ssh_interp",
    dim="num_lines",
    npt=128,
    overlap=0.5,
    max_gap=np.timedelta64(2, "s"),
)
# ds_psd has dims (segment, wavenumber); coords wavenumber [1/km], wavelength [km]
score_avg = ds_psd["psd_score"].mean("segment")
lambda_x = resolved_scale(score_avg, frequency="wavenumber", level=0.5)
```

### 2.2 Geographic-box averaging for global maps

> *I want a global map of λx, with each cell aggregating segments that
> fall in a 10°×10° box around it.*

```python
from xrtoolz.metrics import psd_score_by_region

ds_regional = psd_score_by_region(
    ds_psd,
    lat_centers=np.arange(-80, 91, 1),
    lon_centers=np.arange(0, 360, 1),
    delta_lat=10.0, delta_lon=10.0,
    min_segments=2,
)
# ds_regional["psd_score"] has dims (lat, lon, wavenumber)
```

### 2.3 As a Layer-1 Operator inside a Sequential

```python
from xrtoolz.metrics import SegmentedPSDScore
from xrtoolz.core import Sequential

pipeline = Sequential([
    BandpassWavelength(...),                      # ODC-1.1
    SampleAtPoints(points=ds_track, ...),         # ODC-1.2
    SegmentedPSDScore(
        var_ref="ssha", var_pred="ssh_interp",
        dim="num_lines", npt=128, overlap=0.5,
    ),
])
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| Gridded `psd_score`, `psd_error` | [`metrics/_src/spectral.py:58-128`](../../src/xrtoolz/metrics/_src/spectral.py) | unchanged |
| `resolved_scale` (λx-at-level) | [`spectral.py:131-204`](../../src/xrtoolz/metrics/_src/spectral.py) | **reuse as-is** |
| Gridded `PSDScore` Operator | [`spectral.py:210`](../../src/xrtoolz/metrics/_src/spectral.py) | unchanged |
| Gap-tolerant 1-D segmenter | — | **add** `segment_signal` |
| Per-segment Welch / CSD / coherence | — | **add** thin scipy wrappers |
| Along-track PSD score driver | — | **add** `along_track_psd_score` |
| Geographic box aggregation | — | **add** `psd_score_by_region` |
| `Operator` wrapper | — | **add** `SegmentedPSDScore` |
| Median Δx (km) | proposed in ODC-1.1 | reuse |

## 4. Design

### 4.1 The single missing primitive

`scipy.signal.welch` segments internally but cannot skip NaNs / time
gaps. So we pre-segment, then call scipy per chunk and average.

```python
def segment_signal(
    x: ArrayLike, *,
    npt: int,
    overlap: float = 0.5,
    gap_indices: ArrayLike | None = None,
    min_segment_length: int | None = None,
) -> NDArray:                            # shape (n_segments, npt)
    """Slice a 1-D signal into equal-length, gap-free, overlapping windows.

    `gap_indices` are the positions where contiguous chunks end (e.g.
    indices `i` where `time[i+1] - time[i] > max_gap`). Within each
    contiguous chunk, slide a window of size `npt` with stride
    `int(npt * (1 - overlap))`. Chunks shorter than `min_segment_length`
    (default `npt`) are dropped.
    """
```

Implementation: ~25 LOC using `numpy.lib.stride_tricks.sliding_window_view`.

### 4.2 Private kernels (numpy/scipy)

```python
# src/xrtoolz/metrics/_src/_segmented_psd_kernels.py
def segmented_psd(
    x: ArrayLike, *,
    fs: float, npt: int, overlap: float = 0.5,
    gap_indices: ArrayLike | None = None,
    window: str = "hann", scaling: str = "density",
) -> tuple[NDArray, NDArray]:
    """Mean PSD over gap-free windows. Wraps scipy.signal.welch per segment."""

def segmented_csd(x, y, *, fs, npt, overlap, gap_indices, window, scaling):
    """Mean cross-spectral density. Wraps scipy.signal.csd."""

def segmented_coherence(x, y, *, fs, npt, overlap, gap_indices, window):
    """Mean magnitude-squared coherence. Wraps scipy.signal.coherence."""
```

Each kernel:
1. `segs_x = segment_signal(x, ...)` (and `segs_y` where applicable).
2. Loop over segments, call the corresponding scipy.signal routine with
   `nperseg=npt, noverlap=0` (each scipy call processes a single
   periodogram).
3. Stack and mean across segments.

### 4.3 Layer 0 — xarray driver

```python
# src/xrtoolz/metrics/_src/segmented_psd.py
def along_track_psd_score(
    ds_track: xr.Dataset, *,
    var_ref: str,
    var_pred: str,
    dim: str = "num_lines",
    npt: int = 128,
    overlap: float = 0.5,
    max_gap: np.timedelta64 | float = np.timedelta64(2, "s"),
    spacing_km: float | None = None,
    lon: str = "longitude",
    lat: str = "latitude",
    time: str = "time",
) -> xr.Dataset
```

Returns Dataset with:
- **dims**: `(segment, wavenumber)`
- **data_vars**: `psd_ref`, `psd_pred`, `psd_err`, `psd_score`, `coherence`
- **coords**: `segment_lon`, `segment_lat` (per-segment median, with
  prime-meridian wrap-around handled via `scipy.stats.circmean`),
  `wavenumber` [1/km], `wavelength` [km]

Internal flow:
1. Compute `gap_indices` from `np.diff(ds_track[time]) > max_gap`.
2. If `spacing_km is None`, use ODC-1.1 `median_dx_km(lon, lat)`.
3. Call `segment_signal` once on the index array to derive segment
   start/stop indices; reuse for all variables.
4. Per-segment: call `scipy.signal.welch` for `psd_ref`, `psd_pred`,
   `psd_err = psd of (pred − ref)`; `scipy.signal.coherence` for
   coherence. `psd_score = 1 − psd_err / psd_ref`.
5. Per-segment representative longitude via `scipy.stats.circmean(lon,
   high=360, low=0)` (or equivalently `atan2(mean(sin(lon_rad)),
   mean(cos(lon_rad)))`) — wraps the dateline cleanly without
   Greenwich/dateline branching.

### 4.4 Geographic box aggregation

```python
def psd_score_by_region(
    ds_segments: xr.Dataset, *,
    lat_centers: ArrayLike = np.arange(-80, 91, 1),
    lon_centers: ArrayLike = np.arange(0, 360, 1),
    delta_lat: float = 10.0,
    delta_lon: float = 10.0,
    min_segments: int = 2,
) -> xr.Dataset:
    """Average per-segment PSDs into overlapping (lat, lon) bins.

    For each (lat_c, lon_c), select segments with |lat - lat_c| <= delta_lat/2
    and equivalent longitudinal selection with wrap-around handling.
    Cells with fewer than min_segments segments → NaN.
    """
```

Returns `(lat, lon, wavenumber)`-shaped Dataset of `psd_ref`, `psd_pred`,
`psd_err`, `psd_score`, `coherence`, `n_segments`.

### 4.5 Layer-1 Operator

```python
# src/xrtoolz/metrics/operators.py
class SegmentedPSDScore(Operator):
    """Single-input segmented along-track PSD-score operator.

    Assumes ref and pred live in the same Dataset (after ODC-1.2
    colocation). For two-input use, call along_track_psd_score directly.
    """

    def __init__(self, *,
                 var_ref: str, var_pred: str,
                 dim: str = "num_lines",
                 npt: int, overlap: float = 0.5,
                 max_gap=np.timedelta64(2, "s"),
                 spacing_km=None,
                 lon="longitude", lat="latitude", time="time"): ...

    def __call__(self, ds): return along_track_psd_score(ds, **self._kw)
    def get_config(self): ...
    def __repr__(self): ...
```

### 4.6 λx via existing `resolved_scale`

No new helper. Users compose:

```python
from xrtoolz.metrics import resolved_scale
lambda_x = resolved_scale(ds_psd["psd_score"].mean("segment"),
                          frequency="wavenumber", level=0.5)
```

## 5. Library leverage

| Need | Library |
|---|---|
| Per-segment Welch PSD | `scipy.signal.welch(x, fs, nperseg=npt, noverlap=0, window=...)` |
| Cross-spectral density | `scipy.signal.csd` |
| Magnitude-squared coherence | `scipy.signal.coherence` |
| Sliding-window view | `numpy.lib.stride_tricks.sliding_window_view` |
| Circular mean (per-segment lon) | `scipy.stats.circmean` |
| Median Δx (km) | ODC-1.1 `median_dx_km` |
| λx extraction | existing `xrtoolz.metrics.resolved_scale` |
| Geographic binning | `xarray.Dataset.groupby_bins` |

No new dependencies. The segmenter is the only ~25 LOC primitive that
isn't already in scipy/numpy/xarray.

## 6. Public API surface

```python
# Private kernels (numpy/scipy)
xrtoolz.metrics._src._segmented_psd_kernels.segment_signal(x, *, npt, overlap, gap_indices, min_segment_length)
xrtoolz.metrics._src._segmented_psd_kernels.segmented_psd(x, *, fs, npt, overlap, gap_indices, window, scaling)
xrtoolz.metrics._src._segmented_psd_kernels.segmented_csd(x, y, *, fs, npt, overlap, gap_indices, window, scaling)
xrtoolz.metrics._src._segmented_psd_kernels.segmented_coherence(x, y, *, fs, npt, overlap, gap_indices, window)

# Layer 0 — xarray
xrtoolz.metrics.along_track_psd_score(ds_track, *, var_ref, var_pred, dim,
                                       npt, overlap, max_gap, spacing_km,
                                       lon, lat, time)
xrtoolz.metrics.psd_score_by_region(ds_segments, *, lat_centers, lon_centers,
                                     delta_lat, delta_lon, min_segments)

# Operator
xrtoolz.metrics.SegmentedPSDScore(...)

# Reused (already public)
xrtoolz.metrics.resolved_scale(score, frequency, level=0.5)
```

## 7. Tests

| Test | Asserts |
|---|---|
| `segment_signal` no gaps | `(N - npt) // stride + 1` segments, each `npt` long |
| `segment_signal` with gap_indices | Sub-chunks shorter than `min_segment_length` dropped |
| `segmented_psd` on a pure tone | Peak at expected wavenumber within bin width |
| `segmented_psd` Parseval | `Σ psd · df ≈ var(x)` within tol |
| `segmented_coherence` identical signals | ≈ 1.0 across all freqs |
| `segmented_coherence` independent noise | ≈ 1/N_segments at all freqs |
| `along_track_psd_score` end-to-end | `psd_score ∈ (0, 1)` for noisy pred ≈ ref |
| Gap detection from time coord | `time` jumps > `max_gap` create new segment boundaries |
| `resolved_scale` on a known psd_score | Returns exact crossing wavelength |
| `psd_score_by_region` | Geographic averaging respects `min_segments` threshold |
| `SegmentedPSDScore` round-trip | `get_config` → reconstructed op produces identical output |
| Per-segment lon wrap-around | Segment crossing 0° has lon ≈ 0, not ≈ 180 |

Target: ~12 cases.

## 8. Out of scope

- **2D `(kx, kt)` PSD score with `(λx, λt)` double-contour** — different
  metric; this is gridded reconstruction vs gridded reference. Will be
  proposed in ODC-2.1.
- **Re-implementing `resolved_scale` / `find_intercept_1D`** — already
  exists, reuse.
- **Lomb-Scargle / non-uniform-spacing PSD** — community standard
  assumes uniform within-segment spacing; bandpass/colocation
  pre-conditions ensure this.
- **`pyinterp` backend** — pure scipy + numpy.
- **Two-input Operator variant** — easy to add later if a use case
  arises; single-input matches the post-colocation pipeline shape.

## 9. Effort

≈150 LOC implementation + ≈120 LOC tests. Single PR.

| Slice | LOC |
|---|---|
| `segment_signal` (private kernel) | 25 |
| `segmented_psd` / `segmented_csd` / `segmented_coherence` | 50 |
| `along_track_psd_score` (Layer 0) | 50 |
| `psd_score_by_region` | 25 |
| `SegmentedPSDScore` operator | 20 |
| Tests | ~120 |
| Docs / re-exports | 10 |

## 10. Risks / open questions

1. **Where the new code lives.** Three options:
   (a) [`metrics/_src/spectral.py`](../../src/xrtoolz/metrics/_src/spectral.py)
   alongside `psd_score`, (b) new `metrics/_src/segmented_psd.py`,
   (c) `geo/_src/along_track.py` alongside ODC-1.1/1.2 helpers.
   **Recommend (b)** — sibling metric, deserves its own module given
   size; reuses `resolved_scale` from (a).
2. **Per-segment median lon wrap-around.** Upstream has explicit
   Greenwich/dateline branching. We replace with
   `scipy.stats.circmean(lon, high=360, low=0)` — branchless and
   correct for all cases.
3. **Default `npt`.** Upstream uses 64 or 128 depending on chunk
   length. Make `npt` required (no default) — too small a window
   underspecifies low frequencies; users should pick deliberately.
4. **Operator inputs.** Single-Dataset post-colocation (chosen) vs
   two-Dataset. Two-input variant trivial to add later.
5. **Window choice.** scipy default for `welch` is `'hann'`; community
   sometimes uses boxcar. Default `'hann'`, expose `window=` kwarg.
