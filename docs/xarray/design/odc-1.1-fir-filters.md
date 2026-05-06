# ODC-1.1 — FIR filters (Lanczos + Kaiser) and along-track wavelength bandpass

**Source survey item:** [ocean-data-challenges-survey.md §1.1](ocean-data-challenges-survey.md)
**Status:** proposed
**Maps to upstream:** `mod_filter.py` from `2024_DC_SSH_mapping_SWOT_OSE` (and equivalents in the 2023 / 2024c repos).

---

## 1. Motivation

The ocean-data-challenges evaluation pipeline filters along-track satellite
altimetry (SLA / SSH) before scoring against gridded reconstructions. Two
operations recur in every notebook:

1. **Low-pass an along-track segment** to remove wavelengths shorter than
   λ<sub>min</sub> (instrument noise, sub-mesoscale variability the gridded
   product cannot resolve).
2. **Band-pass an along-track segment** to (λ<sub>min</sub>,
   λ<sub>max</sub>) — keeping mesoscale energy while suppressing both the
   short-wavelength noise and the large-scale geoid/MDT residual.

The community reaches for **Lanczos** (windowed-sinc) FIR filters because:

- FIR → exactly zero-phase (linear-phase, so `filtfilt` with `[1.0]`
  denominator is just symmetric convolution — no phase distortion).
- Sharp roll-off with controlled ripple — important when a downstream
  effective-resolution metric `λx` depends on the filter band edges.
- Cutoffs are specified in **wavelength (km)**, the physical unit users
  reason about. IIR Butterworth (which we already ship) takes cutoffs as
  a fraction of Nyquist, which is opaque for non-uniform along-track
  spacing.

We currently expose only Butterworth IIR
([`xr_toolz.interpolate.lowpass_filter`](../../src/xr_toolz/interpolate/_src/smooth.py)).
That covers gridded data well but is the wrong default for along-track
SLA: gentle roll-off, opaque cutoff units, and IIR phase compensation
relies on `sosfiltfilt`'s reverse-pass which is sensitive to short
segments.

This proposal adds the FIR family alongside Butterworth — Lanczos as the
community standard, Kaiser as the tunable engineering default — plus a
domain-aware wrapper that takes cutoffs in **kilometres** rather than
normalized Nyquist.

## 2. User stories

### 2.1 Single-segment along-track bandpass (primary)

> *As an SSH-mapping researcher, I have a DUACS L4 reconstruction and a
> SWOT along-track product. I want to bandpass-filter the SWOT SLA to
> (65 km, 500 km) before colocating my reconstruction onto it, so that
> the residual I score is in the mesoscale band the reconstruction
> claims to resolve.*

```python
import xarray as xr
from xr_toolz.geo import bandpass_wavelength

ds_track = xr.open_dataset("swot_along_track.nc")
ds_band  = bandpass_wavelength(
    ds_track,
    dim="num_lines",
    lambda_min_km=65.0,
    lambda_max_km=500.0,
    method="lanczos",
)
```

`spacing_km` is auto-derived from `ds_track.longitude` /
`ds_track.latitude` via geodesic distance (`pyproj.Geod`, ellipsoidal —
more accurate than haversine).

### 2.2 Tunable engineering default

> *I'm not in the SSH-mapping community and I just want a clean
> linear-phase low-pass with a 60 dB stop-band on a regular time series.
> I don't want to hand-pick FIR coefficients.*

```python
from xr_toolz.interpolate import fir_filter

ds_smooth = fir_filter(
    ds, dim="time",
    cutoff=0.05,          # normalized Nyquist
    method="kaiser",
    attenuation_db=60.0,  # auto-picks β and tap count
)
```

### 2.3 Reproducible Lanczos for community comparability

> *I'm reproducing the 2024 SSH-mapping-SWOT-OSE leaderboard. I need a
> Lanczos low-pass with cutoff at λ = 65 km on a 7 km-spaced track and I
> need the result to match the reference notebooks bit-for-bit modulo
> floating-point.*

```python
ds_lp = bandpass_wavelength(
    ds, dim="num_lines",
    lambda_min_km=65.0, lambda_max_km=None,   # low-pass only
    method="lanczos", num_taps=129,
)
```

### 2.4 As a Layer-1 Operator inside a Sequential

> *I'm building a reusable along-track scoring pipeline and want the
> bandpass step to be a configurable, serializable Operator I can drop
> into a `Sequential`.*

```python
from xr_toolz.geo import BandpassWavelength
from xr_toolz.core import Sequential

pipeline = Sequential([
    BandpassWavelength(
        dim="num_lines", var="ssha",
        lambda_min_km=65.0, lambda_max_km=500.0,
        method="lanczos",
    ),
    AlongTrackColocate(...),     # ODC-1.2, future
    SegmentedPSDScore(...),      # ODC-1.3, future
])
```

## 3. What we already have / what's missing

| Capability | Current state | This proposal |
|---|---|---|
| Butterworth IIR low/high/band/stop | [`array_smooth.lowpass_filter`](../../src/xr_toolz/interpolate/_src/array_smooth.py) | Unchanged |
| Generic IIR family (Cheby, Ellip, Bessel) | — | Out of scope |
| FIR Lanczos | — | **Add** |
| FIR Kaiser (tunable atten ↔ taps) | — | **Add** |
| FIR application machinery | — | Use `scipy.signal.filtfilt` with FIR taps |
| Cutoff in normalized Nyquist | Yes | Yes, for `fir_filter` |
| Cutoff in **wavelength (km)** | — | **Add** via `bandpass_wavelength` |
| Median along-track spacing | `_haversine_km` private helper | **Add** `median_dx_km` (uses `pyproj.Geod`, public) |
| `Operator` wrapper | — | **Add** `BandpassWavelength` Layer-1 operator |

## 4. Design

### 4.1 Tap design (single helper, two windows)

```python
# src/xr_toolz/interpolate/_src/array_smooth.py
def _fir_taps(
    *,
    cutoff: float | tuple[float, float],
    method: str,                    # "lanczos" | "kaiser"
    btype: str,                     # "low" | "high" | "bandpass" | "bandstop"
    num_taps: int | None = None,    # FIR length, must be odd
    attenuation_db: float | None = None,  # Kaiser only
) -> NDArray[np.floating]:
    """Design symmetric FIR taps via windowed-sinc."""
```

Lanczos taps (low-pass case):

```python
M = (num_taps - 1) // 2
n = np.arange(-M, M + 1)
taps = np.sinc(2 * cutoff * n) * np.sinc(n / M)
taps /= taps.sum()
```

Band-pass = low-pass(f<sub>hi</sub>) − low-pass(f<sub>lo</sub>); high-pass
= identity − low-pass. Same construction for Kaiser, swap the window for
`scipy.signal.kaiser(num_taps, beta)` where β is derived from
`attenuation_db` via `scipy.signal.kaiser_beta`. Tap-count auto-pick uses
`scipy.signal.kaiserord(attenuation_db, transition_width)` when
`num_taps is None` for Kaiser; for Lanczos the default is
`2 * ceil(2 / cutoff) + 1`.

### 4.2 Tier A — array kernel

```python
# src/xr_toolz/interpolate/_src/array_smooth.py
def fir_filter(
    arr: ArrayLike, *,
    axis: int = -1,
    cutoff: float | Sequence[float],
    method: Literal["lanczos", "kaiser"] = "lanczos",
    btype: str = "low",                   # "low" | "high" | "bandpass" | "bandstop"
    num_taps: int | None = None,
    attenuation_db: float | None = None,  # Kaiser-only
) -> NDArray[np.floating]:
    """Zero-phase FIR filter via symmetric windowed-sinc taps."""
```

Implementation: `_fir_taps(...)` → `scipy.signal.filtfilt(taps, [1.0],
arr, axis=axis, padtype='odd', method='gust')`. The `'gust'` method
handles short segments better than the default. Complex inputs filtered
component-wise (existing `_as_floating` helper).

### 4.3 Tier B — xarray wrapper

```python
# src/xr_toolz/interpolate/_src/smooth.py
def fir_filter(
    ds: xr.Dataset, *,
    dim: str,
    cutoff: float | Sequence[float],
    method: str = "lanczos",
    btype: str = "low",
    num_taps: int | None = None,
    attenuation_db: float | None = None,
) -> xr.Dataset
```

Mirrors existing `lowpass_filter` exactly — same `_apply_along_dim`
plumbing, same Dataset-pass-through semantics for non-numeric / no-dim
variables.

### 4.4 Domain-aware bandpass

```python
# src/xr_toolz/geo/_src/along_track.py — new module
def median_dx_km(lon: ArrayLike, lat: ArrayLike) -> float:
    """Median geodesic spacing between consecutive points (km, WGS-84).

    Uses pyproj.Geod(ellps='WGS84').line_lengths.
    """

def bandpass_wavelength(
    ds: xr.Dataset, *,
    dim: str,
    lambda_min_km: float | None,    # high-pass edge; None → no high-pass
    lambda_max_km: float | None,    # low-pass edge;  None → no low-pass
    spacing_km: float | None = None,
    method: str = "lanczos",
    num_taps: int | None = None,
    attenuation_db: float | None = None,
    lon: str = "longitude",
    lat: str = "latitude",
) -> xr.Dataset:
    """Bandpass along ``dim`` with cutoffs in wavelength (km)."""
```

Internals:

1. If `spacing_km is None`, derive from `ds[lon]` / `ds[lat]` via
   `median_dx_km`.
2. Translate `λ → cutoff_norm = 2 * spacing_km / λ_km`.
3. Validate `0 < cutoff_norm < 1`; raise with a helpful message if a
   wavelength is below `2 × spacing_km` (Nyquist).
4. Resolve `btype` from which of (λ<sub>min</sub>, λ<sub>max</sub>) are
   set — at least one required:
    | λ<sub>min</sub> | λ<sub>max</sub> | btype | cutoff |
    |---|---|---|---|
    | None | set | `"low"` | `2·dx/λ_max` |
    | set | None | `"high"` | `2·dx/λ_min` |
    | set | set | `"bandpass"` | `(2·dx/λ_max, 2·dx/λ_min)` |
5. Delegate to `fir_filter`.

### 4.5 Layer-1 Operator

```python
# src/xr_toolz/geo/operators.py
class BandpassWavelength(Operator):
    """Apply a wavelength-domain bandpass to a single variable."""

    def __init__(
        self, *, dim, lambda_min_km=None, lambda_max_km=None,
        spacing_km=None, method="lanczos", num_taps=None,
        attenuation_db=None, lon="longitude", lat="latitude",
    ): ...

    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...
    def get_config(self) -> dict[str, Any]: ...
    def __repr__(self) -> str: ...
```

Standard split-config / `__call__` shape; nothing stateful.

## 5. Library leverage

| Need | Library call | Notes |
|---|---|---|
| FIR design (Kaiser tap count) | `scipy.signal.kaiserord`, `kaiser_beta`, `windows.kaiser` | β-from-attenuation built in |
| Tap application (zero-phase) | `scipy.signal.filtfilt(taps, [1.0], …, method='gust')` | Robust on short segments |
| Sinc | `numpy.sinc` | Normalized (π·x) variant — matches our derivation |
| Geodesic distance | `pyproj.Geod(ellps='WGS84').line_lengths` | Already a dependency. More accurate than haversine; vectorised; consecutive-only (no quadratic blow-up). |
| Lanczos window | — | 1-line: `np.sinc(n / M)`. No scipy primitive. |
| Existing IIR | `scipy.signal.butter`, `sosfiltfilt` | Untouched |

No new top-level dependencies.

## 6. API summary (public surface)

```python
# Tier A — array kernels
xr_toolz.interpolate.array.fir_filter(arr, *, axis, cutoff, method, btype,
                                      num_taps, attenuation_db)

# Tier B — xarray
xr_toolz.interpolate.fir_filter(ds, *, dim, cutoff, method, btype,
                                num_taps, attenuation_db)

# Domain-aware (geo)
xr_toolz.geo.median_dx_km(lon, lat)
xr_toolz.geo.bandpass_wavelength(ds, *, dim, lambda_min_km, lambda_max_km,
                                 spacing_km, method, num_taps,
                                 attenuation_db, lon, lat)

# Operator
xr_toolz.geo.BandpassWavelength(...)
```

Existing `lowpass_filter` (Butterworth) is unchanged. `fir_filter` is the
new sibling for FIR methods.

## 7. Tests

| Test | Asserts |
|---|---|
| Lanczos low-pass on a known sum-of-sines | Out-of-band amplitude < 1% of in-band |
| Kaiser low-pass with `attenuation_db=60` | Stop-band peak ≤ −60 dB (FFT of impulse response) |
| Tap count auto-pick | `num_taps` is odd; passes through `_fir_taps` without error |
| `fir_filter` zero-phase | Phase response (FFT) is purely real within tol |
| `bandpass_wavelength` round-trip | Cutoff translation matches manual `2·dx/λ` |
| `bandpass_wavelength` Nyquist guard | Raises `ValueError` for λ < 2·dx |
| `bandpass_wavelength` lambda-bounds | At least one of (min, max) required; min < max |
| `median_dx_km` against known segment | Within 0.1% of pyproj reference |
| Operator round-trip via `get_config` | Reconstructed operator produces identical output |
| Dataset pass-through | Non-numeric / no-dim variables untouched |

Target: ~12 new test cases.

## 8. Out of scope

- **Generalizing `lowpass_filter` to accept `ftype` (Cheby/Ellip/Bessel)** —
  separate proposal if needed.
- **Savitzky–Golay** (`scipy.signal.savgol_filter`) — useful when
  downstream needs derivatives (geostrophic velocity from filtered SSH),
  but conceptually different from windowed-sinc bandpass. Future item.
- **Tide filters** (Doodson `X0`, Godin, Demerliac) — small wrappers that
  pick specific Lanczos cutoffs. Trivially constructible on top of
  `bandpass_wavelength` once it exists. Future item, only if a user asks.
- **Wavelet bandpass** (`pywt`) — different paradigm, adds a dependency.
- **Median filter** — `scipy.signal.medfilt` is sufficient as-is.
- **`Sequential` recipe** for the full SSH-mapping pipeline — pending
  ODC-1.2 (colocation) and ODC-1.3 (segmented PSD).

## 9. Effort

≈100 LOC implementation + ≈100 LOC tests + docstring examples. Single PR.

| Slice | LOC | Notes |
|---|---|---|
| `_fir_taps` + `fir_filter` (Tier A) | 50 | scipy does the heavy lifting |
| `fir_filter` (Tier B xarray wrapper) | 15 | Reuse `_apply_along_dim` |
| `median_dx_km` + `bandpass_wavelength` | 35 | New `geo/_src/along_track.py` |
| `BandpassWavelength` operator | 20 | Standard pattern |
| Tests | ~100 | 12 cases |
| Docs / re-exports | 10 | `__init__.py` updates, validation docs page |

## 10. Risks / open questions

1. **Default `num_taps` for Lanczos.** The `2 * ceil(2 / cutoff) + 1`
   heuristic gives ~2 main-lobe widths of taps, which matches the
   community implementations but is conservative. We should document
   explicitly and let users override.
2. **`filtfilt` `method='gust'` vs `'pad'`.** `'gust'` is more robust on
   short segments but slower. Pick `'pad'` as the default to match
   scipy and document `method` as a future parameter only if needed.
3. **`bandpass_wavelength` longitude/latitude lookup.** Hard-coded
   default names (`"longitude"`, `"latitude"`) match CF conventions but
   not e.g. `"lon"` / `"lat"`. Solution: expose `lon=` / `lat=` kwargs
   (above), default to the CF names.
4. **NaN handling.** `filtfilt` does not skip NaNs — they propagate
   across the whole convolution kernel. The upstream notebooks
   pre-segment along-track data into NaN-free chunks before filtering.
   We document this limitation; an `interpolate_nans=True` convenience
   flag (using `xr_toolz.interpolate.gap_fill`) is a future enhancement,
   not blocking.
5. **Operator placement.** `BandpassWavelength` could equally live in
   `interpolate.operators` (mirrors the smoother) or `geo.operators`
   (along-track domain). Choosing `geo` because the wavelength-in-km
   semantics and lon/lat lookup are geo-domain concerns.
