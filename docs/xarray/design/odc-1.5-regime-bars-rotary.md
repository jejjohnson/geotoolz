# ODC-1.5 — Regime-stratified bar panel + rotary spectrum & polarization panel

**Source survey item:** [ocean-data-challenges-survey.md §1.5](ocean-data-challenges-survey.md)
**Status:** proposed
**Maps to upstream:** `mod_plot.py` from `2024_DC_SSH_mapping_SWOT_OSE` — specifically `plot_stat_by_regimes` and `plot_polarization`.

---

## 1. Motivation

The 1500-line upstream `mod_plot.py` is mostly redundant with the
existing xrtoolz validation viz (`SpatialMapPanel`,
`PSDIsotropicScorePanel`, leaderboard). After cross-referencing, only
two visualisations are genuinely net-new:

1. **`plot_stat_by_regimes`** — grouped bars / styled DataFrame of the
   variance-score `1 − var(err)/var(ref)` per regime
   (coastal / offshore-highvar / offshore-lowvar / equatorial /
   arctic / antarctic). The natural consumer of `scores_by_region`
   from ODC-1.4.
2. **`plot_polarization`** — heatmap of the **rotary spectrum
   polarization** index `r = (S⁻ − S⁺) / (S⁻ + S⁺) ∈ [−1, 1]` over
   `(wavenumber, lat)`. Tells you which sense of rotation (CW vs CCW)
   dominates at each wavelength × latitude. Standard ocean-velocity
   spectral diagnostic.

Item 2 is the more substantive add: it requires a **rotary spectrum
primitive** that doesn't yet exist in xrtoolz, sitting alongside
`power_spectrum` / `cross_spectrum` / `coherence` in
[`transforms.fourier`](../../src/xrtoolz/transforms/_src/fourier.py).

This issue therefore ships:
- A new `rotary_spectrum` data primitive.
- Two `_ValidationPanel` subclasses, `RegionScoreBarPanel` and
  `RotaryPolarizationPanel`, one for each of the missing viz items.

## 2. User stories

### 2.1 Per-regime bar plot from `scores_by_region` (primary)

> *I have the output of `scores_by_region` (ODC-1.4) — a Dataset with
> dims `(region, method)` and data_vars `rmse`, `bias`,
> `explained_variance`. I want a grouped bar chart, one group per
> region, bars per method.*

```python
import xarray as xr
from xrtoolz.viz.validation import RegionScoreBarPanel

ds_scores = xr.open_dataset("scores_by_region.nc")  # dims: (region, method)

panel = RegionScoreBarPanel(
    metrics=["rmse", "explained_variance"],
    region_dim="region",
    method_dim="method",
)
fig, axes = panel(ds_scores)
```

### 2.2 Rotary spectrum + polarization heatmap

> *I have a gridded velocity field `(u, v)` over `(time, lat, lon)`. I
> want the rotary spectrum along `lon` and the polarization heatmap of
> CCW vs CW power in `(wavenumber, lat)`.*

```python
from xrtoolz.transforms import rotary_spectrum
from xrtoolz.viz.validation import RotaryPolarizationPanel

ds_rot = rotary_spectrum(
    ds, u_var="u", v_var="v",
    dim="lon", avg_dims=["time"],
)
# ds_rot has dims (wavenumber, lat) and data_vars psd_ccw, psd_cw, polarization

panel = RotaryPolarizationPanel(
    var="polarization", wavenumber_dim="wavenumber", y_dim="lat",
    wavelength_axis=True,
)
fig, axes = panel(ds_rot)
```

### 2.3 Operator-in-Sequential

```python
from xrtoolz.transforms import RotarySpectrum  # if/when promoted to Operator
from xrtoolz.viz.validation import RotaryPolarizationPanel

pipeline = Sequential([
    RotarySpectrum(u_var="u", v_var="v", dim="lon", avg_dims=["time"]),
    RotaryPolarizationPanel(...),
])
```

(Operator promotion is optional; the primitive function is enough for
v1.)

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| `_ValidationPanel` base | [`viz/validation/_src/base.py`](../../src/xrtoolz/viz/validation/_src/base.py) | reuse |
| Existing panels (SpatialMap, PSDIsotropic, …) | [`viz/validation/_src/`](../../src/xrtoolz/viz/validation/_src/) | cover other upstream plots |
| `power_spectrum`, `cross_spectrum`, `coherence` | [`transforms/_src/fourier.py`](../../src/xrtoolz/transforms/_src/fourier.py) | sibling of new primitive |
| `scores_by_region` consumer | proposed in ODC-1.4 | feeds `RegionScoreBarPanel` |
| Rotary spectrum | — | **add** `rotary_spectrum` |
| Regime bar panel | — | **add** `RegionScoreBarPanel` |
| Polarization heatmap panel | — | **add** `RotaryPolarizationPanel` |

## 4. Design

### 4.1 Rotary spectrum primitive

```python
# src/xrtoolz/transforms/_src/fourier.py
def rotary_spectrum(
    ds: xr.Dataset, *,
    u_var: str, v_var: str,
    dim: str,
    avg_dims: Sequence[str] | None = None,
) -> xr.Dataset:
    """Rotary (counter-rotating) power spectrum from horizontal velocity.

    Computes the FFT of the complex velocity ``w = u + i*v`` along
    ``dim``. The resulting spectrum splits naturally on the sign of
    wavenumber:

    - ``psd_ccw`` — counter-clockwise component (``k > 0``)
    - ``psd_cw``  — clockwise component (``k < 0``, abs-folded)
    - ``polarization`` — ``(psd_cw - psd_ccw) / (psd_cw + psd_ccw)``,
      in [-1, 1]. +1 = pure CW, -1 = pure CCW, 0 = unpolarised /
      linearly-polarised.
    """
```

Implementation:

1. `w = ds[u_var] + 1j * ds[v_var]` (xarray, lazy).
2. `W = xrft.fft(w, dim=dim)` — full two-sided spectrum since `w` is
   complex.
3. `psd = (W * W.conj()).real / df` — power spectral density.
4. Partition on the sign of `wavenumber` coord: positive → `psd_ccw`,
   negative → `psd_cw` (abs-folded so both share a positive-wavenumber
   axis).
5. `polarization = (psd_cw - psd_ccw) / (psd_cw + psd_ccw)` (with NaN
   guard at `psd_cw + psd_ccw = 0`).
6. If `avg_dims` is non-empty, average each output along those dims.

Returns Dataset with `psd_ccw`, `psd_cw`, `polarization` over
`(|wavenumber|, ...)`.

### 4.2 `RegionScoreBarPanel`

```python
# src/xrtoolz/viz/validation/_src/regime_bars.py
class RegionScoreBarPanel(_ValidationPanel):
    """Grouped bar chart of per-region pixel metrics.

    Input: Dataset with dims (region, [method]) and one data_var per
    metric (output shape of `scores_by_region`).
    """

    def __init__(self, *,
                 metrics: Sequence[str] | None = None,
                 region_dim: str = "region",
                 method_dim: str | None = "method",
                 horizontal: bool = False,
                 cmap: str = "tab10",
                 figsize: tuple[float, float] = (8, 5)): ...

    def _build(self, ds): ...
    def _make_fig_axes(self, ...): ...
    def _apply(self, ds, axes): ...
    def get_config(self): ...
```

Implementation: convert the requested metrics to a `pandas.DataFrame`
indexed on the region dim, columns on the method dim; call
`df.plot.bar(ax=ax, ...)` (or `barh` if `horizontal=True`). Inherits
all `_ValidationPanel` plumbing (figure creation, savefig handling,
Operator pattern).

### 4.3 `RotaryPolarizationPanel`

```python
# src/xrtoolz/viz/validation/_src/rotary.py
class RotaryPolarizationPanel(_ValidationPanel):
    """Heatmap of rotary polarization r ∈ [-1, 1].

    Input: Dataset with `polarization(wavenumber, y_dim)` (output of
    `rotary_spectrum`). Renders pcolormesh with a diverging cmap.
    """

    def __init__(self, *,
                 var: str = "polarization",
                 wavenumber_dim: str = "wavenumber",
                 y_dim: str = "lat",
                 cmap: str = "RdBu_r",
                 vmin: float = -1.0, vmax: float = 1.0,
                 wavelength_axis: bool = True,
                 figsize: tuple[float, float] = (6, 8)): ...
```

Implementation: standard `_ValidationPanel` pcolormesh on
`(wavenumber, y_dim)` with diverging cmap clipped to `[-1, 1]`.
`wavelength_axis=True` adds a secondary x-axis labelled `1/k` in km via
`ax.secondary_xaxis`.

## 5. Library leverage

| Need | Library |
|---|---|
| FFT for rotary spectrum | `xrft.fft` (already used by `power_spectrum`) |
| Bar plot | `pandas.DataFrame.plot.bar` (already a transitive dep) |
| Heatmap | matplotlib `pcolormesh` (already in `_ValidationPanel`) |
| Diverging cmap | matplotlib `RdBu_r` |
| Secondary axis | matplotlib `Axes.secondary_xaxis` |

No new dependencies.

## 6. Public API surface

```python
# Primitive
xrtoolz.transforms.rotary_spectrum(ds, *, u_var, v_var, dim, avg_dims)

# Panels
xrtoolz.viz.validation.RegionScoreBarPanel(...)
xrtoolz.viz.validation.RotaryPolarizationPanel(...)
```

`rotary_spectrum` re-exported from `xrtoolz.transforms.__init__`.
Panels re-exported from `xrtoolz.viz.validation.__init__`.

## 7. Tests

| Test | Asserts |
|---|---|
| `rotary_spectrum` on pure CCW signal `e^{ikx}` (u=cos, v=sin) | `psd_ccw` peaks at expected k; `psd_cw ≈ 0`; `polarization ≈ -1` |
| `rotary_spectrum` on pure CW signal | `polarization ≈ +1` |
| `rotary_spectrum` on real-only signal `(u, 0)` | `polarization ≈ 0` (linearly polarised) |
| `rotary_spectrum` Parseval | `Σ(psd_cw + psd_ccw) · dk ≈ var(u) + var(v)` within tol |
| `rotary_spectrum` `avg_dims` | Output dims match expected after averaging |
| `RegionScoreBarPanel` end-to-end | Bars rendered, region labels match input |
| `RegionScoreBarPanel` with `method_dim` | Grouped bars: one bar per (region, method) |
| `RegionScoreBarPanel` `horizontal=True` | Uses `barh` |
| `RotaryPolarizationPanel` end-to-end | Heatmap rendered, cmap clipped [-1, 1] |
| `RotaryPolarizationPanel` `wavelength_axis=True` | Secondary x-axis present with `1/k` ticks |
| Panel `get_config` round-trips | Reconstructed Operators produce identical output |

Target: ~11 cases.

## 8. Out of scope

- All other upstream `mod_plot.py` functions — covered by existing
  `SpatialMapPanel` / `PSDIsotropicScorePanel` / leaderboard.
- **Movie / animation panels** — repo-3 specific (future ODC-3.X).
- **Seasonal PSD-score mosaic** — repo-2 specific (future ODC-2.X).
- **NetCDF-group loader** for `plot_stat_by_regimes` — pipeline detail
  upstream; our `scores_by_region` returns an in-memory Dataset.
- **`hvplot`** — we render via matplotlib for consistency with other
  panels.
- **`RotarySpectrum` Operator** — primitive function is enough for v1.
  Promote to Operator later if a downstream Sequential calls for it.

## 9. Effort

≈80 LOC implementation + ≈80 LOC tests.

| Slice | LOC |
|---|---|
| `rotary_spectrum` primitive | 30 |
| `RegionScoreBarPanel` | 25 |
| `RotaryPolarizationPanel` | 25 |
| Tests | ~80 |
| Docs / re-exports | 10 |

## 10. Risks / open questions

1. **Where `rotary_spectrum` lives.** Options: (a)
   `transforms/_src/fourier.py` alongside `power_spectrum`, (b) new
   `transforms/_src/rotary.py`, (c) `ocn/_src/` since rotary spectra
   are physically meaningful primarily for ocean velocity.
   **Recommend (a)** — sibling of `power_spectrum`; submodule churn
   not justified for a single function.
2. **`RegionScoreBarPanel` data shape dispatch.** Single-method
   `(region,)` and multi-method `(region, method)` need different
   layouts. Dispatch on `method_dim is None` vs `method_dim in ds.dims`.
3. **`y_dim` on polarization panel.** Upstream uses `lat`; the panel
   also makes sense with `time` for KE-flux-style diagnostics. Made a
   parameter.
4. **Wavelength secondary axis.** Implemented via
   `ax.secondary_xaxis(functions=(lambda k: 1/k, lambda l: 1/l))`;
   guard against `k = 0` ticks.
5. **NaN guard in `polarization` denominator.** When
   `psd_cw + psd_ccw == 0` (no power), the ratio is undefined. Use
   `np.where` to emit NaN rather than dividing.
