# ODC-3.4 — Generic `PairwiseComparePanel` (study-vs-ref with diff)

**Source survey item:** [ocean-data-challenges-survey.md §3.4](ocean-data-challenges-survey.md)
**Status:** proposed
**Maps to upstream:** `compare_stat_score_map`, `compare_psd_score` from
`src/mod_compare.py` in `2024c_DC_4DMedSea-ESA`.

---

## 1. Motivation

The 4DMedSea repo's [`mod_compare.py`](https://github.com/ocean-data-challenges/2024c_DC_4DMedSea-ESA/blob/main/src/mod_compare.py)
ships two paired-comparison plot recipes —
`compare_stat_score_map(study, ref)` and `compare_psd_score(study, ref)` —
each producing the same 6-panel layout (two scales × three methods:
ref, study, relative-diff). Plus two utility functions
(`regional_zoom`, `convert_longitude`) that are already covered by
[`geo/_src/subset.py`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/geo/_src/subset.py).

The new pattern that's *not* covered by existing xrtoolz viz primitives
is the **paired study-vs-ref-with-diff column**:

```
┌─────────┬─────────┬─────────┐
│   ref   │  study  │  diff   │
└─────────┴─────────┴─────────┘
```

Once that exists as a generic panel, the upstream's 6-panel layout
is composed for free:

```python
FacetPanel(                 # ODC-2.3 — facet by scale
    PairwiseComparePanel(   # ODC-3.4 — compare ref vs study + diff
        SpatialMapPanel(var="error_variance"),
        diff="relative",
    ),
    facet_dim="scale",
)
```

So the actual gap is one new `_ValidationPanel`, not a new pipeline.

## 2. User stories

### 2.1 Compare two methods on a spatial RMSE map (primary)

> *I have an `(method, lat, lon)` Dataset with method coord
> `["DUACS", "MIOST"]`. I want a side-by-side comparison plus the
> relative-percent difference panel.*

```python
import xarray as xr
from xrtoolz.viz.validation import SpatialMapPanel, PairwiseComparePanel

ds = xr.open_dataset("error_variance_by_method.nc")  # dims: (method, lat, lon)

panel = PairwiseComparePanel(
    SpatialMapPanel(var="err_var", projection="gulf_stream", cmap="Reds"),
    method_dim="method",
    diff="relative",
    diff_kwargs={"cmap": "coolwarm", "vmin": -20, "vmax": 20},
)
fig = panel(ds)
```

### 2.2 Compare two PSD-score panels (absolute diff)

```python
from xrtoolz.viz.validation import PSDSpaceTimeScorePanel, PairwiseComparePanel

panel = PairwiseComparePanel(
    PSDSpaceTimeScorePanel(threshold=0.5),
    method_dim="method",
    diff="absolute",
)
fig = panel(ds_psd_by_method)
```

### 2.3 Reproduce the upstream 6-panel mosaic via `FacetPanel`

> *I want the upstream "two scales × (ref, study, diff)" layout.*

```python
from xrtoolz.viz.validation import FacetPanel, PairwiseComparePanel, SpatialMapPanel

panel = FacetPanel(
    PairwiseComparePanel(
        SpatialMapPanel(var="error_variance", cmap="Reds"),
        method_dim="method",
        diff="relative",
    ),
    facet_dim="scale",          # e.g. coord values ["all_scale", "filtered"]
    nrows=2, ncols=1,
)
fig = panel(ds)            # dims (scale, method, lat, lon)  — returns Figure
```

### 2.4 As an Operator inside a Sequential

```python
from xrtoolz.core import Sequential

pipeline = Sequential([
    BinnedResiduals2D(...),         # ODC-1.4
    # produces (method, lat, lon) gridded errors
    PairwiseComparePanel(SpatialMapPanel(var="rmse"), diff="relative"),
])
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| `_ValidationPanel` base + Operator | [`viz/validation/_src/base.py`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/viz/validation/_src/base.py) | reuse |
| `SpatialMapPanel`, `PSDSpaceTimeScorePanel`, etc. | [`viz/validation/_src/`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/viz/validation/_src/) | reuse as inner panels |
| `FacetPanel` (N-way over a categorical dim) | proposed in ODC-2.3 | sibling concept |
| `regional_zoom`, `convert_longitude` | [`geo/_src/subset.py`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/geo/_src/subset.py) | already present |
| Pairwise A-vs-B-with-diff comparison | — | **add** `PairwiseComparePanel` |

## 4. Design

### 4.1 Generality

`PairwiseComparePanel` accepts the same inner-panel types as
`FacetPanel`:

- ✅ Any `_ValidationPanel` subclass (`SpatialMapPanel`,
  `PSDSpaceTimeScorePanel`, `PSDIsotropicScorePanel`,
  `RegionScoreBarPanel`, …).
- ✅ Plain callables `(xr.Dataset, mpl.axes.Axes) -> Any`.
- ❌ Plots that own the whole figure (seaborn `JointGrid`, etc.) — same
  boundary as `FacetPanel`.

Cartopy projections forwarded from the inner panel's `projection`
attribute (same trick as `FacetPanel`); explicit `subplot_kw=` overrides
always win.

### 4.2 Class

```python
# src/xrtoolz/viz/validation/_src/compare.py
class PairwiseComparePanel(_ValidationPanel):
    """Side-by-side study vs. reference comparison with optional diff cell.

    Input is a Dataset carrying ``method_dim`` of size 2 (position 0 is
    the reference / baseline; position 1 is the study / candidate).
    Renders three cells: ref, study, diff (or two if ``diff="none"``).
    """

    def __init__(
        self,
        panel: _ValidationPanel | Callable[[xr.Dataset, Any], Any],
        *,
        method_dim: str = "method",
        diff: Literal["relative", "absolute", "none"] = "relative",
        diff_kwargs: dict[str, Any] | None = None,
        diff_label: str | None = None,
        layout: Literal["row", "col"] = "row",
        sharex: bool = True,
        sharey: bool = True,
        figsize_per_panel: tuple[float, float] = (5, 4),
        subplot_kw: dict[str, Any] | None = None,
    ): ...
```

### 4.3 Internals

`_make_fig_axes`:

1. Determine cell count: `n = 3 if diff != "none" else 2`.
2. Build subplot grid `(1, n)` for `layout="row"`, `(n, 1)` for `"col"`.
3. Resolve `subplot_kw` from inner panel's `projection` (when present).
   Explicit user `subplot_kw=` wins.

`_build(fig, axes, ds)`:

1. Validate `ds.sizes[method_dim] == 2`; raise informative error otherwise.
2. `ref = ds.isel({method_dim: 0})`; `study = ds.isel({method_dim: 1})`.
3. Compute diff per mode:
   ```python
   if diff == "absolute":
       ds_diff = study - ref
   elif diff == "relative":
       ds_diff = xr.where(ref != 0, 100 * (study - ref) / ref, np.nan)
   else:                        # "none"
       ds_diff = None
   ```
4. Render in order: ref → cell 0, study → cell 1, diff → cell 2.
   Each call dispatches on inner-panel type via the `_build(fig, ax,
   ds)` hook on the base class:
   ```python
   def _render(panel, ds, ax, kwargs_override=None):
       if isinstance(panel, _ValidationPanel):
           inner = _clone_with_overrides(panel, kwargs_override)
           inner._build(fig, ax, ds)
       else:
           panel(ds, ax)        # callable path
   ```
   `diff_kwargs` is the override applied only to the diff cell — typical
   contents `{"cmap": "coolwarm", "vmin": -20, "vmax": 20}`.
5. Per-cell title from `ds[method_dim].values[i]` for ref/study,
   `diff_label or {"absolute": "Δ", "relative": "Δ%"}[diff]` for diff.
6. Diff label suffix appended: `"DUACS → MIOST: Δ%"`.

`get_config`:
- Recurses into inner panel's `get_config` (when `_ValidationPanel`).
- For callable inner: emits `{"panel": "<callable>"}` flag, non-roundtrippable.
- Includes all top-level kwargs.

### 4.4 The clone-with-overrides helper

For `_ValidationPanel` inner, applying `diff_kwargs` cleanly means
producing a new instance with overridden constructor kwargs:

```python
def _clone_with_overrides(panel: _ValidationPanel, overrides: dict | None):
    if not overrides:
        return panel
    config = panel.get_config()
    config.update(overrides)
    return type(panel)(**config)
```

This is the same pattern Operator-clone uses elsewhere; works because
`_ValidationPanel.get_config()` round-trips through `__init__`.

For callable inner, `diff_kwargs` is ignored with a warning (no
constructor to override).

## 5. Library leverage

| Need | Library |
|---|---|
| Subplot grid (3 cells row / col) | `matplotlib.pyplot.subplots(1, 3)` / `(3, 1)` |
| Cartopy GeoAxes | inherited from inner panel via `subplot_kw` |
| Per-method selection | `xarray.Dataset.isel({method_dim: i})` |
| Diff arithmetic | `xarray` arithmetic + `xr.where` for NaN guard |
| Diverging cmap | matplotlib `coolwarm` / `RdBu_r` |

No new dependencies.

## 6. Public API surface

```python
xrtoolz.viz.validation.PairwiseComparePanel(
    panel,                   # _ValidationPanel | Callable[(ds, ax), Any]
    *,
    method_dim="method",
    diff="relative",         # "relative" | "absolute" | "none"
    diff_kwargs=None,
    diff_label=None,
    layout="row",            # "row" | "col"
    sharex=True, sharey=True,
    figsize_per_panel=(5, 4),
    subplot_kw=None,
)
```

Re-exported from `xrtoolz.viz.validation.__init__`.

## 7. Tests

| Test | Asserts |
|---|---|
| 2-method Dataset → 3-cell row when `diff="relative"` | grid shape `(1, 3)` |
| `diff="absolute"` arithmetic | exact match to `B - A` |
| `diff="relative"` arithmetic | exact match to `100 * (B - A) / A` |
| `diff="relative"` with `A == 0` cells | NaN at those cells |
| `diff="none"` | 2-cell layout |
| `layout="col"` | grid shape `(3, 1)` |
| Wrong-size `method_dim` (≠ 2) | raises informative `ValueError` |
| Cartopy passthrough with `SpatialMapPanel(projection="gulf_stream")` | grid built with `GeoAxes` |
| Explicit `subplot_kw=` overrides inner projection | user value used |
| `diff_kwargs` applies to diff cell only | ref + study cells unchanged |
| Callable inner panel (lambda) | works, `diff_kwargs` warning emitted |
| `get_config` round-trip with `_ValidationPanel` inner | reconstructed panel produces identical figure |
| Composes with `FacetPanel` over scale dim | `(scale, method, ...)` Dataset renders 6-panel mosaic |

Target: ~13 cases.

## 8. Out of scope

- **3-way+ comparisons.** `FacetPanel(panel, facet_dim="method")` covers
  raw N-way side-by-side without diff. Diff over multiple methods needs
  a different UX (matrix? row of pairwise diffs?) — separate proposal.
- **`regional_zoom`, `convert_longitude`** — already in
  [`geo/_src/subset.py`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/geo/_src/subset.py).
- **NetCDF group loading** (`xr.open_dataset(..., group="all_scale")`) —
  pipeline I/O detail, not library concern.
- **`hvplot` rendering** — matplotlib only.
- **Plots that own the whole figure** (seaborn `JointGrid`, etc.).

## 9. Effort

≈90 LOC implementation + ≈90 LOC tests. Single PR.

| Slice | LOC |
|---|---|
| `PairwiseComparePanel` | 80 |
| Tests | ~90 |
| Docs / re-exports | 10 |

## 10. Risks / open questions

1. **Sign convention.** `(study - ref) / ref` (positive = study has more
   error). Matches upstream argument order (ref → position 0, study →
   position 1). Document explicitly.
2. **NaN guard in relative diff.** `xr.where(ref != 0, …, np.nan)`.
3. **Diff cmap default.** When `diff_kwargs` is unset, default to
   `{"cmap": "coolwarm"}` for diff cell so a diverging cmap is used
   regardless of inner panel's choice.
4. **Method-position convention.** Position 0 = ref / baseline; position
   1 = study / candidate. Documented in docstring + `__init__` example.
5. **`_clone_with_overrides`.** Relies on `get_config()` ↔ `__init__`
   round-trip. Works for all existing `_ValidationPanel` subclasses
   (already verified by their own round-trip tests).
6. **Where it lives.** `viz/validation/_src/compare.py` (new). Sibling
   of `facet.py`. Could fold into one module; chosen separate for
   clarity (different abstractions: facet ≠ compare).
7. **Operator promotion.** Inheriting from `_ValidationPanel` already
   gives Operator semantics. No extra work.
