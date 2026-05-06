# ODC-2.3 — Generic `FacetPanel` (subsumes the seasonal PSD-score mosaic)

**Source survey item:** [ocean-data-challenges-survey.md §2.3](ocean-data-challenges-survey.md)
**Status:** proposed
**Maps to upstream:** `plot_psd_score_seasonal` from `src/mod_plot.py` in `2023_SSH_mapping_train_eNATL60_test_NATL60-`.

---

## 1. Motivation

The upstream `plot_psd_score_seasonal(ds_psd, ...)` is **the existing
single-panel `plot_psd_score` rendered four times in a 2×2 grid** —
one panel per `experiment` index, with an `order_mosaic` argument
remapping `experiment` → `(row, col)`. The "seasons" labelling in the
function name is accidental; each subplot is just
`ds_psd.isel(experiment=i)` with the coord value as the title.

xr_toolz already exposes the single-panel renderer
([`PSDSpaceTimeScorePanel`](../../src/xr_toolz/viz/validation/_src/psd.py)).
What's missing is a thin wrapper that turns **any** existing single-axes
panel into a faceted grid.

The seasonal mosaic then drops out as the special case
`FacetPanel(PSDSpaceTimeScorePanel(...), facet_dim="season")` — same
machinery for `experiment`, `method`, `region`, `lead_time`, anything.

This issue ships **one generic `FacetPanel`** plus an optional
`seasonal_groupby` helper. No domain-specific `SeasonalPSDScorePanel`.

## 2. User stories

### 2.1 Reproduce the upstream seasonal mosaic (primary)

> *I have a Dataset of PSD scores indexed by season (DJF/MAM/JJA/SON)
> and want the canonical 2×2 mosaic with the 0.5 contour and shared
> wavelength axes.*

```python
from xr_toolz.viz.validation import (
    PSDSpaceTimeScorePanel, FacetPanel, seasonal_groupby,
)

ds_seasonal = seasonal_groupby(ds_psd_score, time="time")
# dims: (season, freq_lon, freq_time)

panel = FacetPanel(
    PSDSpaceTimeScorePanel(threshold=0.5),
    facet_dim="season",
    ncols=2,
)
fig = panel(ds_seasonal)
```

### 2.2 Facet a SpatialMapPanel by experiment

> *I have a Dataset with a `(experiment, lat, lon)` SSH error map and
> want one cartopy panel per experiment.*

```python
from xr_toolz.viz.validation import SpatialMapPanel, FacetPanel

panel = FacetPanel(
    SpatialMapPanel(var="ssh_err", projection="north_atlantic",
                    cmap="RdBu_r", vmin=-0.3, vmax=0.3),
    facet_dim="experiment",
    sharebar=True,
)
fig = panel(ds_err)   # one cartopy GeoAxes per experiment
```

### 2.3 Facet ad-hoc matplotlib code (callable, no subclass)

> *I have one-off plotting code I don't want to wrap in a panel class.*

```python
def quick_plot(ds, ax):
    ax.plot(ds.x, ds.y)
    ax.set_title(str(ds.attrs.get("label", "")))

panel = FacetPanel(quick_plot, facet_dim="run", ncols=3)
fig = panel(ds_runs)
```

### 2.4 As an Operator inside a Sequential

```python
from xr_toolz.core import Sequential
from xr_toolz.metrics import PSDSpaceTimeScore   # if/when promoted

pipeline = Sequential([
    PSDSpaceTimeScore(...),
    FacetPanel(PSDSpaceTimeScorePanel(...), facet_dim="experiment"),
])
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| `_ValidationPanel` base + Operator | [`viz/validation/_src/base.py`](../../src/xr_toolz/viz/validation/_src/base.py) | reuse |
| `PSDSpaceTimeScorePanel` (single panel) | [`viz/validation/_src/psd.py:518`](../../src/xr_toolz/viz/validation/_src/psd.py) | reuse |
| `SpatialMapPanel` with cartopy | [`viz/validation/_src/spatial.py`](../../src/xr_toolz/viz/validation/_src/spatial.py) | reuse |
| Other single-axes panels (PSD/skill/Lagrangian/events/budgets) | [`viz/validation/_src/`](../../src/xr_toolz/viz/validation/_src/) | reuse |
| Generic facet wrapper | — | **add** `FacetPanel` |
| Seasonal-mean helper | — | **add** optional `seasonal_groupby` |
| `season` / `experiment` mosaic | — | falls out of `FacetPanel` |

## 4. Design

### 4.1 Generality boundary

`FacetPanel` accepts **either** a `_ValidationPanel` **or** a plain
callable `(xr.Dataset, mpl.axes.Axes) -> Any`. This covers:

- ✅ Every existing `_ValidationPanel` subclass (PSDSpaceTime, SpatialMap,
  RegionScoreBar, Rotary, LeadTimeSkill, ScaleSkill, …).
- ✅ Ad-hoc plotting functions and lambdas — no subclassing required.
- ❌ Plots that own the whole figure (seaborn `JointGrid`, `pairplot`)
  — they don't accept a single `ax`. Documented as out of scope.
- ❌ Plotly / hvplot / bokeh — different backend; matplotlib-only by
  design.

Trade-off: callables don't carry `get_config`, so a `FacetPanel`
wrapping a callable can't fully round-trip via Operator
serialization. Document; serialization is best-effort with callables.

### 4.2 Cartopy / projection passthrough

Some panels (notably `SpatialMapPanel`) require `cartopy.GeoAxes`
rather than plain `matplotlib.Axes`. The inner panel knows its
projection; `FacetPanel` doesn't.

Resolution: `FacetPanel` looks for a `projection` attribute on the
inner panel; when present, forwards it to
`plt.subplots(subplot_kw={"projection": ...})`. Users can also pass
`subplot_kw=` directly to override.

```python
def _resolve_subplot_kw(self, panel) -> dict:
    if isinstance(panel, _ValidationPanel) and hasattr(panel, "projection"):
        proj = panel.projection
        if proj is not None:
            return {"projection": _resolve_projection(proj)}
    return {}
```

### 4.3 `FacetPanel`

```python
# src/xr_toolz/viz/validation/_src/facet.py
class FacetPanel(_ValidationPanel):
    """Render any single-axes panel faceted across a categorical dim.

    Accepts a `_ValidationPanel` or a callable
    ``(xr.Dataset, mpl.axes.Axes) -> Any``. Layout is near-square by
    default (`ncols = ceil(sqrt(N))`); cartopy projections are
    forwarded via the inner panel's ``projection`` attribute.
    """

    def __init__(
        self,
        panel: _ValidationPanel | Callable[[xr.Dataset, Any], Any],
        *,
        facet_dim: str,
        ncols: int | None = None,
        nrows: int | None = None,
        sharex: bool = True,
        sharey: bool = True,
        sharebar: bool = False,
        figsize_per_panel: tuple[float, float] = (5, 4),
        title_format: str = "{value}",
        subplot_kw: dict[str, Any] | None = None,
    ): ...
```

**`_make_fig_axes`** — derive `n = ds.sizes[facet_dim]`; pick layout if
not set; build `subplot_kw` from inner-panel `projection` (overridable
by user kwarg); call `plt.subplots(nrows, ncols, sharex=sharex,
sharey=sharey, figsize=(ncols*w, nrows*h), subplot_kw=subplot_kw)`.
Return `(fig, axes)`.

**`_build(fig, axes, ds)`** — iterate
`for i, value in enumerate(ds[facet_dim].values)`:
1. `ds_slice = ds.isel({facet_dim: i})`.
2. Resolve `ax` at `(i // ncols, i % ncols)`.
3. Dispatch: if inner is `_ValidationPanel`, call
   `panel._build(fig, ax, ds_slice)` — the existing render-into-axes
   hook on the base class. Else call `panel(ds_slice, ax)`.
4. Capture returned mappable for shared-colorbar handling.
5. Set per-axes title via `title_format.format(value=value, index=i)`.
6. After the loop, hide unused trailing axes when `n < nrows*ncols`.
7. If `sharebar=True` and any mappable was captured, render a single
   shared colorbar via `fig.colorbar(mappable, ax=axes.ravel().tolist())`;
   else fall back to per-axes (issue a warning if `sharebar=True`
   requested but no mappable returned).

`__call__` (inherited from `_ValidationPanel`) wraps `_build` in
`_apply`, runs `_make_fig_axes` for the grid, applies title / save /
show hooks, and **returns the `Figure`** — same contract as every
other `_ValidationPanel`.

**`get_config`** — emit inner panel's `get_config()` recursively when
it's a `_ValidationPanel`; for callables, emit `{"panel": "<callable>"}`
plus a flag indicating non-roundtrippable. Match the pattern other
composite operators use.

### 4.4 Optional `seasonal_groupby` helper

```python
def seasonal_groupby(
    ds: xr.Dataset, *,
    time: str = "time",
    reduction: str = "mean",
) -> xr.Dataset:
    """Reduce a continuous-time Dataset to (DJF, MAM, JJA, SON) values.

    Wraps ``ds.groupby(f"{time}.season")`` + a reduction. Output dim is
    ``"season"`` with coord values
    ``["DJF", "MAM", "JJA", "SON"]`` (xarray's natural order).
    """
    return getattr(ds.groupby(f"{time}.season"), reduction)()
```

Trivial — exists mainly so users have a one-liner to feed into
`FacetPanel(facet_dim="season")`. Skip if you'd rather keep the surface
minimal; users compose `ds.groupby("time.season").mean()` directly.

### 4.5 No dedicated `SeasonalPSDScorePanel`

Explicit composition wins. The seasonal mosaic is
`FacetPanel(PSDSpaceTimeScorePanel(threshold=0.5), facet_dim="season",
ncols=2)`. Adding a one-shot subclass would duplicate logic and ossify
the layout choice.

## 5. Library leverage

| Need | Library |
|---|---|
| Subplot grid | `matplotlib.pyplot.subplots(nrows, ncols, sharex, sharey, subplot_kw)` |
| Cartopy GeoAxes | `subplot_kw={"projection": ccrs.<...>}` (matplotlib + cartopy) |
| Per-slice selection | `xarray.Dataset.isel({dim: i})` |
| Seasonal groupby | `xarray.Dataset.groupby("time.season")` |
| Layout default | `math.ceil`, `math.sqrt` |

No new dependencies.

## 6. Public API surface

```python
# Generic faceting wrapper
xr_toolz.viz.validation.FacetPanel(
    panel,                     # _ValidationPanel | Callable[(ds, ax), Any]
    *,
    facet_dim,
    ncols=None, nrows=None,
    sharex=True, sharey=True,
    sharebar=False,
    figsize_per_panel=(5, 4),
    title_format="{value}",
    subplot_kw=None,
)

# Optional helper (skip if you'd rather defer)
xr_toolz.viz.validation.seasonal_groupby(ds, *, time="time", reduction="mean")
```

Re-exported from `xr_toolz.viz.validation.__init__`.

## 7. Tests

| Test | Asserts |
|---|---|
| 4-element `experiment` dim → default layout | 2×2 grid (auto `ncols=2`) |
| Explicit `nrows=1, ncols=4` | 1×4 grid honoured |
| 5-element dim, default layout | 2×3 grid; last axis hidden |
| Per-axes title matches `coord[facet_dim].values[i]` | exact string |
| `title_format="{value} ({index})"` | format applied |
| Inner `_ValidationPanel._build` called once per slice with right `(fig, ax)` | exact call count + axes identity |
| Plain callable inner panel | invoked once per slice; no subclassing required |
| `SpatialMapPanel` inner with `projection="gulf_stream"` | grid built with cartopy GeoAxes |
| Explicit `subplot_kw=` overrides inner projection | user value used |
| `sharebar=True` with mappable-returning panel | single colorbar rendered |
| `sharebar=True` with no mappable | warning emitted; falls back to per-axes |
| `get_config` round-trip with `_ValidationPanel` inner | reconstructed FacetPanel produces identical figure |
| `get_config` with callable inner | flagged non-roundtrippable; doesn't crash |
| `seasonal_groupby` end-to-end | continuous time → 4-cell `season` dim → FacetPanel mosaic |

Target: ~14 cases.

## 8. Out of scope

- **Multi-dim faceting** (`row=` AND `col=` like xarray's `FacetGrid`)
  — single `facet_dim` only for v1; extend later if needed.
- **Domain-specific `SeasonalPSDScorePanel`** — declined.
- **Per-axis cmap/norm/clim overrides** — inner panel decides.
- **Heterogeneous panels per cell** — single inner panel, applied
  uniformly. Mixing panel types per cell is a different abstraction.
- **Plotly / hvplot / bokeh** — matplotlib only.
- **Plots that own the whole figure** (seaborn `JointGrid`, `pairplot`).

## 9. Effort

≈70 LOC implementation + ≈90 LOC tests. Single PR.

| Slice | LOC |
|---|---|
| `FacetPanel` | 60 |
| Optional `seasonal_groupby` | 10 |
| Tests | ~90 |
| Docs / re-exports | 10 |

## 10. Risks / open questions

1. **`sharebar` default.** `False` matches upstream and is robust;
   `True` is cleaner for visual comparison. **Default `False`**, opt
   in. With `sharebar=True` and a non-mappable inner panel, emit a
   warning and fall back rather than failing.
2. **Cartopy projection detection.** Reading `panel.projection`
   reaches into a non-standard attribute. Document the convention; an
   explicit `subplot_kw=` always wins.
3. **`title_format` default.** `"{value}"` is minimal. Users override
   for `"{value} ({index})"` or richer formatting.
4. **`get_config` for callable inner panels.** Emit a sentinel
   indicating non-roundtrippable; document. Don't fail at construction.
5. **Layout heuristic.** `ncols = ceil(sqrt(N))` produces 2×2 for N=4
   (matches upstream), 2×3 for N=5/6, 3×3 for N=7-9. Reasonable.
   Override always available.
6. **`figsize_per_panel` vs total `figsize`.** Per-panel chosen so
   adding facets scales the figure; matches xarray `FacetGrid`
   convention. Total override via `subplot_kw=` if needed (or post-hoc
   `fig.set_size_inches`).
