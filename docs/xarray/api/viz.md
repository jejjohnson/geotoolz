# Visualization

`xrops.viz` provides plotting helpers and a family of **validation
panels** — `Operator` subclasses that return a `matplotlib.figure.Figure`,
so a panel slots into a `Sequential` or `Graph` pipeline as the final step.

## Axes & colormaps

`make_axes` builds a (optionally cartopy-projected) axes grid; `cmap_for`
looks up the default colormap for a variable; `shared_norm` derives a
common colour normalization across panels; `PRESETS` is the registry of
named cartopy extents.

::: xrops.viz.make_axes

::: xrops.viz.cmap_for

::: xrops.viz.shared_norm

::: xrops.viz.PRESETS

!!! note "Cartopy presets"
    The `projection` kwarg of `SpatialMapPanel` (and `make_axes`) accepts a
    preset name from `PRESETS` (`"global"`, `"north_atlantic"`,
    `"gulf_stream"`, `"kuroshio"`, `"mediterranean"`), a cartopy class name,
    or an instantiated cartopy CRS. With a preset, the right `set_extent` is
    applied automatically.

!!! note "Variable → colormap registry"
    Default colormaps are looked up from the curated `Variable` registry
    (SSH → `RdBu_r`, SST → `RdYlBu_r`, salinity → `viridis`, ice →
    `Blues`). `cmap_for` performs the lookup; `SpatialMapPanel(var=…)` calls
    it automatically when `cmap` is unset.

## V1 — Scale & spectral skill

::: xrops.viz.validation.LeadTimeSkillPanel

::: xrops.viz.validation.ScaleSkillPanel

::: xrops.viz.validation.SpectralSkillPanel

::: xrops.viz.validation.TimeSeriesErrorPanel

## V1.5 — PSD plots

Power-spectrum visualisations consuming `transforms.power_spectrum` and
`metrics.psd_score` outputs.

::: xrops.viz.validation.PSDIsotropicPanel

::: xrops.viz.validation.PSDIsotropicScorePanel

::: xrops.viz.validation.PSDSpaceTimePanel

::: xrops.viz.validation.PSDSpaceTimeScorePanel

## Region & rotary diagnostics

::: xrops.viz.validation.RegionScoreBarPanel

::: xrops.viz.validation.RotaryPolarizationPanel

## Spatial snapshots

::: xrops.viz.validation.SpatialMapPanel

::: xrops.viz.validation.HovmollerPanel

### Cartesian model output

Idealised f/β-plane runs have `x`/`y` in metres and no lon/lat.
`CartesianMapPanel` draws them on equal-aspect km axes with a zero-centred
1/99 % colour scale and no cartopy; `EnergyTimeSeriesPanel` tracks model
invariants over time. Both compose with the wrappers below:

```python
from xrops.viz.validation import AnimatePanel, CartesianMapPanel, EnergyTimeSeriesPanel

CartesianMapPanel(variable="zeta", contour="psi")(snapshots)
AnimatePanel(CartesianMapPanel(variable="zeta"))(snapshots)  # FuncAnimation
EnergyTimeSeriesPanel(["kinetic_energy", "enstrophy"], relative=True)(invariants)
```

::: xrops.viz.validation.CartesianMapPanel

::: xrops.viz.validation.EnergyTimeSeriesPanel

## V3 — Lagrangian / Eulerian

::: xrops.viz.validation.EulerianLagrangianPanel

## V4 — Process budgets

::: xrops.viz.validation.ProcessBudgetPanel

## V5 — Event verification

::: xrops.viz.validation.EventVerificationPanel

## Composable wrappers

These consume any of the panels above — or a plain
`(ds, ax) -> Any` callable — and multiply it. They nest, so a
per-experiment N-up movie is
`AnimatePanel(FacetPanel(SpatialMapPanel(...), facet_dim="experiment"))`
and the six-panel `scale × (ref, study, diff)` mosaic is
`FacetPanel(PairwiseComparePanel(SpatialMapPanel(...)), facet_dim="scale")`.

!!! note "Generality boundary"
    Only **single-input, single-axes** panels can be wrapped — the
    wrappers hand an inner panel one sliced object and one Axes. That
    covers `SpatialMapPanel`, the PSD panels, `RegionScoreBarPanel`,
    `RotaryPolarizationPanel` and the scale-skill panels, plus any
    `(ds, ax) -> Any` callable.

    `EulerianLagrangianPanel` (eulerian + trajectories) and
    `EventVerificationPanel` (four inputs across an axes pair) take more
    of both and are rejected at construction with a `TypeError`. Plots
    that own the whole figure (seaborn `JointGrid`, `pairplot`) and
    non-matplotlib backends (plotly, hvplot, bokeh) are also out of
    scope.

::: xrops.viz.validation.FacetPanel

::: xrops.viz.validation.seasonal_groupby

::: xrops.viz.validation.PairwiseComparePanel

::: xrops.viz.validation.SpatialDiffPanel

::: xrops.viz.validation.AnimatePanel

::: xrops.viz.validation.save_animation

## Palette helper

::: xrops.viz.validation.method_palette
