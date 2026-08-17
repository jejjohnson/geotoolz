# Visualization

`xrtoolz.viz` provides plotting helpers and a family of **validation
panels** — `Operator` subclasses that return a `matplotlib.figure.Figure`,
so a panel slots into a `Sequential` or `Graph` pipeline as the final step.

## Axes & colormaps

`make_axes` builds a (optionally cartopy-projected) axes grid; `cmap_for`
looks up the default colormap for a variable; `shared_norm` derives a
common colour normalization across panels; `PRESETS` is the registry of
named cartopy extents.

::: xrtoolz.viz.make_axes

::: xrtoolz.viz.cmap_for

::: xrtoolz.viz.shared_norm

::: xrtoolz.viz.PRESETS

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

::: xrtoolz.viz.validation.LeadTimeSkillPanel

::: xrtoolz.viz.validation.ScaleSkillPanel

::: xrtoolz.viz.validation.SpectralSkillPanel

## V1.5 — PSD plots

Power-spectrum visualisations consuming `transforms.power_spectrum` and
`metrics.psd_score` outputs.

::: xrtoolz.viz.validation.PSDIsotropicPanel

::: xrtoolz.viz.validation.PSDIsotropicScorePanel

::: xrtoolz.viz.validation.PSDSpaceTimePanel

::: xrtoolz.viz.validation.PSDSpaceTimeScorePanel

## Region & rotary diagnostics

::: xrtoolz.viz.validation.RegionScoreBarPanel

::: xrtoolz.viz.validation.RotaryPolarizationPanel

## Spatial snapshots

::: xrtoolz.viz.validation.SpatialMapPanel

## V3 — Lagrangian / Eulerian

::: xrtoolz.viz.validation.EulerianLagrangianPanel

## V4 — Process budgets

::: xrtoolz.viz.validation.ProcessBudgetPanel

## V5 — Event verification

::: xrtoolz.viz.validation.EventVerificationPanel

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

::: xrtoolz.viz.validation.FacetPanel

::: xrtoolz.viz.validation.seasonal_groupby

::: xrtoolz.viz.validation.PairwiseComparePanel

::: xrtoolz.viz.validation.AnimatePanel

::: xrtoolz.viz.validation.save_animation

## Palette helper

::: xrtoolz.viz.validation.method_palette
