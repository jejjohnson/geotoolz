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

## Palette helper

::: xrtoolz.viz.validation.method_palette
