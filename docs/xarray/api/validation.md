# Validation Panels

Terminal visualisation operators for V1–V5 metric outputs. Each panel
is an `Operator` subclass returning a `matplotlib.figure.Figure`, so
panels slot into `Sequential` and `Graph` pipelines as the last step.

## V1 — Scale & Spectral Skill

::: xr_toolz.viz.validation.LeadTimeSkillPanel

::: xr_toolz.viz.validation.ScaleSkillPanel

::: xr_toolz.viz.validation.SpectralSkillPanel

## V1.5 — PSD Plots

Power-spectrum visualisations consuming
[`xr_toolz.transforms.power_spectrum`](metrics.md#spectral) and
[`xr_toolz.metrics.psd_score`](metrics.md#spectral) outputs.

::: xr_toolz.viz.validation.PSDIsotropicPanel

::: xr_toolz.viz.validation.PSDIsotropicScorePanel

::: xr_toolz.viz.validation.PSDSpaceTimePanel

::: xr_toolz.viz.validation.PSDSpaceTimeScorePanel

## Spatial snapshots

::: xr_toolz.viz.validation.SpatialMapPanel

### Cartopy presets

The `projection` kwarg of `SpatialMapPanel` accepts preset names from
[`xr_toolz.viz.PRESETS`][xr_toolz.viz.PRESETS] (`"global"`,
`"north_atlantic"`, `"gulf_stream"`, `"kuroshio"`,
`"mediterranean"`), a cartopy class name, or an instantiated cartopy
CRS. With a preset, the right `set_extent` is applied automatically.

::: xr_toolz.viz.make_axes

### Variable → colormap registry

Default colormaps for spatial panels are looked up from the curated
[`xr_toolz.types.REGISTRY`][xr_toolz.types.REGISTRY]: every entry
carries a `cmap` field (e.g. SSH → `RdBu_r`, SST → `RdYlBu_r`,
salinity → `viridis`, ice → `Blues`).
[`xr_toolz.viz.cmap_for`][xr_toolz.viz.cmap_for] performs the
lookup; `SpatialMapPanel(var=...)` calls it automatically when
`cmap` is unset.

::: xr_toolz.viz.cmap_for

## V3 — Lagrangian / Eulerian

::: xr_toolz.viz.validation.EulerianLagrangianPanel

## V4 — Process Budgets

::: xr_toolz.viz.validation.ProcessBudgetPanel

## V5 — Event Verification

::: xr_toolz.viz.validation.EventVerificationPanel
