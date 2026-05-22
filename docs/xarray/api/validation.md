# Validation Panels

Terminal visualisation operators for V1–V5 metric outputs. Each panel
is an `Operator` subclass returning a `matplotlib.figure.Figure`, so
panels slot into `Sequential` and `Graph` pipelines as the last step.

## V1 — Scale & Spectral Skill

::: xrtoolz.viz.validation.LeadTimeSkillPanel

::: xrtoolz.viz.validation.ScaleSkillPanel

::: xrtoolz.viz.validation.SpectralSkillPanel

## V1.5 — PSD Plots

Power-spectrum visualisations consuming
[`xrtoolz.transforms.power_spectrum`](metrics.md#spectral) and
[`xrtoolz.metrics.psd_score`](metrics.md#spectral) outputs.

::: xrtoolz.viz.validation.PSDIsotropicPanel

::: xrtoolz.viz.validation.PSDIsotropicScorePanel

::: xrtoolz.viz.validation.PSDSpaceTimePanel

::: xrtoolz.viz.validation.PSDSpaceTimeScorePanel

## Region & Rotary diagnostics

`RegionScoreBarPanel` consumes region-stratified metric outputs (e.g.
[`xrtoolz.metrics.scores_by_region`](metrics.md)) and
`RotaryPolarizationPanel` consumes rotary-spectrum polarization fields
from [`xrtoolz.transforms.rotary_spectrum`](metrics.md).

::: xrtoolz.viz.validation.RegionScoreBarPanel

::: xrtoolz.viz.validation.RotaryPolarizationPanel

## Spatial snapshots

::: xrtoolz.viz.validation.SpatialMapPanel

### Cartopy presets

The `projection` kwarg of `SpatialMapPanel` accepts preset names from
[`xrtoolz.viz.PRESETS`][xrtoolz.viz.PRESETS] (`"global"`,
`"north_atlantic"`, `"gulf_stream"`, `"kuroshio"`,
`"mediterranean"`), a cartopy class name, or an instantiated cartopy
CRS. With a preset, the right `set_extent` is applied automatically.

::: xrtoolz.viz.make_axes

### Variable → colormap registry

Default colormaps for spatial panels are looked up from the curated
[`xrtoolz.types.REGISTRY`][xrtoolz.types.REGISTRY]: every entry
carries a `cmap` field (e.g. SSH → `RdBu_r`, SST → `RdYlBu_r`,
salinity → `viridis`, ice → `Blues`).
[`xrtoolz.viz.cmap_for`][xrtoolz.viz.cmap_for] performs the
lookup; `SpatialMapPanel(var=...)` calls it automatically when
`cmap` is unset.

::: xrtoolz.viz.cmap_for

## V3 — Lagrangian / Eulerian

::: xrtoolz.viz.validation.EulerianLagrangianPanel

## V4 — Process Budgets

::: xrtoolz.viz.validation.ProcessBudgetPanel

## V5 — Event Verification

::: xrtoolz.viz.validation.EventVerificationPanel
