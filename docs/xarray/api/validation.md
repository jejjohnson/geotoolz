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

## V3 — Lagrangian / Eulerian

::: xr_toolz.viz.validation.EulerianLagrangianPanel

## V4 — Process Budgets

::: xr_toolz.viz.validation.ProcessBudgetPanel

## V5 — Event Verification

::: xr_toolz.viz.validation.EventVerificationPanel
