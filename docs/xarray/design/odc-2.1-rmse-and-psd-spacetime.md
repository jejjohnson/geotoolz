# ODC-2.1 — RMSE-skill bundle + space-time PSD-score with (λx, λt) summary

**Source survey item:** [ocean-data-challenges-survey.md §2.1](ocean-data-challenges-survey.md)
**Status:** proposed
**Maps to upstream:** `src/mod_eval.py` from `2023_SSH_mapping_train_eNATL60_test_NATL60-`.

---

## 1. Motivation

The eNATL60 → NATL60 OSSE evaluation in
`2023_SSH_mapping_train_eNATL60_test_NATL60-` ships the cleanest, most
self-contained version of the canonical SSH-mapping skill report:

1. **`rmse_based_scores`** — four quantities from a single call:
   `rmse_t` (per-time skill score over space), `rmse_xy` (per-cell
   RMSE map over time), `leaderboard_rmse` (scalar), and an
   `error_stability = std(rmse_t)` metric we currently lack.
2. **`psd_based_scores`** — the **2-D space-time PSD score** in
   `(freq_lon, freq_time)`, plus `(λx_min, λt_min)` extracted from the
   level=0.5 contour.

xrtoolz already exposes virtually every primitive these need. What's
missing is:

- A 2-D analog of our 1-D `resolved_scale` that reduces the existing
  `find_intercept_2D` polylines to summary wavelengths.
- A bundled-output convenience for the four RMSE-skill scores
  (notably the standalone `error_stability` reduction).
- A space-time PSD-score driver that wires `psd_score` + level-contour
  reduction together for the most-common ocean-data-challenges call
  shape.

This issue ships those three thin wrappers — almost no new code, lots
of composition.

## 2. User stories

### 2.1 RMSE-skill report (primary)

> *I have a reconstruction `ds_rec(ssh: time, lat, lon)` and a reference
> `ds_ref(ssh: time, lat, lon)`. I want the four standard SSH-mapping
> RMSE-based skill scores in one call.*

```python
import xarray as xr
from xrtoolz.metrics import rmse_skill_scores

ds = rmse_skill_scores(ds_rec, ds_ref, variable="ssh")
# ds has data_vars:
#   rmse_t (time,)            — 1 - RMSE_xy / RMS_ref_xy per time step
#   rmse_xy (lat, lon)        — RMSE over time per cell
#   leaderboard_rmse  scalar  — 1 - RMSE_total / RMS_ref_total
#   error_stability   scalar  — std(rmse_t)
```

### 2.2 Space-time PSD score with (λx, λt) summary

> *I want the canonical 2-D PSD-score map in `(freq_lon, freq_time)`
> averaged over `lat`, plus the shortest space and time wavelengths
> jointly resolved at score = 0.5.*

```python
from xrtoolz.metrics import psd_score_spacetime

score, summary = psd_score_spacetime(
    ds_rec, ds_ref,
    variable="ssh",
    space_dim="lon", time_dim="time",
    avg_dims=("lat",),
    level=0.5,
)
# score: Dataset over (freq_lon, freq_time), positive freqs only.
# summary: {"lambda_space_min": ..., "lambda_time_min": ...,
#           "lambda_space_max": ..., "lambda_time_max": ...}
```

### 2.3 Plug score map into the existing `PSDSpaceTimeScorePanel`

```python
from xrtoolz.viz.validation import PSDSpaceTimeScorePanel

panel = PSDSpaceTimeScorePanel(...)
fig, axes = panel(score)
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| `1 − RMSE / RMS_ref` primitive | [`_pixel_kernels.nrmse`](../../src/xrtoolz/metrics/_src/_pixel_kernels.py) — exactly the upstream formula | reuse |
| `rmse`, `mse`, `bias`, `correlation`, `r2_score` | [`_pixel_kernels`](../../src/xrtoolz/metrics/_src/_pixel_kernels.py) | reuse |
| 2-D `psd_score(ds_pred, ds_ref, variable, psd_dims, avg_dims)` | [`spectral.py:92`](../../src/xrtoolz/metrics/_src/spectral.py) | reuse |
| 1-D `resolved_scale` | [`spectral.py:131`](../../src/xrtoolz/metrics/_src/spectral.py) | reuse |
| 2-D `find_intercept_2D` (skimage `find_contours`) | [`spectral.py:207`](../../src/xrtoolz/metrics/_src/spectral.py) | reuse |
| Space-time score viz | [`viz/validation/_src/psd.py` `PSDSpaceTimeScorePanel`](../../src/xrtoolz/viz/validation/_src/psd.py) | reuse |
| 2-D `resolved_scale_2d` summary | — | **add** |
| RMSE-skill bundle | — | **add** `rmse_skill_scores` |
| Space-time PSD-score driver | — | **add** `psd_score_spacetime` |

## 4. Design

### 4.1 `resolved_scale_2d` — summary along the level contour

```python
# src/xrtoolz/metrics/_src/spectral.py  (alongside resolved_scale)
def resolved_scale_2d(
    score: xr.DataArray | xr.Dataset, *,
    level: float = 0.5,
    space_dim: str = "freq_lon",
    time_dim: str = "freq_time",
) -> dict[str, float]:
    """Reduce the level=`level` contour of a 2-D PSD-score field to
    summary wavelengths.

    Calls :func:`find_intercept_2D`, merges all segments, converts
    frequencies to wavelengths, and returns::

        {
            "lambda_space_min": min(1 / freq_lon along contour),
            "lambda_time_min":  min(1 / freq_time along contour),
            "lambda_space_max": max(...),
            "lambda_time_max":  max(...),
        }

    Returns NaN entries if no contour exists at ``level``.
    """
```

~25 LOC. Implementation:

1. Resolve `score` to a DataArray (accept Dataset with `"score"` var,
   matching `resolved_scale`'s pattern).
2. `segments = find_intercept_2D(score, level, space_dim, time_dim)`.
3. If empty: return `{k: float("nan") for k in keys}`.
4. Concatenate all segments along the point dim.
5. `1 / freq_*` for the wavelength axes; reduce min/max.

### 4.2 `rmse_skill_scores` — RMSE-skill bundle

```python
# src/xrtoolz/metrics/_src/composite.py — new module
def rmse_skill_scores(
    ds_pred: xr.Dataset, ds_ref: xr.Dataset, *,
    variable: str,
    space_dims: Sequence[str] = ("lat", "lon"),
    time_dim: str = "time",
) -> xr.Dataset:
    """Bundle of RMSE-based skill scores for a single variable.

    Returns Dataset with:

    - ``rmse_t``           dim ``(time,)``        — ``1 − RMSE_xy / RMS_ref_xy``
    - ``rmse_xy``          dims ``space_dims``     — ``RMSE`` over time per cell
    - ``leaderboard_rmse`` scalar                  — ``1 − RMSE_total / RMS_ref_total``
    - ``error_stability``  scalar                  — ``std(rmse_t)``
    """
```

~25 LOC composing existing primitives. The four entries become
tagged data_vars; downstream code can consume the Dataset directly or
project to scalars.

### 4.3 `psd_score_spacetime` — space-time score + summary

```python
def psd_score_spacetime(
    ds_pred: xr.Dataset, ds_ref: xr.Dataset, *,
    variable: str,
    space_dim: str = "lon",
    time_dim: str = "time",
    avg_dims: Sequence[str] | None = ("lat",),
    level: float = 0.5,
    **xrft_kwargs: Any,
) -> tuple[xr.Dataset, dict[str, float]]:
    """2-D space-time PSD score plus level-contour wavelength summary.

    Returns:
      - score: 2-D Dataset over ``(freq_<space>, freq_<time>)``,
        positive frequencies only.
      - summary: dict of {lambda_space_min, lambda_time_min,
                          lambda_space_max, lambda_time_max}.
    """
```

Internals:

1. `score = psd_score(ds_pred, ds_ref, variable,
                      psd_dims=(time_dim, space_dim),
                      avg_dims=avg_dims, **xrft_kwargs)`.
2. Drop negative-frequency half on both axes.
3. `summary = resolved_scale_2d(score, level=level,
                                space_dim=f"freq_{space_dim}",
                                time_dim=f"freq_{time_dim}")`.
4. Return `(score, summary)`.

~25 LOC.

### 4.4 Optional Operators

```python
class RMSESkillScores(Operator):    # two-input
class PSDSpaceTimeScore(Operator):  # two-input
```

Skipped for v1 — primitives and functions are enough; promote later if
a Sequential needs them.

## 5. Library leverage

| Need | Library |
|---|---|
| FFT for 2-D PSD | `xrft` (already used by `power_spectrum`) |
| 2-D contour extraction | `skimage.measure.find_contours` (already a dep, wrapped via `find_intercept_2D`) |
| RMSE / MSE / bias | existing `_pixel_kernels` / `pixel` |
| Std of a DataArray | xarray built-in |

No new dependencies.

## 6. Public API surface

```python
xrtoolz.metrics.rmse_skill_scores(ds_pred, ds_ref, *, variable, space_dims, time_dim)
xrtoolz.metrics.resolved_scale_2d(score, *, level, space_dim, time_dim)
xrtoolz.metrics.psd_score_spacetime(ds_pred, ds_ref, *, variable, space_dim,
                                     time_dim, avg_dims, level, **xrft_kwargs)
```

All re-exported from `xrtoolz.metrics.__init__`.

## 7. Tests

| Test | Asserts |
|---|---|
| `nrmse` against hand-computed `1 − RMSE / RMS_ref` | exact |
| `rmse_skill_scores` shape & dim layout | `rmse_t` over time, `rmse_xy` over space, two scalars |
| `rmse_skill_scores` identical fields | `rmse_t == 1`, `error_stability ≈ 0`, `rmse_xy ≈ 0`, `leaderboard ≈ 1` |
| `rmse_skill_scores` random fields | finite values; `error_stability >= 0` |
| `resolved_scale_2d` on synthetic plateau-then-drop field | min/max match analytic contour vertices |
| `resolved_scale_2d` no contour | all four entries NaN |
| `resolved_scale_2d` disconnected segments | merged before reduction |
| `psd_score_spacetime` end-to-end on synthetic noisy field | summary keys present and finite; score in (0, 1) |
| `psd_score_spacetime` reproduces upstream λx_min, λt_min on a fixture | match to 3 sig figs |

Target: ~9 cases.

## 8. Out of scope

- Operator promotion (`RMSESkillScores`, `PSDSpaceTimeScore`).
- New plot panel — `PSDSpaceTimeScorePanel` already exists.
- 3-D contour extraction.
- Non-uniform-grid PSD.
- Per-region variants — orthogonal to ODC-1.4's `scores_by_region`.

## 9. Effort

≈75 LOC implementation + ≈85 LOC tests. Single PR.

| Slice | LOC |
|---|---|
| `resolved_scale_2d` | 25 |
| `rmse_skill_scores` | 25 |
| `psd_score_spacetime` | 25 |
| Tests | ~85 |
| Docs / re-exports | 10 |

## 10. Risks / open questions

1. **Where the new code lives.**
   - `resolved_scale_2d` → [`spectral.py`](../../src/xrtoolz/metrics/_src/spectral.py)
     alongside `resolved_scale` / `find_intercept_2D`.
   - Bundles → new `metrics/_src/composite.py` (cross-cuts RMSE + PSD).
2. **Return type of `psd_score_spacetime`.** Tuple `(score, summary)`
   chosen over single Dataset with `.attrs` to avoid attrs pollution
   and to make summary explicit.
3. **`error_stability` semantics.** `std(rmse_t)` — but ddof=0 vs 1
   matters. Default to numpy/xarray default (ddof=0); document the
   choice. Match upstream behaviour.
4. **Variable-name flexibility.** Upstream hard-codes `"ssh"`; our
   `variable=` kwarg makes it work for any field.
5. **Frequency-coord naming.** `xrft` outputs `freq_<dim>`. We thread
   that through; users naming their dims something else need to pass
   `space_dim=`/`time_dim=` explicitly.
