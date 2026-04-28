---
status: draft
version: 0.1.0
---

!!! note "Imports in this page are from the original `geo_toolz` layout"
    These design docs were adapted from the `geo_toolz` design study.
    Code snippets use the original feature-based import paths
    (`geo_toolz.<module>`). In `xr_toolz`, the domain-agnostic
    operations live under `xr_toolz.geo.<module>`; physics-specific
    operations live under `xr_toolz.ocn` / `xr_toolz.atm` /
    `xr_toolz.rs`. See `xr_toolz/__init__.py` and
    `xr_toolz/geo/__init__.py` for the current export surface.

# API Overview

Each submodule provides Layer 0 pure functions and Layer 1 Operator wrappers. Layer 2 provides the Graph API and inference wrappers.

---

## Surface Inventory

### Core Infrastructure

| Export | Module | Layer | Status |
|---|---|---|---|
| `Operator` | `geo_toolz.core` | Base | v0.1 |
| `Sequential` | `geo_toolz.core` | L1 | v0.1 |
| `Input`, `Node`, `Graph` | `geo_toolz.core` | L2 | v0.1 |
| `ModelOp` | `geo_toolz.inference` | L2 | v0.2 |
| `SklearnModelOp` | `geo_toolz.inference` | L2 | v0.2 |
| `JaxModelOp` | `geo_toolz.inference` | L2 | v0.2 |

### Submodule Inventory

| Submodule | Purpose | L0 Functions | L1 Operators | Status |
|-----------|---------|-------------|-------------|--------|
| `validation` | Coordinate harmonization | `validate_longitude`, `validate_latitude`, ... | `ValidateCoords`, `HarmonizeCoords` | v0.1 |
| `subset` | Spatial/temporal selection | `subset_bbox`, `subset_time`, ... | `SubsetBBox`, `SubsetTime`, `SubsetGeometry` | v0.1 |
| `masks` | Land/ocean/country masks | `create_land_mask`, `apply_mask`, ... | `AddOceanMask`, `AddLandMask`, `AddCountryMask` | v0.1 |
| `regrid` | Grid transformations | `regrid_linear`, `regrid_nearest`, ... | `Regrid` | v0.1 |
| `detrend` | Climatology and anomalies | `calculate_climatology`, `remove_climatology`, ... | `CalculateClimatology`, `RemoveClimatology` | v0.1 |
| `metrics` | Evaluation metrics | `rmse`, `nrmse`, `mae`, `bias`, `correlation` | `RMSE`, `NRMSE`, `MAE`, `Bias`, `PSDScore` | v0.1 |
| `interpolation` | Gap filling, resampling | `fillnan_spatial`, `fillnan_temporal`, ... | `FillNaN`, `Resample` | v0.2 |
| `encoders` | Coordinate encodings | `cyclical_encode`, `fourier_features`, ... | `CyclicalEncoder`, `FourierFeatures` | v0.2 |
| `discretize` | Binning, points-to-grid | `bin_spatial`, `points_to_grid`, ... | `Discretize`, `Coarsen` | v0.2 |
| `spectral` | PSD, cross-spectrum | `psd`, `isotropic_psd`, `cross_spectrum` | `PSD`, `IsotropicPSD` | v0.2 |
| `extremes` | Extreme value analysis | `block_maxima`, `peaks_over_threshold`, ... | `BlockMaxima`, `POT` | v0.2 |
| `kinematics` | Physical quantities | `streamfunction`, `geostrophic_velocity`, ... | `Streamfunction`, `GeostrophicVelocity`, ... | v0.3 |
| `crs` | CRS transforms | `assign_crs`, `reproject`, ... | `AssignCRS`, `Reproject` | v0.3 |
| `sklearn` | sklearn interop | `to_2d`, `from_2d` | `SklearnOp` | v0.3 |
| `metrics.structural` | Structural/geometric skill metrics | `ssim`, `centroid_displacement`, `phase_shift_error` | `SSIM`, `CentroidDisplacement`, `PhaseShiftError` | v0.3 |
| `metrics.forecast` | Lead-time and forecast-horizon diagnostics | `skill_by_lead_time`, `rmse_by_lead`, `acc_by_lead` | `SkillByLeadTime`, `RMSEByLead` | v0.3 |
| `metrics.probabilistic` | Ensemble and probabilistic forecast verification | `crps`, `spread_skill_ratio`, `rank_histogram`, `ensemble_coverage` | `CRPS`, `SpreadSkillRatio`, `RankHistogram` | v0.3 |
| `metrics.physical` | Physical consistency scores | `geostrophic_balance_error`, `divergence_error`, `vorticity_error` | `GeostrophicBalanceError`, `DivergenceError` | v0.3 |
| `lagrangian` | Particle and transport diagnostics | `advect_particles`, `pair_dispersion`, `residence_time`, `ftle` | `AdvectParticles`, `PairDispersion`, `FTLE` | v0.4 |
| `budgets` | Control-volume and conservation diagnostics | `budget_residual`, `heat_budget_residual`, `salt_budget_residual` | `BudgetResidual`, `HeatBudgetResidual` | v0.4 |
| `phenomena` | Object/event detection and matching | `detect_marine_heatwaves`, `detect_eddies`, `match_objects` | `DetectMarineHeatwaves`, `DetectEddies`, `MatchObjects` | v0.4 |
| `metrics.object` | Event/object verification scores | `contingency_table`, `pod`, `far`, `csi`, `iou` | `ProbabilityOfDetection`, `FalseAlarmRatio`, `CriticalSuccessIndex`, `IntersectionOverUnion` | v0.4 |

**Status key:** `v0.1` = Foundation | `v0.2` = Analysis + Inference | `v0.3` = Domain Operators + expanded metrics | `v0.4+` = Validation expansion

---

## Notation and Conventions

### Type Aliases

| Alias | Meaning | Used in |
|---|---|---|
| `Dataset` | `xr.Dataset` | All operators |
| `DataArray` | `xr.DataArray` | Metrics, spectral |

### Operator Contract

Every `Operator` subclass satisfies:
- `__call__(*args, **kwargs)` — execute (single-input: `Dataset → Dataset`, multi-input: `(Dataset, Dataset) → DataArray`)
- `get_config() → dict` — JSON-serializable constructor arguments
- `__repr__()` — human-readable `ClassName(param=value, ...)`
- `__or__(other) → Sequential` — pipe syntax

### Import Conventions

```python
# Core
from geo_toolz.core import Sequential, Graph, Input

# Submodule operators (Layer 1)
from geo_toolz.validation import ValidateCoords
from geo_toolz.regrid import Regrid
from geo_toolz.detrend import CalculateClimatology, RemoveClimatology
from geo_toolz.metrics import RMSE, PSDScore

# Expanded validation operators
from geo_toolz.metrics.forecast import RMSEByLead
from geo_toolz.metrics.structural import SSIM
from geo_toolz.lagrangian import AdvectParticles
from geo_toolz.budgets import HeatBudgetResidual
from geo_toolz.phenomena import DetectMarineHeatwaves

# Inference
from geo_toolz.inference import ModelOp, SklearnModelOp, JaxModelOp

# Layer 0 functions (direct access)
from geo_toolz._src.detrend.climatology import calculate_climatology, remove_climatology
```

---

## Detail Files

| File | Covers |
|---|---|
| [primitives.md](primitives.md) | Layer 0 — pure functions by submodule (validation, subset, regrid, detrend, ...) |
| [components.md](components.md) | Layer 1 — Operator classes by submodule, Operator base class |
| [models.md](models.md) | Layer 2 — Graph API (Node, Input, Graph), ModelOp, inference wrappers |
| [validation.md](validation.md) | Validation API map by scientific question and metric family |

---

*For usage patterns, see [../examples/](../examples/) — organized by layer to match this directory.*
