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

# Boundaries and Ecosystem

## Overview

geo_toolz owns the xarray geoprocessing operator layer: preprocessing, inference wrapping, and evaluation. It delegates compute to numpy/scipy/sklearn, framework-specific inference to user-installed backends, and domain modeling to downstream packages.

---

## Ownership Map

| Concern | Owner | Notes |
|---------|-------|-------|
| Coordinate validation, harmonization | **geo_toolz** | `validation` submodule |
| Spatial/temporal subsetting | **geo_toolz** | `subset` submodule |
| Regridding (scipy-based) | **geo_toolz** | No ESMF dependency |
| Land/ocean/country masks | **geo_toolz** | Via regionmask |
| Climatology, anomalies, detrending | **geo_toolz** | `detrend` submodule |
| Spectral analysis (PSD, cross-spectrum) | **geo_toolz** | Via xrft |
| Evaluation metrics (RMSE, PSD score, etc.) | **geo_toolz** | Via xskillscore + custom |
| Kinematics (oceanography, remote sensing) | **geo_toolz** | `kinematics` submodule |
| Operator/Sequential/Graph abstraction | **geo_toolz** | `core` submodule |
| Model wrapping (ModelOp) | **geo_toolz** | Framework-agnostic via duck typing |
| sklearn xarray bridge | **xarray_sklearn** | Optional companion; `SklearnOp` can delegate |
| Patch-wise tiling | **xrpatcher** | Optional companion |
| Data assimilation (EnKF, 4DVar) | **ekalmX** / **vardax** | Consumes geo_toolz for pre/post-processing |
| GP models | **pyrox_gp** | Independent; may consume geo_toolz features |
| Ocean/atmosphere forward models | **somax** / **diffrax** | Independent |
| Structured linear algebra | **gaussx** | Independent |

---

## Decision Table

| Scenario | Recommendation |
|----------|---------------|
| Preprocess satellite data for ML | `Sequential([ValidateCoords(), Regrid(...), RemoveClimatology(...)])` |
| Run a trained sklearn model on xarray | `SklearnModelOp(model, sample_dim="time")` in a Sequential |
| Run a JAX model on xarray | `JaxModelOp(model, sample_dim="time", jit=True)` |
| Evaluate predictions vs reference | `RMSE(...)`, `PSDScore(...)` — or Graph for multi-metric |
| Multi-input pipeline (preprocess + evaluate) | `Graph(inputs=..., outputs=...)` |
| Patch-wise inference on large grids | `xrpatcher` + `Sequential` per patch |
| Hydra-configurable pipeline | `hydra_zen.builds(Sequential, [...])` |
| Conservative regridding (ESMF) | Use xesmf directly — not in geo_toolz scope |
| Distributed/dask processing | Use xarray+dask directly — deferred in geo_toolz |

---

## Ecosystem Interactions

| External Package | Integration Point | Pattern |
|---|---|---|
| **xarray** | Interface layer for all operators | `Dataset → Dataset` everywhere |
| **scipy** | Regridding, interpolation, spectral | Compute backend for L0 functions |
| **sklearn** | Preprocessing transforms, models | Via `SklearnOp` / `SklearnModelOp` |
| **regionmask** | Land/ocean/country masks | `masks` submodule wraps regionmask |
| **xrft** | Fourier transforms | `spectral` submodule wraps xrft |
| **xskillscore** | Verification metrics | `metrics` submodule wraps xskillscore |
| **hydra-zen** | Config generation from operators | `get_config()` + `builds()` |
| **xarray_sklearn** | Full sklearn bridge with NaN policies | Optional, `SklearnOp` can delegate |
| **xrpatcher** | Patch-wise tiling for large grids | Optional, operators applied per patch |
| **ekalmX** | DA pre/post-processing | geo_toolz upstream (preprocess) and downstream (evaluate) |

---

## Scope

### In Scope

- Coordinate validation and harmonization
- Spatial/temporal subsetting and masking
- Regridding (scipy-based: linear, nearest, coarsen)
- Climatology, anomalies, detrending, filtering
- Coordinate encodings (cyclical, Fourier features)
- Binning and gridding (unstructured → gridded)
- Spectral analysis (PSD, cross-spectrum, coherence)
- Evaluation metrics (pixel-level, spectral, multiscale)
- Physical kinematics (oceanography, remote sensing, atmospheric)
- Operator/Sequential/Graph pipeline abstraction
- Framework-agnostic model wrapping (ModelOp)
- sklearn interop (SklearnOp)

### Out of Scope

- Data assimilation algorithms — ekalmX / vardax
- GP models, kernel learning — pyrox_gp
- Forward models (ocean, atmosphere) — somax / diffrax
- Structured linear algebra — gaussx
- Conservative regridding (ESMF) — xesmf
- Distributed computation (dask-native) — deferred to v0.4+
- Non-Earth geospatial (urban, indoor, planetary) — out of scope

---

## Testing Strategy

### Fixtures

```python
# conftest.py

@pytest.fixture
def ds_global():
    """Global 1° dataset: (time=100, lat=180, lon=360), with coords and attrs."""

@pytest.fixture
def ds_regional():
    """Mediterranean subset: (time=50, lat=40, lon=75)."""

@pytest.fixture
def ds_unstructured():
    """Scattered observations: (obs=5000) with lon, lat, time as 1D coords."""

@pytest.fixture
def ds_with_nans(ds_global):
    """ds_global with ~15% NaN (land mask pattern)."""
```

### Test Categories

- **Operator contract tests**: every operator subclass must return the right type, preserve attrs when appropriate, and have a valid `get_config()` that round-trips through `__repr__`.
- **Numerical correctness tests**: for each Layer 0 function, compare against a known analytical result or a reference implementation (scipy, xskillscore, metpy).
- **Pipeline integration tests**: build realistic multi-step pipelines and verify they run end-to-end without error.
- **Round-trip tests**: `AddClimatology(clim)(RemoveClimatology(clim)(ds))` should recover the original dataset to float precision.
- **Edge cases**: empty datasets, single-point grids, all-NaN slices, non-standard CRS, duplicate time coordinates.

### Test Priorities

1. **Operator contract** — every operator returns the right type, preserves attrs, has valid `get_config()`
2. **Numerical correctness** — L0 functions match scipy/xskillscore reference implementations
3. **Pipeline integration** — realistic multi-step pipelines run end-to-end
4. **Round-trip** — `AddClimatology(clim)(RemoveClimatology(clim)(ds))` recovers original

---

## Roadmap

### v0.1 — Foundation (target: 4-6 weeks)

- `core` (Operator, Sequential, Input, Node, Graph, Lambda, Identity)
- `validation` (full)
- `subset` (full)
- `masks` (full, wrapping regionmask)
- `detrend` (climatology, anomalies, basic filter)
- `regrid` (linear, nearest, coarsen via scipy)
- `metrics` (RMSE, NRMSE, MAE, Bias, Correlation)
- Package scaffolding from `pypackage_template`
- Tests for all of the above
- Basic mkdocs site with API reference

### v0.2 — Analysis Depth + Inference (target: +4 weeks)

- `interpolation` (fillnan spatial/temporal, RBF, resample)
- `discretize` (binning, points-to-grid)
- `encoders` (coordinate transforms, cyclical, Fourier features)
- `spectral` (PSD, isotropic PSD, cross-spectrum)
- `metrics` (PSD score, resolved scale, multiscale RMSE)
- `extremes` (block maxima, POT, point process)
- `inference` (`ModelOp` base, `SklearnModelOp`, `JaxModelOp`, batch support)
- End-to-end examples: preprocess → infer → evaluate pipelines
- Tutorial notebooks

### v0.3 — Domain Operators (target: +4 weeks)

- `kinematics` oceanography (streamfunction, geostrophic velocity, vorticity, Okubo-Weiss, KE, MLD)
- `kinematics` remote sensing (NDVI, radiance/reflectance, brightness temperature)
- `kinematics` methane (column averaging kernel, dry air column, mixing ratio)
- `kinematics` atmospheric (potential temperature, wind speed/direction)
- `crs` (assign, reproject via rioxarray)
- `sklearn` (xarray wrapper via `xarray_sklearn`; optional `xrpatcher` support for patch-wise ML)

### v0.4+ — Continuous Expansion

- Dask integration for chunked computation
- Deeper `xrpatcher` integration (auto-reconstruct, parallel dispatch)
- Additional regridding methods (conservative approximation)
- Quality control operators (outlier detection, observation count thresholds)
- More metrics from xskillscore
- More kinematics as needed
- Numba/JAX kernels for performance-critical operations
- xarray accessor: `ds.geo.validate().regrid(grid).detrend(clim)`


## Open Questions

1. **xarray accessor?** A `ds.geo.validate()` accessor would be convenient but adds coupling. Defer to v0.4+ or include from v0.1?

2. **Dask awareness in v0.1?** Should operators check for dask-backed arrays and call `.compute()` / `.persist()` at appropriate points? Or leave dask entirely to the user? The safest v0.1 approach is to document that operators work on in-memory arrays and dask support comes later.

3. **Unit handling.** Some kinematics operations (streamfunction, geostrophic velocities) benefit from `pint-xarray` unit tracking. Should this be opt-in via a flag, or always-on for kinematics operators?

4. **Grid as an Operator?** The `Grid` / `SpaceTimeGrid` dataclasses in `discretize` could themselves be operators that generate empty coordinate scaffolds. Is `CreateGrid(lon_bnds, lat_bnds, resolution)` useful, or is the dataclass sufficient?

5. **Graph serialization.** The `Graph` can serialize its topology (which operators connect to which) via `get_config()`. But if someone wants to save a full graph to YAML and reload it (including the operator configs), that requires a class registry to map class names back to classes. Should `geo_toolz` include a lightweight registry (`register_operator` / `get_operator_by_name`), or leave that to Hydra-zen which already handles it?
