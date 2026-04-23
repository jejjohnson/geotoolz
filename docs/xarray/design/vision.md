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

# Vision

## One-Liner

> **geo_toolz** is a composable operator library for geoprocessing Earth System Data Cubes — preprocess, infer, and evaluate xarray datasets with a uniform pipeline abstraction.

---

## Motivation

Most Earth science data arrives messy. Satellite swaths come in irregular orbits. Reanalysis products use different grids, different variable names, different conventions for longitude. Model output needs detrending before comparison. Metrics require careful spatial and spectral decomposition. Every applied ML researcher in climate, oceanography, or remote sensing has written these preprocessing pipelines from scratch, multiple times, because existing tools are either too heavy to install, too tightly coupled to a specific domain, or too low-level to compose.

The library covers the full applied-ML lifecycle on geospatial data:

1. **Preprocess** — turn messy, heterogeneous Earth observation data into structured, analysis-ready datasets.
2. **Infer** — run any trained model (sklearn, JAX/Equinox, PyTorch, NumPyro, plain callable) over those datasets and get xarray output back.
3. **Evaluate** — score predictions against references with pixel-level, spectral, and multiscale metrics.

Every stage uses the same operator abstraction, so they compose into a single pipeline.

---

## User Stories

**Applied ML researcher** — "I have satellite SSH data, a trained neural emulator, and reanalysis for reference. I want to preprocess → predict → evaluate in one composable pipeline, not three separate scripts."

**DA researcher** — "I need to preprocess satellite observations before feeding them into ekalmX, and evaluate the analysis fields after. geo_toolz handles the xarray bookkeeping so I focus on the assimilation."

**Climate scientist** — "I want to compute anomalies, regrid to a common grid, and run spectral analysis on model output vs observations. I don't want to install xesmf or deal with ESMF compilation."

**Student / newcomer** — "I want `Sequential([ValidateCoords(), Regrid(grid), RemoveClimatology(clim)])(ds)` and have it just work. I shouldn't need to learn about CRS, irregular grids, or NaN handling up front."

---

## Design Principles

1. **Everything is an operator** — A regridding step, a trained model, a metric, a pipeline of operators — all are callable objects with a uniform interface. They carry configuration, compose into sequences or DAGs, and can be serialized.

2. **Progressive disclosure** — Three layers of complexity. Layer 0: pure functions you can pipe. Layer 1: composable Operator objects with `Sequential`. Layer 2: functional `Graph` API for branching/merging DAGs. Simple things are simple, complex things are possible.

3. **xarray in, xarray out** — Every operator takes xarray and returns xarray. The compute core is numpy/scipy/sklearn, but xarray is the interface. Coordinates, attributes, and metadata are preserved end-to-end.

4. **Bring your own model** — geo_toolz is not just preprocessing. `ModelOp` wraps any trained model (sklearn, JAX, PyTorch, callable) as an Operator that slots into pipelines. The library marshals xarray ↔ arrays; the user brings the model.

5. **No heavy system dependencies** — The library has no dependency on xesmf, ESMF, dask-ml, or anything that requires non-trivial system-level installation beyond pip/uv.

---

## Identity

### What geo_toolz IS

- A library of composable geoprocessing operators for xarray data
- A pipeline abstraction (Sequential for linear, Graph for DAGs)
- An inference wrapper (ModelOp) that turns any model into an Operator
- An evaluation toolkit (pixel-level, spectral, multiscale metrics)
- The xarray bookkeeping layer for the broader ecosystem (ekalmX, pyrox_gp, somax)

### What geo_toolz is NOT

| Not this | Use instead |
|----------|-------------|
| Data assimilation (EnKF, 4DVar) | ekalmX / vardax |
| GP models, kernel learning | pyrox_gp |
| Ocean/atmosphere forward models | somax / diffrax |
| Structured linear algebra | gaussx |
| Full probabilistic programming | NumPyro |
| Dask-native distributed processing | xarray + dask directly (deferred) |
| Heavyweight regridding (ESMF, conservative) | xesmf (if you can install it) |

---

## Migration Context

**Replaces:** Scattered preprocessing scripts across research projects. No single predecessor — geo_toolz consolidates patterns that were previously copy-pasted per-project.

**Key external inspiration:**

| Tool | What geo_toolz takes from it |
|------|------------------------------|
| Keras functional API | `Input`, `Node`, `Graph` composition model |
| sklearn Pipeline | `Sequential` as a linear chain of transforms |
| torchvision transforms | Composable callable transforms with `__call__` |
| xesmf | Regridding concepts (but pure scipy, no ESMF dependency) |
| xskillscore | Verification metrics (consumed as dependency) |

---

## Connection to Ecosystem

```
                    ┌──────────────┐
                    │ raw data     │  Satellite, reanalysis, model output
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  geo_toolz   │  Preprocess → Infer → Evaluate
                    │  (operators) │
                    └──┬───┬───┬──┘
                       │   │   │
         ┌─────────────┘   │   └─────────────┐
         │                 │                  │
  ┌──────▼──────┐  ┌──────▼──────┐   ┌──────▼──────┐
  │   ekalmX    │  │  pyrox_gp   │   │  user code  │
  │ (DA input/  │  │ (GP input/  │   │ (ML models, │
  │  evaluation)│  │  features)  │   │  analysis)  │
  └─────────────┘  └─────────────┘   └─────────────┘

Inference backends (optional, via ModelOp):
    sklearn, JAX/Equinox, PyTorch, NumPyro, any callable

Companion libraries (optional):
    xarray_sklearn — full sklearn bridge with NaN handling
    xrpatcher — patch-wise tiling for large grids
```
