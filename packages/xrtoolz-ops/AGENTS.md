# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`xrtoolz` is a uv workspace of three packages that together provide a composable operator library for geoprocessing Earth System Data Cubes — preprocess, infer, and evaluate xarray datasets with a uniform pipeline abstraction. The Operator / Sequential / Graph composition core lives in the carrier-agnostic [`pipekit`](https://github.com/jejjohnson/pipekit) framework.

The packages (distribution name → import name):

| Package | Import | Purpose |
|---|---|---|
| `xrtoolz-ops` | `xrops` | Main operator families (geo, ocn, budgets, interpolate, transforms, metrics, viz, …) |
| `xrtoolz` | `xrtoolz` | The umbrella: no code; installs the other two |
| `xrtoolz-products` | `xrproducts` | Authenticated archive readers (CMEMS, CDS, AEMET) + the typed request vocabulary / CF `Variable` registry |

`xrtoolz` holds no code and depends on the other two, so `pip install xrtoolz` installs the whole family. There are no compatibility shims: renames and removals are outright, and every caller in the stack moves in the same release.

## Architecture

### Three-layer stack

| Layer | Name | Contents |
|-------|------|----------|
| 0 | Primitives | Pure functions: `(xr.Dataset, ...) → xr.Dataset` |
| 1 | Operators | `pipekit.Operator` subclasses with uniform `__call__` interface, `pipekit.Sequential` chains |
| 2 | Graph | `pipekit.Graph` DAG API (`Input`, `Node`, `Graph`), `ModelOp` inference wrappers |

### Package structure

Each package lives in `packages/<distribution>/src/<import name>/` with its own `pyproject.toml` and `tests/`; the workspace root ships no code (the top-level `pyproject.toml` only configures `[tool.uv.workspace]` plus shared dev/lint/typecheck/docs groups and git sources for `pipekit`/`xrproducts`). The operators live in `packages/xrtoolz-ops/src/xrops/`; `xrops/__init__.py` exports `Operator` and its own combinators. Import the composition primitives (`Sequential`, `Graph`, `Input`, `Node`, `Tap`) from `pipekit`, `Signature` from `pipekit`, and `Operator` from `xrops` (defined in `xrops._src.operator`).

### Submodule layout

| Path | Scope |
|------|-------|
| `xrops.combinators` | `Augment`, `ApplyToEach` — xarray-Dataset-specific combinators built on `pipekit.Operator` |
| `xrops.geo` | Generic xarray geoprocessing — coordinate validation/CF renaming, subset, masks, detrend/climatology, CRS, regions, regimes, extremes, along-track, 1-D/2-D wavelet spectra |
| `xrops.ocn` | Oceanography physics — kinematics (vorticity, divergence, strain, Okubo–Weiss, streamfunction, geostrophic velocities, KE), SSH diagnostics, CF metadata validation |
| `xrops.budgets` | Conservation budgets — heat/salt/volume/KE residuals, control-volume integrals, boundary fluxes |
| `xrops.interpolate` | Gap-filling (`FillNaN*`), regridding, KNN, binning, smoothing, coarsen/refine, downscaling, resampling, points↔grid, vertical coordinate remapping |
| `xrops.metrics` | Verification metrics — pixel, spectral (PSD, resolved scale), distributional, probabilistic, structural, object-based, physical, forecast, masked, multiscale, Diebold–Mariano, region/leadtime composites, leaderboard |
| `xrops.transforms` | Fourier/DCT/wavelet transforms, signal decompositions, morphology, coordinate remapping, space/time/basis encoders |
| `xrops.grad` | Finite-difference calculus on labeled grids (`partial`, `gradient`, `divergence`, `curl`, `laplacian`; `Gradient` and `Difference` operators) over geoops's difference kernels; cartesian, rectilinear and spherical (sphere by default, `ellipsoid=` for an ellipsoidal metric) |
| `xrops.learn` | scikit-learn on xarray over `pipekit_array.Estimator`: `XarrayEstimator` (Dataset variables as one feature table, DataTree per node or pooled) and the `.sklearn` accessor |
| `xrops.inference` | `ModelOp` inference wrappers (duck-typed; sklearn/JAX adapters). Not re-exported at top level — import explicitly |
| `xrops.utils` | Shared helpers — finite-mask utilities, grid spacing/resolution, validation guards |
| `xrops.viz` | Colormaps, norms, projections, and `viz.validation` panel Operators (spatial maps, PSD, rotary, budgets, events, regime bars) |
| `xrops.atm` / `xrops.atm.gas.ch4` | Atmospheric physics — `open_wrfout` (WRF → CF Dataset), wind speed/direction/components, column integral, hypsometric height, bulk-Richardson PBL height; methane columns under `gas.ch4` (column averaging kernel, dry-air column, mixing ratio ↔ column). Potential temperature still planned |

Design rules:

- Anything domain-agnostic lives in `geo`/`interpolate`/`transforms`/`utils` (the einx and scikit-learn bridges are pipekit-array's, with `xrops.learn` on top); only true physics lives in the domain submodules (`ocn`, `budgets`, `atm`, …). The composition primitives themselves live in `pipekit`; their xarray-aware base is `xrops.Operator`.
- Implementation lives in each submodule's `_src/` directory; public names are re-exported through the submodule `__init__.py` (and, for `metrics`, per-family facade modules like `metrics.pixel`). **Every** public Operator class is importable from its domain package (`xrops.ocn.Streamfunction`, not just `xrops.ocn.operators.Streamfunction`).
- Operator constructor conventions: `variable=` (not `var=`) for a single variable name, `variables=` for lists; singular `dim=` (accepting `str | Sequence[str]`) for reduce-style "dimension(s) to act over" parameters, plural `dims=` only when the value is inherently a fixed collection of axes (image-plane pairs, FFT axes).

### Key directories

| Path | Purpose |
|------|---------|
| `packages/xrtoolz/src/xrtoolz/` | Main package source code (domain submodules, combinators, shims) |
| `packages/xrtoolz-products/src/xrproducts/` | Authenticated archive readers + request types |
| `packages/*/tests/` | Per-package test suites |
| `docs/` | Documentation (MkDocs root site), including `docs/design/` with the full design doc |
| `notebooks/` | Jupyter notebooks |

### Key dependencies

| Package | Role |
|---------|------|
| `numpy` / `scipy` | Array computation, interpolation, spectral, signal processing |
| `scikit-learn` | The `[sklearn]` extra: `xrops.learn`, the decomposition presets, nearest-neighbour regridding and KDE (imported when used) |
| `xarray` / `pandas` | Labeled N-dimensional data interface |
| `rioxarray` / `pyproj` | CRS assignment, reprojection |
| `regionmask` | Land/ocean/country masks |
| `xrft` | Fourier transforms on xarray |
| `xskillscore` | Verification metrics |

xrops's calculus is `xrops.grad`, on geoops's difference kernels (`geotoolz-ops`, which also brings einx); no xrtoolz package depends on JAX. PyTorch and sklearn *models* remain non-dependencies — `ModelOp` uses duck typing so the user installs only what they need.

## Common Commands

```bash
make install              # uv sync --all-packages --all-groups --all-extras + hooks
make test                 # Fast tier across all three packages
make test-all             # Everything, including slow/integration
make test-slow            # Only the slow/integration tiers
make test-doctest         # Docstring examples in xrops.Operator and xrops.grad
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # ty check per package (from each package dir)
make precommit            # Run pre-commit on all files
make docs-serve           # Local docs server
```

### Running a single test

Run from the owning package directory so its pytest config applies:

```bash
cd packages/xrtoolz && uv run pytest tests/test_example.py::TestClass::test_method -v
cd packages/xrtoolz-ops && uv run pytest tests/grad/test_spherical.py -v
```

### Test tiers

Tests are markered `slow` / `integration` (strict markers, registered in
each package's `pyproject.toml`). Automatic CI runs only the fast tier;
the slow and integration tiers run manually via the "Extended Tests"
workflow (`workflow_dispatch`) or `make test-slow`. Never add a slow or
network-touching test without one of these markers.

### Pre-commit checklist (all four must pass)

```bash
make test                                     # Tests (all packages)
uv run --group lint ruff check .              # Lint — ENTIRE repo
uv run --group lint ruff format --check .     # Format — ENTIRE repo
make typecheck                                # ty per package
```

**Critical**: Always lint/format with `.` (repo root). CI runs `ruff check .` which includes every package's `tests/`. Each member package keeps its own `[tool.ruff]` (ruff's nearest-pyproject discovery scopes per-package ignores), its own pytest markers/coverage gates, and its own `[tool.ty]` rules — which is why tests and ty run from the package directories.

## Coding Conventions

- Every `Operator` subclass is a callable with `__call__`, `get_config()`, `__repr__()`
- Layer 0 pure functions live alongside Layer 1 operators in the same submodule
- Stateful operations use the split-object pattern (`CalculateX` returns state, `ApplyX(state)` applies it). Exception: sklearn-style fit/transform state is handled by `xrops.learn.XarrayEstimator` (on `pipekit_array.Estimator`) instead
- Operators holding live non-serializable state (models, closures, Datasets, child operators) set `forbid_in_yaml: ClassVar[bool] = True` (pipekit convention)
- Google-style docstrings. The examples in `xrops.Operator` and `xrops.grad` are doctested by `make test-doctest` (CI runs it too). End a doctest block with a blank line before the closing ```` ``` ```` fence, or doctest reads the fence as expected output
- Type hints on all public functions and methods
- Surgical changes only — don't refactor adjacent code or add docstrings to unchanged code

## Documentation Examples

Example notebooks live in `docs/notebooks/` as jupytext percent-format `.py` files. The workflow:

1. Write the `.py` source (jupytext percent format)
2. Convert and execute: `jupytext --to notebook foo.py` then `jupyter nbconvert --execute --inplace foo.ipynb`
3. Delete the `.py` — the executed `.ipynb` is the committed source of truth
4. `mkdocs-jupyter` renders the pre-executed `.ipynb` with `execute: false`

Figures render inline via `plt.show()` — do **not** use `savefig` or commit separate PNG files. The `.ipynb` cell outputs are the single source of rendered figures.

See `.github/instructions/docs-examples.instructions.md` for full standards.

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed). The authoritative design doc is committed in `docs/design/`. Track ongoing work via GitHub issues.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
