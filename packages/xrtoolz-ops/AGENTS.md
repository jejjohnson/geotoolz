# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`xrtoolz` is a uv workspace of five packages that together provide a composable operator library for geoprocessing Earth System Data Cubes — preprocess, infer, and evaluate xarray datasets with a uniform pipeline abstraction. The Operator / Sequential / Graph composition core lives in the carrier-agnostic [`pipekit`](https://github.com/jejjohnson/pipekit) framework.

The packages (distribution name → import name):

| Package | Import | Purpose |
|---|---|---|
| `xrtoolz` | `xrtoolz` | Main operator families (geo, ocn, budgets, interpolate, transforms, metrics, viz, …) |
| `xrtoolz-core` | `xrcore` | xarray-aware `Operator` (DataTree dispatch) + `Signature` shape descriptor |
| `xrtoolz-grad` | `xrgrad` | Finite-difference calculus on labeled grids (sole carrier of the finitediffx/JAX stack) |
| `xrtoolz-einx` | `xreinx` | xarray ↔ einx named-tensor bridge (functions + Operators; einx loads lazily) |
| `xrtoolz-sklearn` | `xrsklearn` | scikit-learn ↔ xarray bridge (`XarrayEstimator`, `.sklearn` accessors, `SklearnOp`) |

`xrtoolz` depends on the other four, so `pip install xrtoolz` behaves exactly as the pre-split package. Deprecation shims keep `xrtoolz.calc` / `xrtoolz.einx` / `xrtoolz.signature` / `xrtoolz.utils.XarrayEstimator` / `xrtoolz.transforms.SklearnOp` importable through 0.x; new code imports `xrgrad` / `xreinx` / `xrcore` / `xrsklearn` directly.

## Architecture

### Three-layer stack

| Layer | Name | Contents |
|-------|------|----------|
| 0 | Primitives | Pure functions: `(xr.Dataset, ...) → xr.Dataset` |
| 1 | Operators | `pipekit.Operator` subclasses with uniform `__call__` interface, `pipekit.Sequential` chains |
| 2 | Graph | `pipekit.Graph` DAG API (`Input`, `Node`, `Graph`), `ModelOp` inference wrappers |

### Package structure

Each package lives in `packages/<distribution>/src/<import name>/` with its own `pyproject.toml` and `tests/`; the workspace root ships no code (the top-level `pyproject.toml` only configures `[tool.uv.workspace]` plus shared dev/lint/typecheck/docs groups and git sources for `pipekit`/`xrreader`). The main package's public API is re-exported through `packages/xrtoolz/src/xrtoolz/__init__.py`; the composition primitives (`Sequential`, `Graph`, `Input`, `Node`, `Tap`) are re-exported from `pipekit`, and `Operator` / `Signature` from `xrcore`.

### Submodule layout

| Path | Scope |
|------|-------|
| `xrtoolz.combinators` | `Augment`, `ApplyToEach` — xarray-Dataset-specific combinators built on `pipekit.Operator` |
| `xrtoolz.signature` | Shim → `xrcore.Signature` (dict-keyed shape descriptor) |
| `xrtoolz.einx` | Deprecation shim → `xreinx` (the xarray ↔ einx bridge package). See `docs/design/bridges/einx/`. |
| `xrtoolz.geo` | Generic xarray geoprocessing — coordinate validation/CF renaming, subset, masks, detrend/climatology, CRS, regions, regimes, extremes, along-track, 1-D/2-D wavelet spectra |
| `xrtoolz.ocn` | Oceanography physics — kinematics (vorticity, divergence, strain, Okubo–Weiss, streamfunction, geostrophic velocities, KE), SSH diagnostics, CF metadata validation |
| `xrtoolz.calc` | Deprecation shim → `xrgrad` (finite-difference calculus: `partial`/`gradient`/`curl`/`divergence`/`laplacian`, geometry dispatch, grid metrics, constants) |
| `xrtoolz.budgets` | Conservation budgets — heat/salt/volume/KE residuals, control-volume integrals, boundary fluxes |
| `xrtoolz.interpolate` | Gap-filling (`FillNaN*`), regridding, KNN, binning, smoothing, coarsen/refine, downscaling, resampling, points↔grid, vertical coordinate remapping |
| `xrtoolz.metrics` | Verification metrics — pixel, spectral (PSD, resolved scale), distributional, probabilistic, structural, object-based, physical, forecast, masked, multiscale, Diebold–Mariano, region/leadtime composites, leaderboard |
| `xrtoolz.transforms` | Fourier/DCT/wavelet transforms, signal decompositions, morphology, coordinate remapping, space/time/basis encoders, sklearn bridge (`SklearnOp`) |
| `xrtoolz.inference` | `ModelOp` inference wrappers (duck-typed; sklearn/JAX adapters). Not re-exported at top level — import explicitly |
| `xrtoolz.utils` | Shared helpers — finite-mask utilities, grid spacing/resolution, validation guards; re-exports `XarrayEstimator` from `xrsklearn` (importing it still registers the `.sklearn` accessors) |
| `xrtoolz.viz` | Colormaps, norms, projections, and `viz.validation` panel Operators (spatial maps, PSD, rotary, budgets, events, regime bars) |
| `xrtoolz.atm` / `xrtoolz.atm.gas.ch4` | Atmospheric / trace-gas physics — **empty namespace stubs**, planned scope in module docstrings |
| `xrtoolz.rs` | Remote sensing — **empty namespace stub** |
| `xrtoolz.ice` | Cryosphere — **empty namespace stub** |

Design rules:

- Anything domain-agnostic lives in `geo`/`interpolate`/`transforms`/`utils` (or a sibling workspace package — `xrgrad`, `xreinx`, `xrsklearn`); only true physics lives in the domain submodules (`ocn`, `budgets`, `atm`, …). The composition primitives themselves live in `pipekit` (and their xarray-aware base in `xrcore`), not here.
- Implementation lives in each submodule's `_src/` directory; public names are re-exported through the submodule `__init__.py` (and, for `metrics`, per-family facade modules like `metrics.pixel`). **Every** public Operator class is importable from its domain package (`xrtoolz.ocn.Streamfunction`, not just `xrtoolz.ocn.operators.Streamfunction`).
- Operator constructor conventions: `variable=` (not `var=`) for a single variable name, `variables=` for lists; singular `dim=` (accepting `str | Sequence[str]`) for reduce-style "dimension(s) to act over" parameters, plural `dims=` only when the value is inherently a fixed collection of axes (image-plane pairs, FFT axes).

### Key directories

| Path | Purpose |
|------|---------|
| `packages/xrtoolz/src/xrtoolz/` | Main package source code (domain submodules, combinators, shims) |
| `packages/xrtoolz-core/src/xrcore/` | `Operator` (DataTree dispatch) + `Signature` |
| `packages/xrtoolz-grad/src/xrgrad/` | Finite-difference calculus |
| `packages/xrtoolz-einx/src/xreinx/` | xarray ↔ einx bridge |
| `packages/xrtoolz-sklearn/src/xrsklearn/` | sklearn ↔ xarray bridge |
| `packages/*/tests/` | Per-package test suites |
| `docs/` | Documentation (MkDocs root site), including `docs/design/` with the full design doc |
| `notebooks/` | Jupyter notebooks |

### Key dependencies

| Package | Role |
|---------|------|
| `numpy` / `scipy` | Array computation, interpolation, spectral, signal processing |
| `scikit-learn` | Nearest-neighbor regridding, preprocessing utilities |
| `xarray` / `pandas` | Labeled N-dimensional data interface |
| `rioxarray` / `pyproj` | CRS assignment, reprojection |
| `regionmask` | Land/ocean/country masks |
| `xrft` | Fourier transforms on xarray |
| `xskillscore` | Verification metrics |
| `einx` | Named-tensor backend for the einx-based kernels still in this package (`transforms.encoders.basis`, `metrics.instance`, `metrics.distributional`); the bridge itself is `xreinx` |

The finitediffx/JAX stack rides exclusively in `xrtoolz-grad`, so it still reaches a full `xrtoolz` install transitively (ocn/budgets/metrics call `xrgrad` eagerly) — but the scoped packages (`xrtoolz-core`, `xrtoolz-einx`, `xrtoolz-sklearn`) install without it, and making `xrtoolz-grad` optional later is now a packaging decision, not a refactor. PyTorch and sklearn *models* remain non-dependencies — `ModelOp` uses duck typing so the user installs only what they need.

## Common Commands

```bash
make install              # uv sync --all-packages --all-groups --all-extras + hooks
make test                 # Fast tier across all five packages
make test-all             # Everything, including slow/integration
make test-slow            # Only the slow/integration tiers
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
cd packages/xrtoolz-grad && uv run pytest tests/test_spherical.py -v
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

**Critical**: Always lint/format with `.` (repo root). CI runs `ruff check .` which includes every package's `tests/`. Each member package keeps its own `[tool.ruff]` (ruff's nearest-pyproject discovery scopes per-package ignores, e.g. xrgrad's jaxtyping `F722`), its own pytest markers/coverage gates, and its own `[tool.ty]` rules — which is why tests and ty run from the package directories.

## Coding Conventions

- Every `Operator` subclass is a callable with `__call__`, `get_config()`, `__repr__()`
- Layer 0 pure functions live alongside Layer 1 operators in the same submodule
- Stateful operations use the split-object pattern (`CalculateX` returns state, `ApplyX(state)` applies it). Exception: sklearn-style fit/transform state is handled by `xrsklearn.XarrayEstimator` + `xrsklearn.SklearnOp` instead
- Operators holding live non-serializable state (models, closures, Datasets, child operators) set `forbid_in_yaml: ClassVar[bool] = True` (pipekit convention)
- Google-style docstrings
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
