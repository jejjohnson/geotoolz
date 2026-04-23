# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`xr_toolz`: composable operator library for geoprocessing Earth System Data Cubes — preprocess, infer, and evaluate xarray datasets with a uniform pipeline abstraction.

## Architecture

### Three-layer stack

| Layer | Name | Contents |
|-------|------|----------|
| 0 | Primitives | Pure functions: `(xr.Dataset, ...) → xr.Dataset` |
| 1 | Operators | `Operator` subclasses with uniform `__call__` interface, `Sequential` chains |
| 2 | Graph | Functional `Graph` DAG API (`Input`, `Node`, `Graph`), `ModelOp` inference wrappers |

### Package structure

All implementation lives in `src/xr_toolz/`. The public API is re-exported through `src/xr_toolz/__init__.py`.

### Submodule layout — organised by Earth-science domain

| Path | Scope |
|------|-------|
| `xr_toolz.core` | `Operator`, `Sequential`, `Input`, `Node`, `Graph` — composition primitives (domain-agnostic) |
| `xr_toolz.geo` | Generic xarray geoprocessing — validation, subset, masks, regrid, detrend, interpolation, metrics, spectral, encoders, crs, sklearn, inference |
| `xr_toolz.ocn` | Oceanography physics — coriolis, streamfunction, geostrophic velocities, vorticity, MLD, Brunt–Väisälä, KE, Okubo–Weiss |
| `xr_toolz.atm` | Atmospheric physics — potential temperature, wind speed/direction |
| `xr_toolz.atm.gas.ch4` | Trace-gas (methane) physics — column averaging kernel, dry air column, mixing ratio |
| `xr_toolz.rs` | Remote sensing — radiance/reflectance, brightness temperature, NDVI |
| `xr_toolz.ice` | Cryosphere — reserved namespace, no content yet |

Design rule: anything domain-agnostic lives in `geo`; only true physics lives in the other domain submodules.

### Key directories

| Path | Purpose |
|------|---------|
| `src/xr_toolz/` | Main package source code |
| `src/xr_toolz/core/` | Layer 1 + 2 composition primitives |
| `src/xr_toolz/<domain>/` | Domain submodules |
| `tests/` | Test suite |
| `docs/` | Documentation (MkDocs), including `docs/design/` with the full design doc |
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

JAX, PyTorch, sklearn models are **not** transitive dependencies — `ModelOp` uses duck typing so the user installs only what they need.

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests: uv run pytest -v
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/xr_toolz
make precommit            # Run pre-commit on all files
make docs-serve           # Local docs server
```

### Running a single test

```bash
uv run pytest tests/test_example.py::TestClass::test_method -v
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v                              # Tests
uv run --group lint ruff check .              # Lint — ENTIRE repo, not just src/xr_toolz/
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check src/xr_toolz  # Typecheck — package only
```

**Critical**: Always lint/format with `.` (repo root), not `src/xr_toolz/`. CI runs `ruff check .` which includes `tests/` and `scripts/`.

## Coding Conventions

- Every `Operator` subclass is a callable with `__call__`, `get_config()`, `__repr__()`
- Layer 0 pure functions live alongside Layer 1 operators in the same submodule
- Stateful operations use the split-object pattern (`CalculateX` returns state, `ApplyX(state)` applies it)
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
