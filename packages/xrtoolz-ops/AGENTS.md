# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`xrtoolz`: composable operator library for geoprocessing Earth System Data Cubes — preprocess, infer, and evaluate xarray datasets with a uniform pipeline abstraction. The Operator / Sequential / Graph composition core lives in the carrier-agnostic [`pipekit`](https://github.com/jejjohnson/pipekit) framework; `xrtoolz` is a direct consumer that adds the xarray-specific operator families (geo, ocn, atm, rs, …) on top.

## Architecture

### Three-layer stack

| Layer | Name | Contents |
|-------|------|----------|
| 0 | Primitives | Pure functions: `(xr.Dataset, ...) → xr.Dataset` |
| 1 | Operators | `pipekit.Operator` subclasses with uniform `__call__` interface, `pipekit.Sequential` chains |
| 2 | Graph | `pipekit.Graph` DAG API (`Input`, `Node`, `Graph`), `ModelOp` inference wrappers |

### Package structure

All implementation lives in `src/xrtoolz/`. The public API is re-exported through `src/xrtoolz/__init__.py`. The composition primitives (`Operator`, `Sequential`, `Graph`, `Input`, `Node`, `ConfigMixin`, `Tap`) are re-exported from `pipekit`.

### Submodule layout

| Path | Scope |
|------|-------|
| `xrtoolz.combinators` | `Augment`, `ApplyToEach` — xarray-Dataset-specific combinators built on `pipekit.Operator` |
| `xrtoolz.signature` | `Signature` — dict-keyed shape descriptor used by `compute_output_signature` |
| `xrtoolz.einx` | Labeled named-tensor algebra bridging xarray + [einx](https://github.com/fferflo/einx) — `einsum`/`rearrange`/`reduce`/`repeat`, `matmul`/`outer`/`batch_matmul`, `pack_dataset`/`unpack_dataset`, and matching Operators. Pattern axis tokens are DataArray dim names. See `docs/design/bridges/einx/`. |
| `xrtoolz.geo` | Generic xarray geoprocessing — coordinate validation/CF renaming, subset, masks, detrend/climatology, CRS, regions, regimes, extremes, along-track, 1-D/2-D wavelet spectra |
| `xrtoolz.ocn` | Oceanography physics — kinematics (vorticity, divergence, strain, Okubo–Weiss, streamfunction, geostrophic velocities, KE), SSH diagnostics, CF metadata validation |
| `xrtoolz.calc` | Finite-difference calculus on labeled grids — `partial`/`gradient`/`curl`/`divergence`/`laplacian` with cartesian/rectilinear/spherical geometry dispatch, grid metrics, physical constants |
| `xrtoolz.budgets` | Conservation budgets — heat/salt/volume/KE residuals, control-volume integrals, boundary fluxes |
| `xrtoolz.interpolate` | Gap-filling (`FillNaN*`), regridding, KNN, binning, smoothing, coarsen/refine, downscaling, resampling, points↔grid, vertical coordinate remapping |
| `xrtoolz.metrics` | Verification metrics — pixel, spectral (PSD, resolved scale), distributional, probabilistic, structural, object-based, physical, forecast, masked, multiscale, Diebold–Mariano, region/leadtime composites, leaderboard |
| `xrtoolz.transforms` | Fourier/DCT/wavelet transforms, signal decompositions, morphology, coordinate remapping, space/time/basis encoders, sklearn bridge (`SklearnOp`) |
| `xrtoolz.inference` | `ModelOp` inference wrappers (duck-typed; sklearn/JAX adapters). Not re-exported at top level — import explicitly |
| `xrtoolz.utils` | Shared helpers — finite-mask utilities, grid spacing/resolution, validation guards, `XarrayEstimator` sklearn wrapper + `.sklearn` accessor |
| `xrtoolz.viz` | Colormaps, norms, projections, and `viz.validation` panel Operators (spatial maps, PSD, rotary, budgets, events, regime bars) |
| `xrtoolz.atm` / `xrtoolz.atm.gas.ch4` | Atmospheric / trace-gas physics — **empty namespace stubs**, planned scope in module docstrings |
| `xrtoolz.rs` | Remote sensing — **empty namespace stub** |
| `xrtoolz.ice` | Cryosphere — **empty namespace stub** |

Design rules:

- Anything domain-agnostic lives in `geo`/`calc`/`interpolate`/`transforms`/`utils`; only true physics lives in the domain submodules (`ocn`, `budgets`, `atm`, …). The composition primitives themselves live in `pipekit`, not here.
- Implementation lives in each submodule's `_src/` directory; public names are re-exported through the submodule `__init__.py` (and, for `metrics`, per-family facade modules like `metrics.pixel`). **Every** public Operator class is importable from its domain package (`xrtoolz.ocn.Streamfunction`, not just `xrtoolz.ocn.operators.Streamfunction`).
- Operator constructor conventions: `variable=` (not `var=`) for a single variable name, `variables=` for lists; singular `dim=` (accepting `str | Sequence[str]`) for reduce-style "dimension(s) to act over" parameters, plural `dims=` only when the value is inherently a fixed collection of axes (image-plane pairs, FFT axes).

### Key directories

| Path | Purpose |
|------|---------|
| `src/xrtoolz/` | Main package source code |
| `src/xrtoolz/combinators.py` | `Augment`, `ApplyToEach` — xarray-Dataset combinators on top of `pipekit.Operator` |
| `src/xrtoolz/signature.py` | `Signature` — dict-keyed shape descriptor |
| `src/xrtoolz/<domain>/` | Domain submodules |
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
| `einx` | Named-tensor algebra backend for `xrtoolz.einx` (core dep; lazily imported) |

JAX, PyTorch, sklearn models are **not** transitive dependencies — `ModelOp` uses duck typing so the user installs only what they need.

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Fast tier: pytest -m "not slow and not integration"
make test-all             # Everything, including slow/integration
make test-slow            # Only the slow/integration tiers
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/xrtoolz
make precommit            # Run pre-commit on all files
make docs-serve           # Local docs server
```

### Running a single test

```bash
uv run pytest tests/test_example.py::TestClass::test_method -v
```

### Test tiers

Tests are markered `slow` / `integration` (strict markers, registered in
`pyproject.toml`). Automatic CI runs only the fast tier; the slow and
integration tiers run manually via the "Extended Tests" workflow
(`workflow_dispatch`) or `make test-slow`. Never add a slow or
network-touching test without one of these markers.

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v                              # Tests
uv run --group lint ruff check .              # Lint — ENTIRE repo, not just src/xrtoolz/
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check src/xrtoolz  # Typecheck — package only
```

**Critical**: Always lint/format with `.` (repo root), not `src/xrtoolz/`. CI runs `ruff check .` which includes `tests/` and `scripts/`.

## Coding Conventions

- Every `Operator` subclass is a callable with `__call__`, `get_config()`, `__repr__()`
- Layer 0 pure functions live alongside Layer 1 operators in the same submodule
- Stateful operations use the split-object pattern (`CalculateX` returns state, `ApplyX(state)` applies it). Exception: sklearn-style fit/transform state is handled by `utils.XarrayEstimator` + `transforms.SklearnOp` instead (documented in `transforms/operators.py`)
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
