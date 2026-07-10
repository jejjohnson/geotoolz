# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`geotoolz` is a composable Operator library for remote sensing, sitting on top of `georeader.GeoTensor`. The Operator / Sequential / Graph composition core lives in the carrier-agnostic [`pipekit`](https://github.com/jejjohnson/pipekit) framework — `geotoolz` is a direct consumer that re-exports the common names at the top level and adds the RS-specific operator families (radiometry, spectral indices, cloud masking, sampling, tiled inference, sensor presets). Two-tier model: jaxtyped numpy primitives in `_src/array.py` per module (shape aliases like `Float[np.ndarray, "c h w"]`, annotation-only — no runtime checking), carrier-aware `pipekit.Operator` subclasses in `_src/operators.py`. Operators accept either a `GeoTensor` or a plain `np.ndarray` and return the same carrier kind (`geotoolz._src.wrap.wrap_like`); genuinely geo-dependent ops (reprojection, rasterisation, footprint math) require a GeoTensor and raise a clear `TypeError` on plain arrays. Built with Python 3.12+, uv, pytest, and MkDocs. See the design report in `research_journal_v2/notes/geotoolz/plans/geotoolz/geotoolz.md`.

The four-axis Patcher framework that used to live at `geotoolz.patch` now ships as its own package: [`geopatcher`](https://github.com/jejjohnson/geopatcher). Install with the `[patch]` extra (`pip install 'geotoolz[patch]'`), which pulls in `geopatcher[pipekit]` and exposes the Operator-graph bridge (`GridSampler`, `ApplyToChips`, `Stitch`) at both `geotoolz.patch_ops` and `geopatcher.integrations.pipekit` — both module paths return the same classes; pick whichever reads better in your code.

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Fast tier: pytest -m "not slow and not integration"
make test-all             # Everything, including slow/integration
make test-slow            # Only the slow/integration tiers
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/geotoolz
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
uv run --group lint ruff check .              # Lint — ENTIRE repo, not just src/geotoolz/
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check src/geotoolz  # Typecheck — package only
```

**Critical**: Always lint/format with `.` (repo root), not `src/geotoolz/`. CI runs `ruff check .` which includes `tests/` and `scripts/`.

## Architecture

### Package structure

All implementation lives in `src/geotoolz/`. The public API is re-exported through `src/geotoolz/__init__.py`.

Top-level re-export policy: every public Operator class is available at
`geotoolz.*`; Tier-A numpy primitives live in their submodules only. On
name collisions the domain-canonical class wins the top-level name
(`ApplySRF` → radiometry, `NormalizedDifference` → indices).

Deprecated compatibility aliases (do not add new code under them):
`geotoolz.cloud` (contents moved: extraction → `geotoolz.qa`,
application → `geotoolz.mask`) and `geotoolz.model` (`ModelOp` moved to
`geotoolz.learn`).

Shared cross-module primitives live in the top-level `geotoolz/_src/`:
`wrap.py` (`wrap_like` carrier rewrap), `shape.py` (`single_band`),
`config.py` (`jsonable` get_config coercion), `stretch.py`
(`percentile_stretch`), `blending.py` (overlap-add). Use these instead
of re-implementing per module.

### Key directories

| Path | Purpose |
|------|---------|
| `src/geotoolz/` | Main package source code |
| `tests/` | Test suite |
| `docs/` | Documentation (MkDocs) |
| `notebooks/` | Jupyter notebooks |
| `scripts/` | Example scripts |

## Documentation Examples

Example notebooks live in `docs/notebooks/` as jupytext percent-format `.py` files. The workflow:

1. Write the `.py` source (jupytext percent format)
2. Convert and execute: `jupytext --to notebook foo.py` then `jupyter nbconvert --execute --inplace foo.ipynb`
3. Delete the `.py` — the executed `.ipynb` is the committed source of truth
4. `mkdocs-jupyter` renders the pre-executed `.ipynb` with `execute: false`

Figures render inline via `plt.show()` — do **not** use `savefig` or commit separate PNG files. The `.ipynb` cell outputs are the single source of rendered figures.

See `.github/instructions/docs-examples.instructions.md` for full standards.

## Coding Conventions

- Google-style docstrings
- `dataclasses` or `attrs` for data containers
- Type hints on all public functions and methods
- Pure functions where possible; side effects isolated and explicit
- Surgical changes only — don't refactor adjacent code or add docstrings to unchanged code

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed). Track work via GitHub issues instead.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
