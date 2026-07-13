# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`geotoolz` is a uv workspace of three packages that together provide a
composable remote-sensing stack on top of `georeader.GeoTensor`, with the
Operator / Sequential / Graph composition core supplied by the external
[`pipekit`](https://github.com/jejjohnson/pipekit) framework. Built with
Python 3.12+, uv, pytest, and MkDocs.

The packages (distribution name → import name):

| Package            | Import       | Purpose                                                                 |
|--------------------|--------------|-------------------------------------------------------------------------|
| `geotoolz`         | `geotoolz`   | RS operator families (radiometry, indices, qa/mask, geom, readers, einx, patch_ops bridge, …) |
| `geotoolz-patcher` | `geopatcher` | Four-axis Patcher framework (Geometry × Sampler × Window × Aggregation over a `Field` protocol) |
| `geotoolz-catalog` | `geocatalog` | Queryable spatiotemporal index (GeoSlice contract, in-memory + DuckDB backends, GeoParquet interchange, sources/matchup/staging) |

Import names are unchanged from the pre-monorepo repos (`import geopatcher`,
`import geocatalog`) — only the distribution names carry the `geotoolz-`
prefix. Cross-package wiring: `geotoolz[patch]` → `geotoolz-patcher[pipekit]`;
`geotoolz-catalog[patch]` → `geotoolz-patcher` (for `staging.field_for`);
soft imports (obstore pools) resolve when co-installed. The workspace root
ships no code — the top-level `pyproject.toml` only configures
`[tool.uv.workspace]` plus shared dev/lint/typecheck/docs groups.

## Common Commands

```bash
make install              # uv sync --all-packages --all-groups --all-extras + hooks
make test                 # Fast tier across all three packages
make test-all             # Everything incl. geotoolz slow/integration tiers
make format               # ruff format . && ruff check --fix .
make lint                 # ruff check .   (entire repo)
make typecheck            # ty check per package (from each package dir)
make docs-serve           # Local MkDocs preview (root site)
```

### Running a single test

Run from the owning package directory so its pytest config applies:

```bash
cd packages/geotoolz && uv run pytest tests/test_indices.py::test_ndvi -v
cd packages/geotoolz-patcher && uv run pytest tests/test_sampler.py -v
cd packages/geotoolz-catalog && uv run pytest tests/test_geoslice.py -v
```

### Pre-commit checklist (all must pass)

```bash
make test                                       # Tests (all packages)
uv run --group lint ruff check .                # Lint — ENTIRE repo
uv run --group lint ruff format --check .       # Format — ENTIRE repo
make typecheck                                  # ty per package
```

**Critical**: Always lint/format with `.` (repo root). CI runs `ruff check .`
which includes every package's `tests/`. Each member package keeps its own
`[tool.ruff]` (ruff's nearest-pyproject discovery scopes per-package ignores,
e.g. geotoolz's jaxtyping `F722`), its own pytest markers/coverage gates, and
its own `[tool.ty]` rules — which is why tests and ty run from the package
directories.

## Architecture

### Workspace layout

```
packages/
├── geotoolz/                 # src/geotoolz — operator families; two-tier model
│   │                         # (jaxtyped numpy primitives in _src/array.py per
│   │                         # module, carrier-aware pipekit.Operator classes in
│   │                         # _src/operators.py). Carrier-preserving via
│   │                         # geotoolz._src.wrap.wrap_like.
├── geotoolz-patcher/         # src/geopatcher — SpatialPatcher / TemporalPatcher /
│   │                         # SpatioTemporalPatcher, Field adapters, hooks,
│   │                         # journal, PatchCache, pipekit integration.
└── geotoolz-catalog/         # src/geocatalog — GeoCatalog Protocol (InMemory +
                              # DuckDB), GeoSlice, loaders, sources, matchup,
                              # staging, cyclopts CLI.
```

Each package's public API is re-exported through its `src/<import>/__init__.py`.
Per-package docs live under `packages/*/docs/`; the root `docs/` + `mkdocs.yml`
is the geotoolz site.

### Test tiers

`packages/geotoolz` tests are markered `slow` / `integration` (fast tier runs
in CI; extended tiers via the "Extended Tests" workflow_dispatch).
`packages/geotoolz-catalog` has a `live` marker (real external APIs, always
deselected) and an opt-in `tests/bench` suite (`pytest tests/bench
--benchmark-only`). Never add a slow or network-touching test without a marker.

## Coding Conventions

- Google-style docstrings; `dataclasses`/`attrs` for data, `Protocol` for seams.
- Type hints on all public functions and methods.
- Pure functions where possible; side effects isolated and explicit.
- Surgical changes only — don't refactor adjacent code or add docstrings to
  unchanged code.
- Releases via release-please with per-package components (`geotoolz-vX.Y.Z`,
  `geotoolz-patcher-vX.Y.Z`, `geotoolz-catalog-vX.Y.Z`); conventional-commit
  titles are enforced.

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed). Track work via GitHub issues instead.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
