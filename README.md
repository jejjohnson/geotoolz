# geotoolz

[![Tests](https://github.com/jejjohnson/geotoolz/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/geotoolz/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/geotoolz/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/geotoolz/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/geotoolz/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/geotoolz)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

> **Status:** pre-alpha (`0.0.0`). API is in flux; the public surface is not yet stable.

A composable Operator library for remote sensing, built on top of `georeader.GeoTensor`. Pipelines are written as a `Sequential` of operators or a `Graph` of named ops — declared in code or YAML, executed eagerly on `GeoTensor`s. The Operator / Sequential / Graph composition core lives in the carrier-agnostic [`pipekit`](https://github.com/jejjohnson/pipekit) framework; `geotoolz` is a direct consumer that adds the remote-sensing operator families on top. The sibling library to `xr_toolz` for climate workflows: same architectural patterns, different substrate (`numpy`-subclass `GeoTensor` with `__array_ufunc__` vs `xarray.Dataset`), different audience.

```python
import geotoolz as gz

# NDVI on cloud-masked Sentinel-2 in two operators composed
ndvi = gz.Sequential([
    gz.cloud.MaskClouds(qa_band="QA60", bits=[10, 11]),
    gz.indices.NDVI(red_idx=2, nir_idx=3),
])(gt)
```

The full design proposal lives in
[`research_journal_v2/notes/geotoolz/plans/geotoolz/geotoolz.md`](../research_journal_v2/notes/geotoolz/plans/geotoolz/geotoolz.md)
(two-tier model, ~80 operators in 12 modules, sensor presets, Hydra-zen interop). Note that **no operators have been implemented yet** — the repo currently holds only the package skeleton.

## Installation

This repo is pre-release and not yet on PyPI. `geotoolz` depends on
[`pipekit`](https://github.com/jejjohnson/pipekit) (also pre-PyPI), so
use `uv` — it reads the GitHub git source declared in `pyproject.toml`
and resolves `pipekit` transitively:

```bash
git clone https://github.com/jejjohnson/geotoolz.git
cd geotoolz
make install        # uv sync --all-groups + pre-commit hooks
```

Or install in one shot from GitHub:

```bash
uv pip install "git+https://github.com/jejjohnson/geotoolz@main"
```

Plain `pip install git+https://...` will fail until `pipekit` reaches
PyPI, because pip doesn't read `[tool.uv.sources]`.

## Development

```bash
make install     # uv sync --all-groups + pre-commit hooks
make test        # uv run pytest -v
make lint        # uv run --group lint ruff check .
make format      # ruff format + ruff check --fix
make typecheck   # uv run --group typecheck ty check src/geotoolz
make precommit   # uv run pre-commit run --all-files
make docs-serve  # local MkDocs server
```

Pre-commit checklist (all four must pass, mirrors CI):

```bash
uv run pytest -v
uv run --group lint ruff check .
uv run --group lint ruff format --check .
uv run --group typecheck ty check src/geotoolz
```

## Layout

```
geotoolz/
├── pyproject.toml          # PEP 621 + ruff + ty + pytest config
├── src/geotoolz/           # Package source (src layout)
├── tests/                  # Test suite
├── docs/                   # MkDocs site
├── notebooks/              # Example notebooks
├── CLAUDE.md AGENTS.md     # Agent instructions
└── Makefile                # Common dev commands
```

Plans and design docs are tracked outside this repo in `research_journal_v2/notes/geotoolz/`. Work items are tracked as GitHub issues.

## License

MIT — see [LICENSE](LICENSE).
