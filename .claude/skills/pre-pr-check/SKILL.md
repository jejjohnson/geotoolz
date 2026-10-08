---
name: pre-pr-check
description: Run the full pre-PR verification for the geotoolz workspace — lint, format, types, every affected package's tests with coverage, the capability index, the docs build, a no-extras install and a clean-checkout check. Use before committing, pushing or opening a pull request, and after any change to packaging, imports or public API.
---

# Pre-PR check

Run from the repo root and fix what fails before committing. Report results
honestly: what ran, what passed, what was skipped.

## Always

```bash
uv run --group lint ruff check .          # entire repo, incl. every tests/
uv run --group lint ruff format --check .
make typecheck                            # ty, from each package directory
```

Tests, with coverage, from each package directory you touched (and any
package that imports it):

```bash
cd packages/<package> && uv run pytest -q
```

geotoolz's fast tier: `uv run pytest -q -m "not slow and not integration"`.

## When the public API changed

```bash
make capabilities                         # regenerate docs/capabilities.md
uv run python scripts/capabilities.py --check
```

The check also fails on a name two packages export for different objects —
rename rather than allowlist, unless the user decides otherwise.

## When docs changed

```bash
uv run --group docs mkdocs build --strict
```

The notebook plugin can stall locally; if it does, build with a copy of
`mkdocs.yml` that drops `mkdocs-jupyter` and say so.

## When packaging, extras or imports changed

- **No-extras install** — mirror `.github/workflows/base-install.yml` for the
  package: a scratch environment with only its base dependencies and `dev`
  group (`UV_PROJECT_ENVIRONMENT=<scratch> uv sync --locked --package
  <dist> --no-default-groups --group dev`), then run that job's smoke
  script and the package's tests.
- **Everything is tracked** — `git status --ignored` must not list a new
  source file as ignored (a bare `build/` rule once swallowed
  `geocatalog/build/`). Then check a clean checkout of the commit:
  `git worktree add --detach <scratch> HEAD`, run ruff and an import of
  every module there, and remove the worktree.

## Before pushing

- Conventional Commits title with a lowercase subject; `!` and a
  `BREAKING CHANGE:` footer for a breaking change.
- Push only to your feature branch, only when asked.
