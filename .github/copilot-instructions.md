# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the
single source of truth for every coding agent working here (package map,
"reuse before you write", commands, the pre-commit checklist, git and PR
rules). Each package adds its own rules in `packages/<package>/AGENTS.md`;
read that file before changing the package.

The essentials, in case you only read this file:

- This is a uv workspace of five packages under `packages/`
  (`geotoolz`, `geotoolz-patcher`, `geotoolz-catalog`, `geotoolz-cloud`,
  `geotoolz-products`); each has `src/<import name>/` and `tests/`.
- Keep the two contracts in `AGENTS.md` ("The two contracts"):
  - pipekit `Operator`: a keyword-only constructor, an auto-derived JSON
    `get_config()`, `_apply` only, carrier in and carrier out;
  - `GeoTensor`: rewrap with `wrap_like`, fresh `attrs`, a fill value
    that matches the output, and the band axis at `-3`.
- Reuse the shared toolkits in each package's `_src/` instead of writing new
  helpers; add missing helpers there, not inline.
- Before committing: `make test`, `uv run --group lint ruff check .`,
  `uv run --group lint ruff format --check .`, `make typecheck` — always from
  the repo root.
- Path-scoped standards live in `.github/instructions/`; code review follows
  [`CODE_REVIEW.md`](../CODE_REVIEW.md).
