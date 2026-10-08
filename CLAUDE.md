# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Package rules load on demand.** Each `packages/<package>/CLAUDE.md`
  imports that package's `AGENTS.md`, so its layout and conventions are in
  context whenever you work on files there. Read the root "Reuse before you
  write" table before adding any helper.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - building on the stack: `add-operator`, `add-product-reader`,
    `add-catalog-source`, `add-field-adapter`, `docs-diagram`;
  - shipping: `pre-pr-check`, `geotoolz-review`, `squash-commit`;
  - GitHub housekeeping: `create-gh-issue`, `link-gh-issues`.
- **Subagent** `reuse-reviewer` (`.claude/agents/`): a read-only check that a
  diff does not re-implement something in `docs/capabilities.md` or a
  shared `_src/` toolkit. Run it before committing new helpers, readers or
  modules; `geotoolz-review` uses it.
- **GitHub** — when the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
