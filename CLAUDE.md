# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Package rules load on demand.** Each `packages/<package>/CLAUDE.md`
  imports that package's `AGENTS.md`, so its layout and conventions are in
  context whenever you work on files there. Read the root "Reuse before you
  write" table before adding any helper.
- **Commands** in `.claude/commands/`: `/create-gh-issue` (issues from the
  templates in `.github/ISSUE_TEMPLATE/`), `/link-gh-issues` (native
  sub-issue / blocked-by links), `/squash-commit` (a squash message for a PR).
- **GitHub** — when the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
