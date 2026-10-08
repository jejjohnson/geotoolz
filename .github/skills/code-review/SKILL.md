---
name: code-review
description: Review a change or pull request in the geotoolz workspace against CODE_REVIEW.md, the package AGENTS.md contracts and reuse of the stack's primitives.
---

# Code review

Follow the same steps as the repo's review skill,
[`.claude/skills/geotoolz-review/SKILL.md`](../../../.claude/skills/geotoolz-review/SKILL.md):
read the touched packages' `AGENTS.md`, check the diff for re-implemented
functionality with the procedure in
[`.claude/agents/reuse-reviewer.md`](../../../.claude/agents/reuse-reviewer.md)
(search `docs/capabilities.md` and the shared `_src/` toolkits for every
helper the diff adds), then apply [`CODE_REVIEW.md`](../../../CODE_REVIEW.md).
