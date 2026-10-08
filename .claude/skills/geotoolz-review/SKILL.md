---
name: geotoolz-review
description: Review a change or pull request in the geotoolz workspace against the repo's own rules — CODE_REVIEW.md, the package AGENTS.md contracts, and reuse of the stack's primitives. Use when asked to review a diff, branch or PR in this repo.
---

# Review a geotoolz change

1. **Get the diff** as described in `CODE_REVIEW.md` ("How to Obtain the
   Diff"), or from the PR. Note which packages it touches and read their
   `packages/<package>/AGENTS.md`.
2. **Reuse** — run the `reuse-reviewer` subagent on the diff. Its findings
   are the most important part of the review: code that re-implements a
   primitive is the main way this stack decays.
3. **Contracts** — first "The two contracts" in the root `AGENTS.md`, for
   every new or changed operator in any package (the geotoolz families,
   `geotoolz.patch_ops`, `geopatcher.integrations.pipekit`, the
   geoproducts presets). Check:
   - pipekit `Operator`:
     - a keyword-only constructor that holds no carrier;
     - `_apply` only, never `__call__`;
     - an auto-derived JSON `get_config()`;
     - `forbid_in_yaml` on live objects and `_terminal` on non-carrier
       outputs;
     - nothing that rebuilds a pipekit block (retry, fallback, cache, tap,
       parallel map).
   - `GeoTensor`:
     - plain array in gives plain array out, GeoTensor in gives
       GeoTensor out, via `wrap_like`;
     - fresh `attrs` with per-band keys that match the output;
     - an output fill that fits its dtype and meaning;
     - the band axis at `-3`, with `(T, C, H, W)` handled or rejected;
     - second carriers grid-checked;
     - the input never mutated.

   Then check the package rules it touches:
   - geotoolz: Tier A / Tier B split, keyword-only constructors, the
     parameter vocabulary, `wrap_like` rewrap, nodata via `_src/valid.py`;
   - geopatcher: one home per name, unprefixed axes, `Field` contract,
     config round-trip, `PatchCache` format id untouched;
   - geocatalog: namespace homes, lazy extras, `GeoSlice` / naive-UTC
     contract, schema version + migration for persisted changes;
   - geocloud: obstore clients only from `geocloud.store`;
   - geoproducts: `ProductReader` contract, lazy windows, the shared
     toolkit, synthetic-file tests validated against a reference.
4. **Checklist** — the rest of `CODE_REVIEW.md` (idioms, docs, errors,
   tests, security), skipping anything ruff or ty already enforces.
5. **Verify claims** — for anything you would flag as a bug, trace a real
   input to the failure (or run it) before reporting it.

Report in the format `CODE_REVIEW.md` gives (summary, then findings by
priority), each with file:line and a concrete proposed change.
