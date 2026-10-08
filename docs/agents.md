# Building with agents

The stack is meant to be built on: your code should be the science, and the
plumbing — discovery, indexing, reading, tiling, stitching, object storage —
should come from these packages. Coding agents tend to re-implement what
they cannot see, so the stack ships three things that let them find it.

## The capability index

[Capability index](capabilities.md) lists every public name in the five
packages, once, at its home, with a one-line summary. It is regenerated from
the code and checked in CI, so it never drifts. Point an agent at it before
it writes a helper.

## The Claude Code plugin

The repository is a Claude Code plugin marketplace. In any project:

```text
/plugin marketplace add jejjohnson/geotoolz
/plugin install geotoolz@geotoolz
```

The plugin adds:

- **`build-geo-pipeline`** (skill) — loads whenever a task involves rasters,
  satellite imagery, STAC or tiling: which package does what, the canonical
  catalog → patcher → operators → write pipeline, the pipekit `Operator` and
  `GeoTensor` contracts your own steps must keep, and the anti-patterns to
  avoid.
- **`geostack-reuse-reviewer`** (subagent) — a read-only check of a diff for
  code that re-implements something the installed packages already provide.

## llms.txt

For other agents and tools, the docs site serves
[`llms.txt`](https://jejjohnson.github.io/geotoolz/llms.txt): a curated map
of the stack and its key pages.

## Rules for your project's `AGENTS.md`

Paste this into the agent instructions of a project that builds on the
stack:

```markdown
## Geospatial code: build on the geotoolz stack

This project uses geotoolz (operators), geocatalog (find / index / load),
geopatcher (tile / stitch), geoproducts (EO product readers) and geocloud
(object storage). Before writing any raster, catalog, tiling, product-reading
or object-store helper, search the capability index
(https://jejjohnson.github.io/geotoolz/capabilities/) or the installed
packages' `__all__`, and compose what exists. New processing steps are
`pipekit.Operator`s that keep the two contracts, so they chain with
`geotoolz.Sequential` and run in a `pipekit.Graph`:

- **Operator.**
  - Implement `_apply`, never `__call__`.
  - Use a keyword-only constructor that holds configuration only (no
    rasters), so the auto-derived `get_config()` is JSON and
    `Operator.from_state(op.state)` round-trips.
  - Set `forbid_in_yaml` on live objects, and `_terminal` when the step
    returns a non-raster.
  - Use pipekit's `Retry` / `Try` / `Branch` / `Cache` / `ThreadMap`
    instead of writing your own.
- **GeoTensor.**
  - Bands are on axis -3: `(C, H, W)` or `(T, C, H, W)`.
  - A GeoTensor in returns a GeoTensor on the same transform and CRS; a
    plain array in returns a plain array; the input is never mutated.
  - Every output gets a fresh `attrs` dict, with per-band keys matching
    the output bands and band names under `band_names`.
  - `fill_value_default` is nodata: `False` for masks, `0` for labels,
    `NaN` for new floats.
  - Several rasters must share a grid; mask `True` means drop.
```

The `build-geo-pipeline` skill in the plugin below spells the contracts out,
with a worked operator and a self-check.

## Working on the stack itself

Contributors (and their agents) follow [`AGENTS.md`](https://github.com/jejjohnson/geotoolz/blob/main/AGENTS.md)
in the repository: the package map, "reuse before you write", per-package
rules, and recipe skills for adding operators, product readers, catalog
sources and patcher adapters.
