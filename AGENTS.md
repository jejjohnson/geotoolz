# AGENTS.md

Standing instructions for **every** coding agent working in this repository
(Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth:
`CLAUDE.md` and `.github/copilot-instructions.md` point here, and each package
adds its own rules in `packages/<package>/AGENTS.md`.

## What this repo is

`geotoolz` is a uv workspace of five Python packages: a remote-sensing stack on
`georeader.GeoTensor`, with the Operator / Sequential / Graph composition core
from [`pipekit`](https://github.com/jejjohnson/pipekit). People build bigger
things on top of it, so **the packages are primitives**: new code should
compose what is here, and anything genuinely new should land where the next
person will find and reuse it.

| Package (dist) | Import | Use it for | Depends on |
|---|---|---|---|
| `geotoolz` | `geotoolz` | Operator families on `GeoTensor` (radiometry, indices, mask, geom, viz, learn, …) and the `patch_ops` bridge | georeader, pipekit; `[patch]` → geotoolz-patcher |
| `geotoolz-patcher` | `geopatcher` | Split a field into patches, process, stitch back (geometry × sampler × window × aggregation) | georeader; `[cog]` → geotoolz-cloud; `[pipekit]` |
| `geotoolz-catalog` | `geocatalog` | Find, index, join, load and stage files (`GeoSlice` queries, GeoParquet) | geopandas / pyarrow; `[patch]` → geotoolz-patcher |
| `geotoolz-cloud` | `geocloud` | Object storage: the one obstore client pool, async COG reads | georeader, obstore |
| `geotoolz-products` | `geoproducts` | Readers for EO products (GOES, Himawari, CarbonMapper, …) → `GeoData` / `GeoTensor` | georeader; `[obstore]` → geotoolz-cloud; `[operators]` → geotoolz |

Dependencies point one way and every cross-package link is an optional extra:
`geotoolz` never depends on `geoproducts`, the patcher never on `geotoolz`,
and no package depends on `geotoolz-catalog`.

## Reuse before you write

Before writing a helper, an adapter or a reader, find out whether it exists:

1. **Search the capability index.**
   [`docs/capabilities.md`](docs/capabilities.md) lists every public name in
   the five packages, once, at its home, with a one-line summary; each
   package's `__init__.py` docstring maps its namespaces.
2. **Search the private toolkits.** Shared plumbing lives in `_src/` modules
   that every family or reader of that package uses (the table below, the
   end of the capability index, and each package's `AGENTS.md`). A package
   never imports another package's `_src` — ruff's `TID251` rejects it.
3. **If it is missing, add it to the shared home** — the package's `_src/`
   toolkit, or the lowest package every caller already depends on — not inline
   in the one module that needs it today.
4. **One object, one home.** Each public object is exported from exactly one
   canonical module, under one name. Two packages never export the same name
   for different things.
   - Qualified namespaces may reuse a name for parallel things:
     `geopatcher.spatial.aggregation.Mean` and
     `geopatcher.temporal.aggregation.Mean`, or each sensor's `Reader` and
     `BANDS`.
   - Renames and removals are outright: no aliases, no shim modules.

| You need… | Use |
|---|---|
| An obstore client / byte-range read on `s3://` `gs://` `az://` `https://` | `geocloud.store` (`get_obstore`, `get_range_bytes`) — building obstore stores elsewhere fails lint (`TID251`) |
| Credentials for a bucket / container / host (anonymous, SAS, keys, a TOML file), the GDAL options for them, masking secrets in messages | `geocloud.credentials` (`set_credentials`, `load_credentials`, `gdal_access`, `redact`) — never set credentials in `os.environ` |
| Listing, reading, writing, downloading, uploading, copying, syncing or pre-signing objects (cloud or local) | `geocloud.files` (`ls`, `read_bytes`, `download`, `upload`, `copy`, `sync`, `rm`, `sign`); a stand-in store for tests, `geocloud.store.mount` |
| Windowed or async reads from a Cloud-Optimized GeoTIFF | `geocloud.cog.CogSource`; as a patcher field, `geopatcher.fields.CogField` |
| HTTP with retries, `Retry-After`, streaming downloads (in a reader) | `geoproducts._src.net` (`retrying`, `fetch_bytes`, `download_url`) |
| Listing / downloading public S3 buckets without credentials | `geoproducts._src.s3` |
| Atomic file writes, owner-only credential files | `geoproducts._src.files`, `geoproducts._src.credentials` |
| Lon/lat bbox validation, UTC times for an API query | `geoproducts._src.query` |
| NetCDF-4 / HDF5 variables with CF scale/offset/fill | `geoproducts._src.hdf` (`PackedGridReader`, `unpacked`) |
| A geostationary fixed grid (GOES, Himawari, MTG, SEVIRI) | `geoproducts._src.geostationary` (`FixedGrid`, `geos_crs`) |
| Several readers on one grid | `geoproducts._src.stack.stack` |
| Band lookup by index or name, wavelengths | `geotoolz._src.bands` (`resolve_band`, `resolve_bands`) |
| Nodata handling | `geotoolz._src.valid` |
| Wrapping a result back into a `GeoTensor` | `geotoolz._src.wrap.wrap_like` |
| The carrier helpers above from another package or a downstream project | `geotoolz.carrier` (the same objects, public); check an operator with `geotoolz.testing.check_operator` |
| A spatial index of files, a `GeoSlice` query, loading a mosaic | `geocatalog` (`build`, `query`, `load`) |
| Tiling, overlap-blending, stitching | `geopatcher` (`SpatialPatcher` + `spatial.*` axes) |
| Optional-dependency imports with an install hint | the package's `_src` extras helper (`geotoolz._src.optional`, `geoproducts._src.extras`, `geopatcher._src._extras`, `geocatalog._src._extras`, `geocloud._src.extras`) |
| Retry, fallback, branching, first-that-works | `pipekit` `Retry`, `Try`, `Branch` / `Switch`, `Coalesce` |
| Caching, side-effect taps, QC assertions | `pipekit` `Cache` / `Memoize`, `Tap` / `Sink` / `Snapshot`, `AssertShape` / `AssertDType` / `Quarantine` |
| Mapping an operator over many items | `pipekit` `ThreadMap` / `ProcessMap` / `AsyncMap` / `BatchedMap`; over patches, `geopatcher.run` |
| Saving and reloading a pipeline | `pipekit` `dumps` / `loads` (`loads_sandboxed` for untrusted input) |

## The two contracts

Everything in the stack composes because it keeps two contracts: pipekit's
`Operator` and georeader's `GeoTensor`. Code that breaks either runs fine
alone and fails the moment someone puts it in a pipeline, so treat both as
hard rules. The long form, with examples, is
[`docs/concepts.md`](docs/concepts.md) and
[`docs/recipes/define-an-operator.md`](docs/recipes/define-an-operator.md).

### pipekit `Operator`: how pieces compose

- **Subclass `pipekit.Operator` and implement `_apply`.** Never override
  `__call__`. It runs `_apply` on a value, and on an `Input` / `Node` it
  records a graph node, so the same operator works eagerly, in a
  `Sequential` (`a | b`) and in a `Graph`.
- **Keyword-only constructor that holds configuration only.** Use
  `__init__(self, *, ...)` and store each argument under its own name.
  Hold scalars, band references, nested operators and similar settings,
  never a carrier. A second raster is a positional `_apply` argument, so a
  `Graph` can wire it in.
- **Let `get_config()` derive itself.** `ConfigMixin` builds it from the
  constructor. It must be strict JSON, and `Operator.from_state(op.state)`
  must rebuild an equal operator. Don't write a `get_config` that restates
  the constructor. Keep runtime-only attributes out with
  `__config_exclude__`.
- **Set `forbid_in_yaml = True` on anything holding a live object** (a
  callable, model, estimator, open handle or RNG). Its config is then a
  debug repr only, and `from_state` refuses to rebuild it.
- **Carrier in, carrier out.** An operator that returns anything else (a
  table, a number, `None`) sets `_terminal = True`, so `Sequential` accepts
  it only as the last step. A fan-out that returns a list of carriers is
  the documented exception.
- **Learned state uses fit / transform.** `fit(x) -> self` writes
  trailing-underscore attributes (`mean_`), and `transform` is read-only.
  `_apply` never overwrites constructor attributes. Use
  `geotoolz._src.fitted.fit_once` to fit on first call without races.
- **Don't rebuild what pipekit already has.** Control flow, caching,
  observation, QC, parallel maps, carry-state (`StatefulOperator`) and
  serialisation are all in pipekit (see the table above). pipekit is
  carrier-agnostic: a GeoTensor-specific need belongs in geotoolz, and a
  generic one upstream in pipekit.

### `GeoTensor`: what flows between them

`georeader.GeoTensor` is an `np.ndarray` subclass that carries
`transform`, `crs`, `fill_value_default` and `attrs`.

- **Shape.** Dimensions are `("time", "band", "y", "x")[-ndim:]`, so a
  carrier is 2-D to 4-D. The spatial axes are the last two, and the band
  axis is always `-3`, never `0`.
  - Handle a `(T, C, H, W)` stack frame by frame
    (`geotoolz._src.shape.map_frames` / `over_frames`), or reject it with
    an error that names the operator.
  - A result that collapses the bands keeps a singleton band axis,
    `(T, 1, H, W)`.
- **Both carriers in, same carrier out.** Read the data with
  `np.asarray(x)`. A GeoTensor input returns a GeoTensor with the input's
  georeferencing, and a plain array returns a plain array.
  - Rewrap with `geotoolz._src.wrap.wrap_like`. Pass `transform=` for an
    output on a new grid.
  - Inside an operator, never build a `GeoTensor` by hand, and never
    mutate the input.
- **Fresh, consistent `attrs`.** Every output gets a new dict, never the
  input's. Each per-band key (`band_names`, `descriptions`, `wavelengths`,
  …) has one entry per output band, or is dropped; `wrap_like` does this
  for you. New band names go under `band_names` only.
- **Nodata.** `fill_value_default` *is* nodata: georeader's `validmask()`
  is `values != fill`.
  - A pixel is invalid when any of its bands is non-finite or equals the
    fill (`geotoolz._src.valid`).
  - Leave invalid pixels out of statistics and fits, then write the
    output's fill back into them.
  - The output fill follows the output's dtype and meaning: `False` for
    masks, `0` for labels and counts, `NaN` for new float quantities, and
    `carried_fill` for outputs that carry the input's values.
- **Bands.** A band is referenced by integer position or by name. Resolve
  it with `geotoolz._src.bands.resolve_band`, which looks names up in
  `band_names`, then `descriptions`, then `bands`.
- **Masks.** `True` means drop the pixel. Region masks pick the side that
  survives with `keep=`.
- **Several carriers.** Carriers are positional and must share a pixel
  grid: the same `(H, W)`, the same CRS and an exactly equal transform
  (`geotoolz._src.geo.require_grid_match`). An operator that measures in
  metres requires a projected CRS (`require_projected_crs`).
- **Readers.** A reader returns a `GeoTensor`, or a lazy georeader
  `GeoData` (`transform`, `crs`, `shape`, `load()`, `read_from_window()`).
  - georeader's read helpers and the patcher's `RasterField` accept a lazy
    reader as it is.
  - A geotoolz operator needs pixels: call `.load()` (or use an `io`
    operator) first.

`packages/geotoolz/tests/test_operator_contract.py` walks every operator
class automatically and checks:

- state round-trip and JSON config;
- keyword-only constructors and the parameter vocabulary;
- graph mode;
- terminal outputs;
- carriers are never constructor arguments;
- fresh `attrs` and fills that match the dtype;
- `(T, C, H, W)` stacks.

A new operator is covered as soon as it exists. If it can't be built with no
arguments, add it to `CTOR_KWARGS` (or `RUNTIME_CTOR_KWARGS` when it needs
live objects).

Outside geotoolz — another package of this repo, or a project built on the
stack — the same helpers are public in `geotoolz.carrier`, and
`geotoolz.testing.check_operator(op, scene)` runs the same checks (they share
`geotoolz._src.contract`) on one operator.

The other contracts each belong to one package, and their rules live in
that package's `AGENTS.md`:

- geopatcher's `Field`;
- geocatalog's `GeoSlice`;
- geoproducts' `ProductReader`.

## Package rules

Each package's `AGENTS.md` holds its layout, conventions and the tests that
enforce them. Read it before changing that package:

- [`packages/geotoolz/AGENTS.md`](packages/geotoolz/AGENTS.md) — operator families, two-tier layout, parameter vocabulary
- [`packages/geotoolz-patcher/AGENTS.md`](packages/geotoolz-patcher/AGENTS.md) — axes, `Field` contract, config envelopes
- [`packages/geotoolz-catalog/AGENTS.md`](packages/geotoolz-catalog/AGENTS.md) — namespaces, `GeoSlice` contract, lazy extras
- [`packages/geotoolz-cloud/AGENTS.md`](packages/geotoolz-cloud/AGENTS.md) — the client pool and the COG engine
- [`packages/geotoolz-products/AGENTS.md`](packages/geotoolz-products/AGENTS.md) — `ProductReader`, sensor subpackages, the reader toolkit

Conventions shared by all five:

- Python 3.12+, `from __future__ import annotations` in every module, type
  hints on every public function, Google-style docstrings with a short
  example for public classes and functions.
- Public API in each package's facades (`__init__.py`, re-exports only);
  implementation in `_src/`.
- Optional dependencies sit behind extras, are imported lazily, and fail at
  the point of use with an install hint naming the extra.
- Constructors do no network I/O; downloads happen on first use.
- Tests that are slow or touch the network carry a marker (`slow`,
  `integration`, `live` — see each package's `pyproject.toml`); the fast tier
  runs in CI.
- Satpy is never a dependency.

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in
`.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any
agent can read and follow them):

| Job | Recipe |
|---|---|
| Add an operator / primitive to geotoolz | `add-operator` |
| Add a reader for a new EO product | `add-product-reader` |
| Add a catalog discovery source | `add-catalog-source` |
| Add a patcher field adapter or axis | `add-field-adapter` |
| Draw or update a docs diagram | `docs-diagram` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `geotoolz-review` (+ the read-only `.claude/agents/reuse-reviewer.md`) |

Downstream users get the stack's guidance through the Claude Code plugin in
`plugins/geotoolz/` (published by `.claude-plugin/marketplace.json`) and
`docs/llms.txt`; see `docs/agents.md`. When the public API or the canonical
pipeline changes, update `plugins/geotoolz/skills/build-geo-pipeline/` too.

## Working in the repo

Always run Python tools through `uv run` (never the system Python).

```bash
make install              # uv sync --all-packages --all-groups --all-extras + hooks
make test                 # fast tier across all five packages
make test-all             # everything incl. slow / integration tiers
make format               # ruff format . && ruff check --fix .
make lint                 # ruff check .   (entire repo)
make typecheck            # ty check, per package
make docs-serve           # local MkDocs preview
```

Run a single test from the owning package directory so its pytest config
(markers, coverage gate) applies:

```bash
cd packages/geotoolz-catalog && uv run pytest tests/test_geoslice.py -v
```

### Before every commit

All of these must pass, from the repo root:

1. `make test` — zero failures.
2. `uv run --group lint ruff check .` — the **entire** repo, not a package
   directory (CI lints every package's `tests/`).
3. `uv run --group lint ruff format --check .`
4. `make typecheck` — ty runs from each package directory.
5. After changing a public API: `make capabilities` (CI checks
   `docs/capabilities.md` is current).

When a change touches packaging, extras or imports, also check what CI's
`base-install.yml` checks (the package with no extras installed), and make
sure every new file is tracked: `git status` must not show it as ignored.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several
   readings, name them instead of picking one silently; ask when unsure.
2. **Simplicity first.** The minimum code that solves the problem — no
   speculative features, no single-use abstractions.
3. **Surgical changes.** Touch only what the task needs; match the existing
   style; remove only what your change made unused.
4. **Goal-driven.** Turn the task into a check (a failing test, a reproduced
   bug) and loop until it passes.

## Git, commits and pull requests

- Never push to or merge into `main` unless explicitly told to. Work on a
  feature branch; push only when asked.
- Commit messages and PR titles follow
  [Conventional Commits](https://www.conventionalcommits.org/) with a
  lowercase subject (`feat(catalog): add …`); CI validates PR titles.
  Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- Releases are cut by release-please per package (`geotoolz-vX.Y.Z`,
  `geotoolz-patcher-vX.Y.Z`, …); don't bump versions by hand.
- Never replace an existing PR title or description; append to it.
- After fixing a review comment, resolve its thread
  (`resolveReviewThread` GraphQL mutation; list thread ids with
  `repository.pullRequest.reviewThreads`). Don't resolve threads you didn't
  address.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md).
- Plans and scratch design notes go in `.plans/` (gitignored, never
  committed); track work in GitHub issues.

## Documentation

MkDocs + Material + mkdocstrings; the root `mkdocs.yml` builds one site with a
section per package. Example notebooks are committed executed (`.ipynb`, see
`.github/instructions/docs-examples.instructions.md`). Diagrams are HTML files
in `docs/assets/diagrams/` rendered to PNG with `render.py` — edit the HTML,
re-render, commit both.
