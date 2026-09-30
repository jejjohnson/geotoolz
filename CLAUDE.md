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

### geotoolz family layout

Every operator family (`radiometry`, `indices`, `compositing`, `segment`, …)
has the same shape, enforced by `tests/test_geotoolz.py::test_family_layout`:

```
geotoolz/<family>/
├── __init__.py         # re-exports only — no def / class
└── _src/
    ├── array.py        # Tier A: pure numpy primitives, jaxtyping-annotated,
    │                   # no GeoTensor / metadata / operator state (may be small)
    ├── operators.py    # Tier B: every pipekit.Operator of the family; judges
    │                   # nodata, calls Tier A, rewraps via wrap_like
    └── <topic>.py      # optional: constants / lookup tables / non-Operator
                        # helpers (qa/_src/scl.py, radiometry/_src/solar.py,
                        # io/_src/errors.py, learn/_src/estimators.py)
```

- A primitive lives in `array.py` and is exported from the family
  `__init__` (Tier-A names stay out of the top-level `gz.*` namespace; every
  public Operator class is top-level, including subnamespaces such as
  `geom.coregister`, except the justified few listed in the
  `geotoolz/__init__.py` docstring). Each public name has one home family —
  no cross-family re-exports. Enforced by
  `tests/test_geotoolz.py::test_public_operators_exported` and
  `::test_one_home_per_public_name`.
- Plumbing shared by two or more families (band resolution, nodata,
  rewrapping, labelling, sample flattening, …) goes in `geotoolz/_src/`
  (`bands.py`, `valid.py`, `wrap.py`, `labels.py`, `samples.py`, …). A family
  may call another family's primitive or operator when that *is* the maths
  (`compositing` → `indices.ndvi`, `plume` → `segment` thresholding,
  `viz` → `radiometry` stretches) rather than re-implementing it inline.
- `learn/_src`: `array.py` axis bookkeeping, `estimators.py` the
  non-Operator `GeoTensorEstimator` adapter, `operators.py` every Operator
  (`SklearnOp`, the `Pixelwise*` wrappers, `ModelOp`).
- Not families: `readers/` (sensor-reader framework — `_src/` plus public
  per-sensor subpackages such as `readers.toy_sensor`), the top-level
  `patch_ops.py` geopatcher bridge (optional `[patch]` extra).
- Removals and renames are outright — no deprecated aliases or shim modules
  (`tests/test_geotoolz.py::test_removed_modules_are_gone`).
- Constructors do no network I/O: downloads (e.g. the Natural Earth
  masks) are deferred to the first call.

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

### Operator parameter vocabulary

Every `pipekit.Operator` constructor is **keyword-only** — `__init__(self, *, ...)`,
with no exception for a wrapped estimator / model / patcher
(`PixelwisePCA(estimator=PCA())`, `ModelOp(model=net)`, `GridSampler(patcher=p)`).
One concept has one parameter name across every family:

| Concept | Name | Notes |
|---|---|---|
| Band axis position | `axis: int` (default `-3`) | Only ever the band axis — never a reduction or an orientation (`segment`'s skimage wrappers, formerly `channel_axis`, default to `0` of their per-frame `(C, H, W)` input). |
| Reduction axes | `reduce_axes` | Tuple of axes (or `None` = global): `radiometry.PercentileClip`, `viz.StretchToUint8`, the `normalize` primitives, `radiometry.dos1`. |
| Orientation | `direction` | `"column"` / `"row"` (`restore.DestripeColumn`), `"scan"` / `"sample"` (`geom.SegmentStitch`). |
| One band | one `BandRef` name per band (`red`, `nir`, `swir1`, `qa_band`, `band`, …) | Integer position *or* band name; no `*_idx` twins. Tier-A primitives take integer `*_idx` positions. |
| Several bands | `bands` | List of `BandRef` (`spectral.SelectBands`, `viz.Composite`). `io` readers keep rasterio's 1-based file `indexes`. |
| Value written into pixels | `fill_value` | The carrier attribute stays georeader's `fill_value_default`; a strategy string is `strategy` (`restore.ReplaceOutliers`). |
| RNG seed | `seed` | |
| Neighbourhood side length | `window` | Pixels (odd int, or `(h, w)`): despeckle, destripe, `MedianDenoise`, `NLMeans`, `CLAHE`, `AdaptiveWindowBackground`, `SpectralSmoothing`. |
| Neighbourhood half-width / distance | `radius`, `search_radius` | Not a window (radius `r` ≈ window `2r + 1`): `mask.BufferMask` (with `unit`), `restore.GapFillIDW`, `restore.NLMeans`, `viz.AnnotatePoints`. |
| Output size | `size` | Crop / tile / chip size (`augment.RandomCrop`, `geom.Tile`, `geom.SlidingWindow`, `patch_ops` samplers) — not a window. `io.ReadWindow(window=...)` is a rasterio pixel window, not a size. |
| Gaussian scale | `sigma` | `segment.Quickshift(kernel_size=...)` keeps skimage's name: a kernel *width*, not a window. |
| Areas | `min_area_px`, `max_hole_area_px`, `min_area_m2` | Unit suffix is mandatory. |
| Connectivity | `connectivity: 4 \| 8` | Converted internally for skimage / scipy (`1` / `2` is rejected). |
| Wavelengths | `wavelengths` (nm) | Also the `attrs["wavelengths"]` key; qualified variants `source_wavelengths` / `target_wavelengths`. |
| Percentile stretch bounds | `lower` / `upper` | |
| QA selection | `qa_band`, `bits`, `values`, `targets` | `targets` = `SENSOR_QA_REGISTRY` names. |
| Denominator stabiliser | `eps` (default `1e-10`) | |
| scikit-learn wrappers | `Pixelwise*` | `PixelwisePCA`, `PixelwiseKMeans`, … never shadow the sklearn class they wrap. |

`tests/test_operator_contract.py::test_constructors_are_keyword_only` and
`::test_constructor_vocabulary` enforce the style and reject the retired spellings.

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed). Track work via GitHub issues instead.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
