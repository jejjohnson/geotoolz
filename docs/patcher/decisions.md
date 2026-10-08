# Design Decisions

This page records the locked-in design decisions that shape `geopatcher`'s
public API. Each one was an open question in the design phase; once
decided, the rationale lives here so future contributors can answer "why
this and not that?" without rerunning the discussion.

The format is loose ADR: **Decision** → **Context** → **Consequences** →
**Alternatives considered**. Decisions are numbered in the order they
were locked in; existing decisions do not change without a follow-up
entry that supersedes them.

---

## ADR-001 — `Patcher.split` returns `Iterator[Patch]`

**Decision.** All four patchers (`SpatialPatcher`,
`AsyncSpatialPatcher`, `TemporalPatcher`, `SpatioTemporalPatcher`)
expose `split` as an **iterator**, not a list. Eager materialisation is
one `list(patcher.split(field))` call away when needed.

**Context.** The patcher walks anchors placed by the sampler and reads
each neighborhood out of the field. For large fields the natural mode
is one patch at a time — the field has lazy `Field.select`, so a
generator yields patches as they're read rather than holding them all in
memory.

**Consequences.**

- Streaming is the default. `Patcher.merge` consumes the iterator
  directly; on-disk accumulators (see [ADR-002](#adr-002-disk-backed-aggregations-use-zarr))
  never need the full patch list in RAM.
- `prefetch=N` (jejjohnson/geopatcher#9), `asplit()` (jejjohnson/geopatcher#8), and `max_in_flight` backpressure
  (jejjohnson/geopatcher#16) compose for free — they all wrap the iterator without changing
  the patcher contract.
- The pipekit `GridSampler` operator (`geopatcher.integrations.pipekit`)
  *does* materialise to a list at its operator boundary. That is a
  pragmatic concession to the `Sequential` pipeline shape; callers who
  want streaming inside an operator graph should consume
  `patcher.split` directly, not through `GridSampler`.
- `len(patcher.split(field))` does not work. The equivalent is
  `patcher.n_anchors(field)`, which the sampler can answer without
  touching the field.

**Alternatives considered.**

- *Return a list by default.* Cheaper ergonomics (`len()`, indexing,
  reuse), but forces every consumer to hold all patches at once.
  Equivalent surface area is recovered via `list(patcher.split(field))`
  with no loss; the reverse — making an eager list stream lazily — would
  require an architectural rewrite.
- *Return a `Sequence`-shaped lazy container.* Adds complexity (the
  container must implement `__len__` and `__getitem__` for arbitrary
  geometries, which the sampler doesn't always know how to compute);
  doesn't unlock anything the iterator + `n_anchors()` pair can't.

---

## ADR-002 — Disk-backed aggregations use Zarr

**Decision.** Streaming aggregations that need an out-of-RAM target
(`spatial.aggregation.OverlapAdd(streaming=True, target_path=...)`, future
`spatial.aggregation.InvVarWeightedMean(streaming=True, ...)`, etc.) write to a
**framework-managed Zarr store** by default.

**Context.** The streaming asymmetry (see §4 of the `scaling.md`
design note in the planning archive; not shipped with these docs) is on the
*output* side: the input field already has a lazy `Field.select`, so
input scales as long as `split` returns an iterator. Output
preallocability is the bottleneck. A disk-backed accumulator solves it.

Zarr was picked over memmap, HDF5, and "bring your own store":

- Zarr v3 (`zarr>=3`, the `streaming` extra) is already a hard
  requirement of the streaming `spatial.aggregation.OverlapAdd` implementation;
  users of streaming inference already have it installed.
- Chunked, append-friendly, parallel-writable, plays well with Dask
  and downstream COG conversion (jejjohnson/geopatcher#15) — `writer="cog"` streams through
  a temporary Zarr store and converts it block by block.
- Chunk-wise access keeps the final normalisation O(chunk): the store
  is preallocated and never read back whole.

**Consequences.**

- Default usage is one line: `spatial.aggregation.OverlapAdd(streaming=True,
  target_path="out/", chunks=geometry.size)`. No `import zarr` in user
  code. The chunk shape is required rather than guessed from the first
  patch (a shrunk edge chip would otherwise chunk the whole store).
- A merge refuses to overwrite an existing store unless
  `overwrite=True`.
- A pre-opened-store path (passing a `zarr.Array` for Dask / distributed
  writers) is not implemented; re-open if a concrete need surfaces.
- V3-sharded outputs (`shard_shape=`, jejjohnson/geopatcher#14) and the COG target
  (`writer="cog"`, jejjohnson/geopatcher#15) layer on top of the Zarr default without
  changing aggregation APIs.
- Memmap / HDF5 / parquet targets are out of scope for v0.x. Re-open if
  a concrete user need surfaces.

**Alternatives considered.**

- *NumPy memmap.* Single-file, no chunking, no concurrent writers.
  Loses the path to distributed.
- *HDF5.* Locking story is poor; concurrent writes from multiple
  processes require SWMR mode with caveats; adds a heavy C dependency
  for a feature most users won't need.
- *Always pass in a store.* Friendlier for advanced users, hostile for
  casual ones. The two-form API above gives both.

---

## ADR-006 — `streaming_safe` violations: configurable, warn by default

> Renumbered from ADR-003 — that number had accidentally been assigned
> twice. References to "ADR-003" for the `streaming_safe` /
> `set_strict` decision (e.g. in the `get_strict` docstring) resolve
> here; ADR-003 now refers only to the `MatchedField` decision below.

**Decision.** When a caller passes a `streaming_safe = False`
aggregation into a context that expects streaming (`Patcher.merge`,
streaming `OverlapAdd`, future PatchJournal jobs), the framework emits
a `RuntimeWarning` by default. A module-level toggle promotes the
warning to a hard `RuntimeError` for callers (CI, batch jobs) that want
to fail fast.

Toggle API:

```python
import geopatcher as gp

gp.observe.set_strict(True)      # promotes streaming_safe warnings to errors
gp.observe.set_strict(False)     # back to warn-only (default)
gp.observe.get_strict()          # bool
```

Environment variable equivalent: `GEOPATCHER_STRICT=1` (read once at
import time; runtime `set_strict()` overrides it).

**Context.** Today `_warn_if_unsafe_streaming` always emits a warning.
That is right for interactive notebook work — the user sees the warning
and either ignores it (the in-RAM merge fits fine) or swaps in a
streaming-safe alternative. It is wrong for batch / CI contexts where
silently falling back to RAM defeats the streaming guarantee that
called the job into existence.

Three options were on the table:

1. **Hard error.** Loud, but breaks every quick-iteration use of
   `spatial.aggregation.Median` / `spatial.aggregation.Learned` in a notebook.
2. **Warning only.** What we have. Quiet failures in batch jobs.
3. **Configurable.** Best of both — default-permissive, opt-in strict.

**Consequences.**

- Casual / notebook users see no behavior change.
- Batch / CI users can lock down with `gp.observe.set_strict(True)` (or the env
  var in their orchestration layer).
- Tests that intentionally exercise the warn path continue to work; the
  `_warn_if_unsafe_streaming` helper checks the strict flag first and
  raises before warning.
- Future `streaming_safe` checks elsewhere in the framework (PatchJournal
  registration, COG target compatibility, …) call the same helper and
  inherit the toggle for free.

**Alternatives considered.**

- *Per-call `strict=` argument on `Patcher.merge`.* Adds keyword noise
  to every call site; doesn't help the "global policy for this job"
  case which is the actual ask.
- *Always error.* Too disruptive for the existing user base; would
  require a deprecation cycle for a problem most users do not have.

---

## ADR-003 — `MatchedField` is a composite `Field`, not a new top-level type

**Decision.** Co-located patching across N sources lives in
`geopatcher.matched.MatchedField`, which **satisfies the existing
`Field` Protocol** via its primary's `domain`, plus three matched
patchers (`MatchedSpatialPatcher`, `MatchedTemporalPatcher`,
`MatchedSpatioTemporalPatcher`) that split and merge it. Concretely:

- `MatchedField.primary` is a regular `Field`; its CRS / bounds /
  shape define the anchor space.
- `MatchedField.secondaries: Mapping[str, Field]` carries the
  matched sources keyed by name.
- `MatchedField.coreg: Mapping[str, Callable]` carries one
  coregistration callable per secondary, called as
  `coreg[name](raw_secondary, primary_data)`. Type is the broad
  `Callable[[Any, Any], Any]`, but the **recommended** value is a
  `pipekit.Operator` from `geotoolz.geom.coregister.*` so the
  alignment step round-trips through YAML.
- `MatchedField.select(indexer)` returns a plain
  `dict[str, data]` — the primary's read under `"primary"`
  (`MatchedPatch.PRIMARY_KEY`), each secondary's coregistered data
  under its name. For raster domains each secondary is read over the
  primary chip's **geographic footprint** (the primary window's bounds
  mapped onto the secondary's own grid, reprojected when the CRSs
  differ, rounded outward), so a secondary on a different resolution,
  origin or CRS hands the coreg callable a chip covering the whole
  footprint. Non-raster secondaries are read with the primary's
  indexer and must share its index space.
- The matched patchers turn that dict into the sibling carriers
  `MatchedPatch` / `MatchedTemporalPatch` / `MatchedSpatioTemporalPatch`
  (not subclasses of `Patch`): one member patch per source under
  `members[name]`, a per-source `valid_mask` (False where a member
  equals its carrier's nodata — `fill_value_default` / `rio.nodata` —
  or, for floats, is NaN / ±inf) and per-source `weights`.
- `merge` is per source: each patcher returns `dict[str, output]`,
  every value the source's raw aggregation output on the **primary's**
  grid (a bare `np.ndarray` for the dense aggregations, ADR-007);
  `MatchedSpatialPatcher.merge_to_field` wraps each one through the
  primary's `with_data`.

**Context.** The cross-package query→matchup→patch design
(`docs/patcher/design/query-matchup.md`) introduces matchups between
LEO, GEO, vector, and point-cloud sources. The patching side has to
read co-located neighborhoods across these heterogeneous sources
without duplicating coregistration logic (which lives in `geotoolz`)
and without forcing geopatcher's framework-free core to depend on
`pipekit`.

The composite-Field approach satisfies all three constraints: every
existing sampler, geometry and window places anchors and indexers on
a `MatchedField` unchanged (they only see the primary's domain); the
heavy alignment work lives in `geotoolz.geom.coregister.*`
`pipekit.Operator`s; geopatcher's only new typing dependency is the
standard-library `Callable` (since `pipekit.Operator` IS callable).

**Consequences.**

- A user with no matchup needs continues to write `SpatialPatcher`
  pipelines against a `Field` — nothing changes.
- A user with matchups writes `MatchedField(primary, secondaries,
  coreg)` and wraps the `SpatialPatcher` they already use in a
  `MatchedSpatialPatcher(primary=..., secondary_aggregators=...)`
  (or the temporal / spatio-temporal mirrors). **The matched patcher
  is required for both split and merge**: a plain
  `SpatialPatcher.split(matched_field)` yields `Patch`es whose `data`
  is the per-source dict (no masks), and `SpatialPatcher.merge` fails
  on that dict because the aggregations expect one numeric array; a
  plain `SpatioTemporalPatcher` cannot slice the dict along time.
- The matched patchers reuse the primary patcher's sampler, geometry,
  window, aggregation, `on_error` policy (which covers every source's
  read and coregistration), hooks, prefetch, journal and
  `max_in_flight`; `MatchedSpatialPatcher.split(cache=...)` caches each
  source's raw read under its own `PatchCache` identity. The only new
  configuration is `secondary_aggregators: {name: aggregation}`;
  omitting a secondary skips it on merge.
- Per-source outputs live on the primary's grid, because the
  coregistration mapped each secondary there. Getting a secondary back
  onto its own grid means inverting the coregistration, which is the
  caller's job.
- `MatchedPatch` is intentionally **not** a subclass of `Patch`:
  `Patch[AnchorT, IndicesT, DataT]` is parameterised over a single
  data type, but `MatchedPatch` holds a heterogeneous dict of
  patches whose types differ across keys. Consumers that don't
  care about matchups continue to type against plain `Patch`;
  consumers that do explicitly type against `MatchedPatch`.

**Alternatives considered.**

- *A dedicated `CoregistrationStrategy` ABC in geopatcher.* Would
  duplicate the operators already present in
  `geotoolz.geom.coregister`, force geopatcher to import or
  reimplement reprojection / rasterization / KDTree binning, and
  break the "core is numpy + scipy only" invariant. Rejected;
  geotoolz owns the coreg logic.
- *Make `MatchedPatch` a subclass of `Patch`.* Liskov-substitution
  surprises: a consumer that types `Patch` and unpacks `data`,
  `anchor`, `indices`, `weights` would break on a `MatchedPatch`
  because there is no single `data`. Sibling carrier sidesteps the
  whole question.
- *Type `coreg` as `Mapping[str, pipekit.Operator]`.* Imports
  pipekit into geopatcher's runtime, breaking the framework-free
  core. Rejected; the broader `Callable` is enough.
  `pipekit.Operator` users are still first-class — they're just
  not the only allowed value.
- *A separate `MatchedPatcher` family alongside `SpatialPatcher` /
  `TemporalPatcher` with its own samplers, geometries, windows and
  aggregations.* Would mean rewriting every axis to accept matched
  fields. The shipped matched patchers are thin wrappers instead: they
  drive the primary patcher over the composite field and only add the
  per-source unpacking and merge fan-out.

---

## ADR-004 — Coordinate-aware temporal patching is opt-in; stride-1 only in v0.1

**Context.** The temporal stack works in integer index space:
`temporal.sampler.Sampler.anchors(time_len) → Iterable[int]`, `temporal.geometry.Geometry.window(
time_len, anchor) → slice`. This is correct and fast for in-memory arrays at a
known cadence. It breaks down for ARCO-ERA5-style workloads where the natural
specification is *physical* — "a 9-hour lookback at the source cadence,
whatever that is" — and the data cadence is a property of the store, not of
the caller. A `TimeStencil`-based layer (ported from `neuralgcm/terrax`)
expresses windows in coordinate units and validates that the requested step
exactly tiles the source grid.

**Decision.** Extend the temporal protocol via two ClassVar capability flags:

- `temporal.geometry.Geometry.needs_coord: ClassVar[bool] = False` (default).
- `temporal.sampler.Sampler.needs_coord: ClassVar[bool] = False` (default).

Coordinate-aware subclasses (`temporal.geometry.StencilGeometry`,
`temporal.sampler.StencilSampler`) set the flag to `True`. `TemporalPatcher` reads the
flag from both components; when either is `True`, every public method that
takes `series` (`split`, `asplit`, `patches_at`, `anchors`, `n_anchors`) also
requires a `coord=` keyword: a 1-D monotonic-ascending coordinate array along
`time_axis`. Missing `coord=` raises `ValueError` at the entry point — *before*
the sampler is invoked — so mis-wiring fails loudly.

Dispatch inside `_patches_for_anchor`:

- If `geometry.needs_coord`, call `geometry.window_coord(coord, anchor_idx)`
  and expect a contiguous `slice(start, stop)`.
- Otherwise, call the existing `geometry.window(time_len, anchor)`.

The sampler always returns `int` anchors (indices into `coord`, not coordinate
values), so the rest of `_patches_for_anchor`, the patch carrier, and the hook
contract are byte-identical to the integer path.

**v0.1 stride-1 constraint.** `temporal.window.Window.weights(geometry, length)` and
`temporal.aggregation.Aggregation.merge(patches)` both assume contiguous integer index
ranges (`s.stop - s.start` is the realised window length). A stencil with
`step > source_step` would yield a strided slice, silently breaking both.
`temporal.geometry.StencilGeometry.__post_init__` raises when `source_step` is supplied
and `stencil.step / source_step != 1`; `window_coord` re-checks the resolved
slice's stride at resolve time as a belt-and-braces guard for callers that
omit `source_step`. Strided reads are deferred to v0.2 — they need
`temporal.window.Window.weights` and `temporal.aggregation.Aggregation.merge` to take a
realised-length argument.

**Hook payload extension.** `PatcherHook.on_patch_start` and `on_patch_done`
gain an optional trailing `coord_value` (the resolved `coord[anchor]`, or
`None` for the integer path). `_dispatch` trims trailing args to the
callback's positional arity so pre-extension hooks written as
`on_patch_start(self, anchor)` keep working without `RuntimeWarning`s.

**Consequences.**

- Integer pipelines unchanged. All pre-existing samplers, geometries,
  windows, and aggregations inherit `needs_coord = False`; their patcher
  dispatch path is byte-identical.
- Coordinate-aware pipelines are explicit and discoverable: the
  `temporal.geometry.StencilGeometry`/`temporal.sampler.StencilSampler` classes carry the flag,
  and the patcher's error message names `coord=` as the required argument.
- Cadence-independence: the same `TimeStencil('-9h', '3h', '3h')` against
  any 3-hourly source produces the same 5-point window. Re-pointing the
  notebook at a 1-hourly store raises at construction (stride > 1) rather
  than silently producing a shorter window.
- `cftime`-typed coords are out of scope; `XarrayField.time_coord` raises a
  typed `TypeError` pointing at the conversion path.

**Alternatives considered.**

- *Overload `geometry.window(time_len, anchor)` to accept an optional
  coord.* Conflates two coordinate systems on one method, makes mixed
  integer/coord pipelines harder to reason about, and forces every existing
  geometry to know about coordinate space.
- *Have the sampler emit `datetime64` origins directly.* Forces the patcher
  to convert at every call site and changes the public sampler protocol's
  return type. Keeping `anchors → Iterable[int]` lets the rest of the
  pipeline stay integer-only.
- *Drop strided stencils to be permissive.* Would make `window` /
  `aggregation` correctness subtle and dependent on the stencil shape.
  Better to raise loudly and ship a real fix in v0.2.

**See also.** GitHub issue jejjohnson/geopatcher#56 (the design doc and tracking issue for this
work); the upstream `neuralgcm/terrax` `xreader.stencils` module (Apache-2.0,
© Google LLC) from which the stencil math was ported.

---

## ADR-005 — Random access is a Sequence wrapper, cache lives on the view

> Superseded in part by ADR-008: `IndexedPatchView` now lives in
> `geopatcher.run`, not at the root.

**Context.** The patcher's canonical surface is
`SpatialPatcher.split → Iterator[Patch]` (ADR-001). xrpatcher's
`XRDAPatcher[i]` API gives random-access by integer index, and `xrpatcher`
also bundles in-memory caching (`cache=True`/`preload=True`) for ML
loaders that re-read the same anchors per epoch. Migrants want the same
ergonomics without losing the iterator-first contract; the question is
*where* the random-access surface lives and *what protocol* it speaks.

**Decision.** Add `geopatcher.run.IndexedPatchView`:

- A stdlib `collections.abc.Sequence[Patch]` over a `(patcher, field)`
  pair. Supports `len(view)`, `view[i]`, slicing, negative indexing,
  `for p in view`.
- No torch / Grain / jax dependency — `Sequence[Patch]` is the protocol.
  Framework wrappers (`torch.utils.data.Dataset.__getitem__`,
  `grain.RandomAccessDataSource.__getitem__`) are one-liners over the
  view; the recipes demonstrate.
- Constructor flags `cache=True`/`preload=True` mirror xrpatcher
  one-for-one. The cache is indexed by integer (the simplest possible
  scheme that matches xrpatcher). `preload=True` calls
  `patch.data.load()` / `.compute()` via duck-typing — works for xarray
  `DataArray`, dask arrays, and is a no-op for numpy.
- The cache lives on the view, not on the patcher. Reason: a user can
  have two views on the same patcher with different cache settings
  (e.g. training with `preload=True`, eval with `cache=False`), and the
  patcher itself stays a frozen value object.
- The view materialises the anchor list at construction time
  (`patcher.anchors(field)`) — no lazy re-walking on every `__getitem__`.

Alongside the view, the PR ships three sympathetic conveniences:

- `spatial.sampler.RegularStride(check_full_scan=True)` raises
  `IncompleteScanConfiguration` at anchor time when `(length - size) %
  step` is nonzero on any axis. Off by default to preserve existing
  silent-truncation semantics; opt in for the xrpatcher
  strict-tiling story. Spatial analogue of `exact_quotient` (ADR-004).
- `XarrayField.coords_per_patch(patches)` returns one coord-only
  `xr.Dataset` per patch (xrpatcher's `get_coords()` equivalent).
- `SpatialPatcher.merge_to_xarray(patches, field)` returns a
  `DataArray` with the original coords intact — wraps `merge` +
  `field.with_data` so xrpatcher migrators don't have to discover the
  two-step pattern.

Precursor change: `XarrayField.select` now returns the bare
`xarray.DataArray` rather than another `XarrayField`. This brings it in
line with `RasterField.select → GeoTensor` (select returns the natural
data payload, not another field wrapper) and unblocks
`spatial.aggregation.OverlapAdd.merge`, which `np.asarray`'s every patch's data.

**Consequences.**

- **Iterator-first split stays canonical.** ADR-001 is unchanged. The
  view is a wrapper, not a replacement; consumers that prefer
  iterators see no change.
- **`from geopatcher import IndexedPatchView` works.** Root re-export
  + `__all__` entry, alongside `SpatialPatcher`, `TimeStencil`, etc.
- **No framework adapter packages.** Following the same stance as
  ADR-004 and PRs jejjohnson/geopatcher#41 / jejjohnson/geopatcher#57, we ship primitives (the Sequence-shaped
  view) plus recipes (one-line torch / Grain wrappers), not adapter
  classes.
- **Content-addressed cache is still future work.** This PR's
  index-keyed in-memory cache is the cheap xrpatcher port; the deeper
  cross-session content-addressed cache is tracked at jejjohnson/geopatcher#24. The
  `IndexedPatchView` is the natural home for it.
- **`xrpatcher` can be archived.** After this PR ships in a tagged
  release, `XRDAPatcher` users can swap one import:
  `XRDAPatcher(da, patches=..., strides=...)` →
  `IndexedPatchView(SpatialPatcher(...), XarrayField(da))`. See
  `recipes/xarray-nd-patching.md` for the side-by-side.

**Alternatives considered.**

- *Bundle a torch `Dataset` subclass in `geopatcher`.* Drags torch
  into core deps for every user. Same primitives-not-adapters stance
  the user took on jax and torch in PRs jejjohnson/geopatcher#11 / jejjohnson/geopatcher#23 (and the temporal
  stencils work in jejjohnson/geopatcher#57).
- *Put the cache on `SpatialPatcher` itself.* Conflates the patcher's
  value-object identity with mutable per-loader state. A user with
  one patcher + two loaders (train, eval) would have to re-construct
  the patcher to vary cache settings.
- *`view[i]` lazily walks the sampler each call.* Quadratic in
  pathological samplers (random with replacement, Poisson-disk on
  large domains). Materialising at construction is the standard
  random-access trade-off.
- *Flag `check_full_scan` on by default.* Breaking change — existing
  pipelines that intentionally tile partially would start raising.
  Opt-in matches the additive-only convention of the rest of the
  framework.

**See also.** GitHub issue jejjohnson/geopatcher#60 (the design doc and tracking issue for
this work); the upstream `xrpatcher` (the migration target);
[`recipes/xarray-nd-patching.md`](recipes/xarray-nd-patching.md) for
side-by-side migration; jejjohnson/geopatcher#24 for the deeper content-addressed cache;
ADR-001 (iterator-first split) and ADR-004 (coordinate-aware temporal).

---

## ADR-007 — `merge` returns the raw aggregation output; `merge_to_field` rebuilds the Field

> Partly superseded by ADR-009: the output's band axes come from the
> patches, and `merge_to_field` accepts a changed band count on raster
> fields.

**Decision.** `SpatialPatcher.merge(patches, domain)` keeps returning
whatever the aggregation's `merge` produces — a bare `np.ndarray` on the
domain grid for the dense aggregations, a `dict` for `spatial.aggregation.MeanStd` /
`spatial.aggregation.InvVarWeightedMean` / `spatial.aggregation.ByIndex`, a zarr array for
streaming `spatial.aggregation.OverlapAdd`. A sibling,
`SpatialPatcher.merge_to_field(patches, field)`, returns
`field.with_data(merged)`: a `GeoTensor` for `RasterField` (transform,
CRS, `fill_value_default` and `attrs` of the source), a `RioXarrayField`
/ `XarrayField` wrapping the rebuilt `DataArray` for the xarray
adapters. It raises `TypeError` for a `dict` output, a non-array output,
or an array whose shape is not `field.domain.shape`.
`MatchedSpatialPatcher.merge_to_field(patches, mfield)` does the same per
source, wrapping every source through the primary's `with_data` (each
secondary was coregistered onto the primary's grid). The temporal and
spatio-temporal patchers have no field-shaped merge output and get no
`merge_to_field`.

`merge_to_field` and `merge_to_xarray` share one dtype rule: the merged
values are cast back to the source dtype when every value is
representable in it — finite, integral and in range for an integer /
bool source, no finite overflow for a floating one. Otherwise the
aggregation's dtype is kept (a fractional mean or a NaN fill on a
`uint16` source stays float64).

**Context.** The `Field` protocol docstring said `with_data` was "used by
`Aggregation.merge` to rebuild a global field", and the matched-patcher
docstring said merge returns "typically a `GeoTensor`". Neither was
true: every aggregation returns a bare float64 array and only
`merge_to_xarray` called `with_data` (#194). Two fixes were on the
table — make `merge` wrap its output through `with_data` (option A in
#194), or keep `merge` raw and fix the docs (option B).

**Consequences.**

- Non-breaking: `merge`'s return type is unchanged, so `patch_ops`,
  notebooks and anything that `np.asarray`s the result keep working.
- A georeferenced result is one call away (`merge_to_field`) instead of
  a manual `field.with_data(...)` that every downstream consumer had to
  rediscover. The protocol and matched docstrings now say who calls
  `with_data`.
- The same change routed `reduce` / `two_pass` through `split` (so
  `on_error`, hooks, journal, cache, prefetch and backpressure apply and
  `reduce` runs the strict streaming check), made `amerge` stream an
  async iterable into the aggregation through a thread adapter instead
  of materialising it, and attributed the `streaming_safe` warning to
  the caller's line.

**Alternatives considered.**

- *`merge` wraps through `field.with_data` (option A).* `merge` only
  receives a `domain`, not the `Field`, so it would need a signature
  change, and it would change the return type of every existing call
  site. Pre-1.0 we could break it, but an additive method gives the
  same capability without the churn.
- *Docs-only (option B).* Leaves every merged raster ungeoreferenced
  with no supported way back.

**See also.** #194; ADR-005 (`merge_to_xarray`).

---

## ADR-008 — Public namespaces are organised by task, one home per name

**Status.** Accepted.

**Context.** The root namespace had grown to 112 names, mostly every
axis twice — once at the root and again in its family module, both with
a `Spatial` / `Temporal` name prefix — and the public modules mixed concepts (`spatial`, `time`, `fields`,
`matched`) with plumbing (`runners`, `dask`, `jax`, `hooks`, `objstore`,
`cog`). Users had no map from "what am I trying to do" to "where is it".

**Decision.**

- The root keeps only what every job touches: the patchers, the patch
  carriers, `Field` / `AsyncField` / `Domain` and `RasterField`.
- Everything else has exactly one public home, grouped by task:
  `spatial` / `temporal` (one module per axis — `geometry`, `sampler`,
  `window`, `aggregation`, plus temporal `stencils`), `fields`,
  `matched`, `run` (runners, Dask / JAX, `PatchCache`, random access),
  `observe` (hooks, journal, error records, strict mode) and `config`.
  `tests/test_exports.py::test_one_home_per_public_name` enforces it.
- Axes drop their family prefix — the namespace carries it:
  `spatial.window.Hann`, `temporal.aggregation.Mean`. The `time` module
  becomes `geopatcher.temporal` (no shadowing of the standard library).
- Config envelopes name classes by public path (`"spatial.window.Hann"`)
  so the unprefixed names never collide.
- The object-store pool and the COG engine move to their own package,
  geotoolz-cloud (`geocloud.store`, `geocloud.cog`): they are I/O shared
  by the whole stack, not patching. `ObstoreCogField` becomes
  `geopatcher.fields.CogField`, a thin `Field` over `geocloud.cog.CogSource`.
- Clean break: no aliases for the old spellings (0.x, called out in the
  release notes).

**Consequences.** Every import of an axis changes
(`from geopatcher import spatial` then `spatial.window.Hann()`), and
configs saved with the old bare class names must be re-saved. In return
each name is findable from its task, and the root is short enough to read.

## ADR-009 — The domain fixes the grid, the patches fix the bands

> Supersedes the shape rule of ADR-007.

**Status.** Accepted.

**Context.** Every dense aggregation sized its accumulator from
`domain.shape`. A per-patch operator that changes the band count — NDVI
turns a `(4, 256, 256)` chip into a `(256, 256)` map, a model returns `K`
class scores — was then numpy-broadcast into the domain's bands: merging
NDVI onto a 4-band domain returned four identical copies of the index,
silently. `merge_to_field` refused the honest shape, so the documented
workaround was a hand-built one-band "output grid" domain, and the pipekit
`MergePatches` (then named `Stitch`) returned a bare array where
its input chips were `GeoTensor`s.

**Decision.**

- The domain fixes the grid, the trailing `(H, W)`; the patches fix what
  each cell holds. For raster-window patches the accumulator is
  `(*patch.data.shape[:-2], H, W)`: four bands in and one index out
  merges into `(1, H, W)`, and a 2-D map into `(H, W)`. The first patch
  decides, and a patch with different leading axes raises `ValueError`.
  `{dim: slice}` (grid) patches index the domain's own axes and keep its
  shape. One helper (`_dense_layout`) applies this to every dense
  aggregation, the streaming `OverlapAdd` included. `SoftVote` and
  `InvVarWeightedMean` read the cell axes past their class axis and
  `(mu, var)` pair.
- `merge_to_field` accepts any band count on a raster field: the output
  needs the domain's trailing `(H, W)`. The xarray adapters keep their
  dims and coords, so they still need the domain's own shape.
- A raster `with_data` (and `MergePatches`) follows the `GeoTensor` carrier
  contract:
  - `attrs` is a fresh copy of the source's, without the per-band keys
    once the band count changed;
  - the declared nodata follows one rule, decided by the merge and never
    by scanning values: an aggregation's finite `fill_value` (`-1` for the
    votes, a caller's sentinel) is the nodata; an output that carries the
    source's values (domain shape, source dtype restored) keeps the
    source's nodata, with its NaN gaps rewritten to it; anything else is a
    new quantity with a NaN fill (its per-patch operator marks nodata as
    NaN, per the `GeoTensor` contract). Every raster wrapper, `CogField`
    included, gets the same nodata;
  - the source dtype is restored only for an output of the domain's own
    shape.
- The pipekit `MergePatches` returns a `GeoTensor` on a georeferenced domain
  (one with `transform` and `crs`), through the same rewrap; `dict`,
  streaming and non-georeferenced outputs are unchanged.

**Consequences.** `MergePatches(domain=field.domain)` is right for every
operator, band-collapsing or not, and its output is a georeferenced
`GeoTensor` that writes straight to a COG. Code that relied on the
broadcast (`SoftVote` `(K, h, w)` chips on a `(band, H, W)` domain gave
`(band, H, W)` copies; it now gives one `(H, W)` map) sees the patches'
shape instead. A patch with more axes than the domain no longer raises
in `OverlapAdd`: its leading axes are kept.

---

## How to add a decision

1. Open a PR with the proposed addition. The PR description argues the
   decision; the diff adds the ADR to this page.
2. Decisions are not changed in place. A new ADR supersedes an older
   one with a `> Supersedes ADR-NNN` note at the top and the
   superseded ADR keeps a `> Superseded by ADR-MMM` line.
3. Cross-reference the affected issues and design docs. Each ADR should
   be reachable from the issue or design discussion it resolved.
