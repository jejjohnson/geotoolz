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

**Decision.** All three patcher families (`SpatialPatcher`,
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
- `prefetch=N` (#9), `asplit()` (#8), and `max_in_flight` backpressure
  (#16) compose for free — they all wrap the iterator without changing
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
(`SpatialOverlapAdd(streaming=True, target_path=...)`, future
`SpatialInvVarWeightedMean(streaming=True, ...)`, etc.) write to a
**framework-managed Zarr store** by default. A pre-opened `zarr.Array`
may be passed in for callers that need Dask / distributed writers to
share the same store.

**Context.** The streaming asymmetry (see §4 of the `scaling.md`
design note in the planning archive; not shipped with these docs) is on the
*output* side: the input field already has a lazy `Field.select`, so
input scales as long as `split` returns an iterator. Output
preallocability is the bottleneck. A disk-backed accumulator solves it.

Zarr was picked over memmap, HDF5, and "bring your own store":

- Zarr v2 / v3 are already a hard requirement of the streaming
  `SpatialOverlapAdd` implementation; users of streaming inference
  already have it installed.
- Chunked, append-friendly, parallel-writable, plays well with Dask
  and downstream COG conversion (#15).
- The chunk shape can be derived from the first patch's data shape —
  the patcher knows the natural chunking without the user having to
  spell it out.

**Consequences.**

- Default usage is one line: `SpatialOverlapAdd(streaming=True,
  target_path="out/")`. No `import zarr` in user code.
- Pre-opened-store path remains supported for power users: pass a
  `zarr.Array` (or any object satisfying the same write contract) via
  a future `target_store=` keyword. Both shapes coexist; the managed
  path is the documented default.
- Future v3-sharded outputs (#14) and COG aggregation target (#15)
  layer on top of the Zarr default without changing aggregation APIs.
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
> `set_strict` decision (e.g. in `geopatcher._src.config`) resolve
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

gp.set_strict(True)      # promotes streaming_safe warnings to errors
gp.set_strict(False)     # back to warn-only (default)
gp.get_strict()          # bool
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
   `SpatialMedian` / `SpatialLearned` in a notebook.
2. **Warning only.** What we have. Quiet failures in batch jobs.
3. **Configurable.** Best of both — default-permissive, opt-in strict.

**Consequences.**

- Casual / notebook users see no behavior change.
- Batch / CI users can lock down with `gp.set_strict(True)` (or the env
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
`Field` Protocol** via its primary's `domain`. Concretely:

- `MatchedField.primary` is a regular `Field`; its CRS / bounds /
  shape define the anchor space.
- `MatchedField.secondaries: Mapping[str, Field]` carries the
  matched sources keyed by name.
- `MatchedField.coreg: Mapping[str, Callable]` carries one
  coregistration callable per secondary. Type is the broad
  `Callable[[Any, Any], Any]`, but the **recommended** value is a
  `pipekit.Operator` from `geotoolz.geom.coregister.*` so the
  alignment step round-trips through YAML.
- `MatchedField.select(indexer)` returns a `MatchedPatch`
  (sibling carrier — not a subclass of `Patch`) holding one patch
  per source under `members[name]`, plus optional per-source
  `valid_mask` for partial coverage.

**Context.** The cross-package query→matchup→patch design
(`docs/design/query-matchup.md`) introduces matchups between LEO,
GEO, vector, and point-cloud sources. The patching side has to read
co-located neighborhoods across these heterogeneous sources without
duplicating coregistration logic (which lives in `geotoolz`) and
without forcing geopatcher's framework-free core to depend on
`pipekit`.

The composite-Field approach satisfies all three constraints: every
existing sampler, geometry, window, and aggregation works on a
`MatchedField` unchanged; the heavy alignment work lives in
`geotoolz.geom.coregister.*` `pipekit.Operator`s; geopatcher's only
new typing dependency is the standard-library `Callable` (since
`pipekit.Operator` IS callable).

**Consequences.**

- A user with no matchup needs continues to write `SpatialPatcher`
  pipelines against a `Field` — nothing changes.
- A user with matchups writes `MatchedField(primary, secondaries,
  coreg)` and **passes that to the same `SpatialPatcher` they
  already use**. The `split()` iterator yields `MatchedPatch`es
  instead of `Patch`es; downstream code branches once on
  `isinstance(p, MatchedPatch)` if it wants per-source access.
- Per-source merge needs a new wrapper (`MatchedSpatialPatcher`)
  because the existing `SpatialPatcher.merge` returns one `Field`.
  This is the *only* API shape change introduced — and it lives in
  a new class so backwards compatibility is preserved.
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
  `TemporalPatcher`.* Would mean rewriting samplers, geometries,
  windows, and aggregations to accept matched fields. The composite
  approach gets the same surface for free.

---

## ADR-004 — Coordinate-aware temporal patching is opt-in; stride-1 only in v0.1

**Context.** The temporal stack works in integer index space:
`TemporalSampler.anchors(time_len) → Iterable[int]`, `TemporalGeometry.window(
time_len, anchor) → slice`. This is correct and fast for in-memory arrays at a
known cadence. It breaks down for ARCO-ERA5-style workloads where the natural
specification is *physical* — "a 9-hour lookback at the source cadence,
whatever that is" — and the data cadence is a property of the store, not of
the caller. A `TimeStencil`-based layer (ported from `neuralgcm/terrax`)
expresses windows in coordinate units and validates that the requested step
exactly tiles the source grid.

**Decision.** Extend the temporal protocol via two ClassVar capability flags:

- `TemporalGeometry.needs_coord: ClassVar[bool] = False` (default).
- `TemporalSampler.needs_coord: ClassVar[bool] = False` (default).

Coordinate-aware subclasses (`TemporalStencilGeometry`,
`TemporalStencilSampler`) set the flag to `True`. `TemporalPatcher` reads the
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

**v0.1 stride-1 constraint.** `TemporalWindow.weights(geometry, length)` and
`TemporalAggregation.merge(patches)` both assume contiguous integer index
ranges (`s.stop - s.start` is the realised window length). A stencil with
`step > source_step` would yield a strided slice, silently breaking both.
`TemporalStencilGeometry.__post_init__` raises when `source_step` is supplied
and `stencil.step / source_step != 1`; `window_coord` re-checks the resolved
slice's stride at resolve time as a belt-and-braces guard for callers that
omit `source_step`. Strided reads are deferred to v0.2 — they need
`TemporalWindow.weights` and `TemporalAggregation.merge` to take a
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
  `TemporalStencilGeometry`/`TemporalStencilSampler` classes carry the flag,
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

**See also.** GitHub issue #56 (the design doc and tracking issue for this
work); the upstream `neuralgcm/terrax` `xreader.stencils` module (Apache-2.0,
© Google LLC) from which the stencil math was ported.

---

## ADR-005 — Random access is a Sequence wrapper, cache lives on the view

**Context.** The patcher's canonical surface is
`SpatialPatcher.split → Iterator[Patch]` (ADR-001). xrpatcher's
`XRDAPatcher[i]` API gives random-access by integer index, and `xrpatcher`
also bundles in-memory caching (`cache=True`/`preload=True`) for ML
loaders that re-read the same anchors per epoch. Migrants want the same
ergonomics without losing the iterator-first contract; the question is
*where* the random-access surface lives and *what protocol* it speaks.

**Decision.** Add `geopatcher.IndexedPatchView`:

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

- `SpatialRegularStride(check_full_scan=True)` raises
  `IncompleteScanConfiguration` at anchor time when `(length - size) %
  step` is nonzero on any axis. Off by default to preserve existing
  silent-truncation semantics; opt in for the xrpatcher
  strict-tiling story. Spatial analogue of `divide_evenly` (ADR-004).
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
`SpatialOverlapAdd.merge`, which `np.asarray`'s every patch's data.

**Consequences.**

- **Iterator-first split stays canonical.** ADR-001 is unchanged. The
  view is a wrapper, not a replacement; consumers that prefer
  iterators see no change.
- **`from geopatcher import IndexedPatchView` works.** Root re-export
  + `__all__` entry, alongside `SpatialPatcher`, `TimeStencil`, etc.
- **No framework adapter packages.** Following the same stance as
  ADR-004 and PRs #41 / #57, we ship primitives (the Sequence-shaped
  view) plus recipes (one-line torch / Grain wrappers), not adapter
  classes.
- **Content-addressed cache is still future work.** This PR's
  index-keyed in-memory cache is the cheap xrpatcher port; the deeper
  cross-session content-addressed cache is tracked at #24. The
  `IndexedPatchView` is the natural home for it.
- **`xrpatcher` can be archived.** After this PR ships in a tagged
  release, `XRDAPatcher` users can swap one import:
  `XRDAPatcher(da, patches=..., strides=...)` →
  `IndexedPatchView(SpatialPatcher(...), XarrayField(da))`. See
  `recipes/xarray-nd-patching.md` for the side-by-side.

**Alternatives considered.**

- *Bundle a torch `Dataset` subclass in `geopatcher`.* Drags torch
  into core deps for every user. Same primitives-not-adapters stance
  the user took on jax and torch in PRs #11 / #23 (and the temporal
  stencils work in #57).
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

**See also.** GitHub issue #60 (the design doc and tracking issue for
this work); the upstream `xrpatcher` (the migration target);
[`recipes/xarray-nd-patching.md`](recipes/xarray-nd-patching.md) for
side-by-side migration; #24 for the deeper content-addressed cache;
ADR-001 (iterator-first split) and ADR-004 (coordinate-aware temporal).

---

## How to add a decision

1. Open a PR with the proposed addition. The PR description argues the
   decision; the diff adds the ADR to this page.
2. Decisions are not changed in place. A new ADR supersedes an older
   one with a `> Supersedes ADR-NNN` note at the top and the
   superseded ADR keeps a `> Superseded by ADR-MMM` line.
3. Cross-reference the affected issues and design docs. Each ADR should
   be reachable from the issue or design discussion it resolved.
