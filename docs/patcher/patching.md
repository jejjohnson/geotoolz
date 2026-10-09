# Patching

`geopatcher` is the locality layer of the stack. Where an operator-graph
composition library (e.g. [`geotoolz`](https://github.com/jejjohnson/geotoolz)) settles
*what to compute*, and the [`Field` / `Domain`](#protocols-field-and-domain)
Protocols settle *what backend the data lives on*, the Patcher settles
the third orthogonal question: **what slice of the data does the
operator see at once, and how do local outputs become a global field?**

Three Patcher classes compose the four-axis framework:

- `SpatialPatcher` — neighborhoods in space (raster, grid, points, polygons).
- `TemporalPatcher` — windows along a time axis.
- `SpatioTemporalPatcher` — composition of the two with explicit coupling.

## The four spatial axes

| Axis | Controls | Examples |
|------|----------|----------|
| **Geometry** | Shape + scale of the neighborhood (and the domain topology). | `spatial.geometry.Rectangular`, `spatial.geometry.SphericalCap`, `spatial.geometry.KNNGraph`, `spatial.geometry.RadiusGraph`, `spatial.geometry.PolygonIntersection` |
| **Sampler** | Where anchors are placed; overlap is emergent. | `spatial.sampler.RegularStride`, `spatial.sampler.JitteredStride`, `spatial.sampler.Random`, `spatial.sampler.PoissonDisk`, `spatial.sampler.Explicit`, `spatial.sampler.ExplicitCoords`, `spatial.sampler.AlongTrack` |
| **Window** | Boundary treatment (spectral leakage, edge artefacts). | `spatial.window.Boxcar`, `spatial.window.Hann`, `spatial.window.Tukey`, `spatial.window.Gaussian`, `spatial.window.Custom` |
| **Aggregation** | Local predictions → global field. | `spatial.aggregation.OverlapAdd`, `spatial.aggregation.Mean`, `spatial.aggregation.WeightedSum`, `spatial.aggregation.InvVarWeightedMean`, `spatial.aggregation.HardVote`, `spatial.aggregation.ByIndex`, … |

The Patcher composes them and exposes a tiny surface:

```python
patcher = SpatialPatcher(geometry=..., sampler=..., window=..., aggregation=...)
for patch in patcher.split(field):    # Iterator[Patch]
    out = operator(patch.data)
stitched = patcher.merge(outputs_as_patches, field.domain)
```

`split` is an iterator by design — streaming is the default; materialise
with `list(...)` when convenient.

## Window convention

`spatial.window.Hann` and `spatial.window.Tukey` are **periodic** (DFT-even) tapers —
per axis exactly `scipy.signal.windows.hann(n, sym=False)` /
`tukey(n, alpha, sym=False)`, combined as an outer product:

- **Hann is COLA at hop `N/2`.** `w[k] + w[k + N/2] = 1`, so with an even
  patch size and `step = size // 2`, every interior seam is covered at
  full weight — even `spatial.aggregation.OverlapAdd(normalize_by_window=False)`
  reproduces a constant there. Other overlapping strides still
  reconstruct exactly under the default normalisation.
- **`spatial.window.Tukey(alpha=1.0)` is `spatial.window.Hann`**, `alpha=0.0` is
  `spatial.window.Boxcar`; Tukey is COLA at hop `N * (1 - alpha/2)`.
- **Endpoints.** `w[0] = 0`, `w[-1] > 0`. Merged with `spatial.aggregation.OverlapAdd`,
  the domain's **first row and first column** (the leading border ring)
  get zero accumulated weight, because regular samplers start at anchor 0
  and the only chip covering them puts its zero sample there; those cells
  come back as the aggregation's `fill_value` — NaN by default, or e.g.
  `spatial.aggregation.OverlapAdd(fill_value=domain.fill_value_default)` for the
  domain's nodata — never as a value that looks like data (see
  [Aggregation fill and ties](#aggregation-fill-and-ties)). `"pad"` /
  `"reflect"` only extend the trailing
  (bottom/right) edge, so they do not remove this ring. If it matters,
  crop the 1-pixel ring, use `spatial.window.Gaussian` (never zero) or
  `spatial.window.Boxcar`, or supply a `spatial.window.Custom` taper.
- **`step == size` with Hann/Tukey** leaves every chip's first row and
  column at zero weight (the `fill_value` seams) — tapers need overlapping
  chips; use
  `spatial.window.Boxcar` for exact non-overlapping tiling.
- **Short axes.** An axis shorter than 3 samples cannot carry a taper and
  is boxcar (all ones) on that axis.

`spatial.window.Gaussian(sigma=...)` takes `sigma` as a **fraction of the patch
half-width** (std = `sigma * n / 2` pixels per axis), symmetric about the
patch centre.

## Determinism (stochastic samplers)

`spatial.sampler.Random`, `spatial.sampler.JitteredStride`, `spatial.sampler.PoissonDisk`, and
`temporal.sampler.Random` accept a `seed: int | None`. The contract (issue jejjohnson/geopatcher#18,
pinned by `tests/test_determinism.py`):

| `seed` value | Behavior |
|---|---|
| `int` | Two samplers with the same config return bit-identical anchors across calls *and* across instances. Use this whenever you need reproducible runs (ML evaluation, CI, journal-resume). |
| `None` (default) | The sampler re-seeds from OS entropy on every call; anchors will differ. Pick this for casual exploration when reproducibility doesn't matter. |

The Hypothesis round-trip suite (`tests/test_roundtrip.py`, issue jejjohnson/geopatcher#21)
leans on the `int` contract — given a seed, it shrinks failing
examples to the minimal `(shape, stride, seed)` triple and replays
them deterministically.

## Boundary policy

What happens when an anchor sits close enough to the edge that the
neighborhood would overflow the domain? `spatial.geometry.Rectangular` exposes
this as a first-class parameter (issue jejjohnson/geopatcher#19):

```python
geom = spatial.geometry.Rectangular(size=(256, 256), boundary="pad")
```

| Mode | Behavior |
|------|----------|
| `"drop"` (default) | Samplers only place anchors whose patch lies wholly in-domain; the trailing residual is dropped. A patch larger than the domain places **no** anchor (with a `RuntimeWarning`). |
| `"pad"` | Samplers also place the edge anchor — the first whose patch reaches the edge, never an extra one past it. The patch is the full geometry size, padded in the overflow region with the reader's nodata (or `pad_value`, which must be representable in the field's dtype). |
| `"reflect"` | As `"pad"`, but the overflow region is mirror-padded from the in-domain interior (numpy `mode="reflect"`, repeated when the overflow exceeds the domain) — so a tapered window's trailing (bottom/right) flank lands on mirrored data rather than a constant fill. It does not reach the leading row/column (regular samplers start at anchor 0) — see [Window convention](#window-convention). Needs at least two cells on a padded axis. |
| `"shrink"` | As `"pad"` for anchor placement, but the window is clipped to the domain on every side — a negative anchor included — so the patch is *smaller* at the edge. Weights crop to the same in-domain part. |
| `"raise"` | As `"pad"` for anchor placement; `SpatialPatcher.split` raises a `ValueError` on the first overflowing window. Useful with `spatial.sampler.Explicit` when the caller wants strict edge handling. |

Every mode survives `merge`: each dense aggregation (`spatial.aggregation.OverlapAdd`
in memory and streaming, `spatial.aggregation.Sum`, `spatial.aggregation.Mean`, `spatial.aggregation.Max`, …)
crops a chip's data and weights to the in-domain part of its window, so
padded or reflected cells are read for context but never written back.
`spatial.sampler.RegularStride(check_full_scan=True)` only applies under `"drop"`
— the other modes cover the trailing edge themselves.

`"pad"` and `"reflect"` are guaranteed by the patcher itself — the
overflowing window is clipped to the domain (grown inward under
`"reflect"` so the mirror source is in hand), read once, then padded up
to the full geometry size, with the chip's transform shifted so its
georeferencing stays exact (a `GeoTensor` via its own `pad`; a rioxarray
`DataArray` gets its spatial coords rebuilt from the shifted affine).
This is **field-independent**: it works identically for `RasterField`, `RioXarrayField`, and any other `Field`.
Set a specific constant fill with `pad_value`:

```python
geom = spatial.geometry.Rectangular(size=(256, 256), boundary="pad", pad_value=0.0)
```

`spatial.geometry.Rectangular` honours the parameter on raster domains and on
`GridDomain` (`XarrayField`, `DaskField`): grid chips are padded by dim
name, with their coordinates continued past the edge at the edge
spacing. Graph and polygon geometries always behave as if `"drop"`
(their natural clipping is already correct).

## Mixed-CRS patching

Anchors and fields don't have to share a CRS (issue jejjohnson/geopatcher#20). Two
independent levels:

**Level 1 — anchor reprojection (cheap, metadata-only).** The
coordinate-consuming samplers take a `crs=` for coordinates expressed in
a CRS other than the domain's; they are reprojected to the domain CRS
before the pixel mapping. `spatial.sampler.AlongTrack` resamples by `spacing` in
*domain* units after the transform, and the new `spatial.sampler.ExplicitCoords`
centres a chip on each world coordinate:

```python
import geopatcher as gp

# Event catalogue in lon/lat, imagery in UTM.
sampler = gp.spatial.sampler.ExplicitCoords(
    coords=list(zip(catalog.lon, catalog.lat)),
    crs="EPSG:4326",            # None ⇒ coords already in the domain CRS
)
# Ground track in lon/lat over a UTM field.
sampler = gp.spatial.sampler.AlongTrack(track_lonlat, spacing=5_000.0, crs="EPSG:4326")
```

A `polar_guard` (`"warn"` / `"raise"` / `"ignore"`) flags unreliable
reprojection near the poles (`|lat| > 80°`) or across the ±180°
antimeridian when the source CRS is geographic.

**Level 2 — pixel reprojection (heavy, opt-in).** `ReprojectingRasterField`
presents the *destination* grid as its domain, so every sampler /
geometry / aggregation works on the target grid unchanged and each chip
is warped from the source:

```python
import geopatcher as gp

field = gp.fields.ReprojectingRasterField(reader, dst_crs="EPSG:3857", resolution=30.0)
field.domain.crs                       # EPSG:3857 — samplers see the dst grid
patches = list(patcher.split(field))   # chips are (*bands, H, W) in dst_crs
```

The domain keeps the source's leading dims (`(bands, H, W)` for a
multi-band reader), so chips merge back without a rank mismatch. Each
chip warps only a crop of the source around its footprint (plus a small
kernel margin), so per-chip cost scales with the chip, not the scene.
Chips are warped independently, so a stitched mosaic can differ very
slightly from one full-scene warp (GDAL's approximate transformer);
call `georeader.read.read_reproject` once when you need that exactly.

Use Level 1 when the field is already on the grid you want and only the
anchor coordinates are foreign; reach for Level 2 when you need the whole
pipeline to run on a different grid than the source raster's.

## Caching reads across runs

Iterating on an operator means reading the same patches many times.
`PatchCache` (issue jejjohnson/geopatcher#24) is a cross-run, content-addressed on-disk cache
keyed by `sha256(field_id ‖ geometry+window config ‖ anchor)`: the second
*process* skips the source read entirely and only consults the field for
its `domain` metadata.

```python
import geopatcher as gp

cache = gp.run.PatchCache("./.geopatcher_cache", max_bytes=20 * 2**30)

for patch in patcher.split(field, cache=cache):   # run 1: reads + cache fill
    out = my_op_v1(patch.data)
for patch in patcher.split(field, cache=cache):   # run 2: zero source reads
    out = my_op_v2(patch.data)

cache.stats()   # {"hits": ..., "misses": ..., "bytes": ..., "entries": ...}
```

It composes with `journal=` (completion tracking) and `prefetch=`, and
plugs into random access via `IndexedPatchView(patcher, field, cache=cache)`.

**What the key covers.** `field_id` is everything the field reads:

- the *source* — a reader's file paths (`realpath` + mtime + size of
  *every* path, so editing any file of a multi-file reader invalidates
  it), a `url`, or the `encoding["source"]` file of a `rioxarray.open_rasterio` /
  `xr.open_dataset` array. Pass `PatchCache(..., field_id="scene")` for
  in-memory (`GeoTensor`- or `DataArray`-backed) fields, which have no
  stable identity of their own — and also for a file-backed `DataArray`
  whose values you changed in memory, since its `encoding["source"]` still
  names the unchanged file;
- the *domain* — CRS, transform, shape and dtype (or a digest of the grid
  coordinates for `XarrayField`);
- the reader's band selection (`indexes`) and boundless fill;
- the adapter's own `cache_id()`: `CogField` folds in its store /
  `path` (with an explicit `store=` the `url` is only a label) and
  `ifd_index`, `ReprojectingRasterField` its `dst_crs`, `resolution` and
  `resampling`. A custom `Field` can define `cache_id() -> str` the same way;
  it is trusted, not verified, so it must change whenever the patches
  could (a non-string or empty result raises `TypeError`).

A plain `url` carries no version: an object overwritten in place is not
detected. `CogField` closes that gap with one `HEAD` per `split`
(the object's ETag, else size + last-modified); for other URL-backed
fields, give them a `cache_id()` that includes a version, or `clear()`
the cache after the remote data changes.

**Hits are bit-identical.** Entries store the carrier, not just the
pixels: a `GeoTensor` comes back with its transform, CRS,
`fill_value_default` and `attrs`; an `xarray.DataArray`
(`RioXarrayField`, `XarrayField`, `DaskField`) with its dims, coords,
attrs and encoding, so `rio.transform()` / `rio.crs` / `rio.nodata`
match the uncached chip. A carrier that cannot round-trip exactly — a
`GeoDataFrame` / `XvecField` patch, an object-dtype array, an attribute
such as a `datetime` — raises `TypeError` instead of being served back
degraded; drop `cache=` for that field.

**Only real reads are stored.** `on_error="mask"` placeholders are never
written, so a run after a transient source outage reads the recovered
source. A damaged entry (zero-byte, truncated, not a zip) is treated as
a miss, deleted and rewritten; entries are always published atomically
(temp file + `os.replace`).

**Eviction is incremental.** With `max_bytes` set, the directory is
scanned once when the `PatchCache` is built; every `get` / `put` then
updates an in-memory size tally and least-recently-used order, and
entries are evicted only when a write takes the total over the cap. An
entry larger than `max_bytes` on its own is not stored (a
`RuntimeWarning` says so) rather than flushing the whole cache and then
itself. The tally is per `PatchCache` instance, so with several
processes writing one directory the cap applies per writer.

!!! warning "Cache layout changed"
    The key and entry layout changed in this release: existing cache
    directories are never hit again. Call `cache.clear()` (or delete the
    directory) to reclaim the space.

## Protocols: `Field` and `Domain`

The Patcher consumes a `Field` (something with `domain`, `select(indexer)`,
`with_data(array)`). The raster path reuses
[`georeader.GeoData`](https://github.com/IPL-UV/georeader) verbatim through
the thin `RasterField` adapter; the non-raster Fields (`XarrayField`,
`GeoPandasField`, `XvecField`, `RioXarrayField`) live under
`geopatcher.fields` and lazy-import their optional extras.

| Field | Domain | Backend |
|---|---|---|
| `RasterField`, `AsyncRasterField` | `RasterDomain` (`georeader.GeoDataBase`) | `RasterioReader`, `AsyncGeoTIFFReader`, `GeoTensor` |
| `RioXarrayField` | `RasterDomain` | rioxarray `DataArray` |
| `XarrayField` | `GridDomain` | `xarray.DataArray` (non-raster) |
| `DaskField` | `GridDomain` | dask-backed `xarray.DataArray` (lazy chunks) |
| `GeoPandasField` | `VectorDomain` / `PointDomain` | `geopandas.GeoDataFrame` |
| `XvecField` | `PointDomain` | `xvec.Dataset` |

Geometry × Domain dispatch is explicit `isinstance` (Protocol nominal typing
doesn't play well with `singledispatch`). Unsupported pairings raise
`NotImplementedError` at runtime.

## The four temporal axes

Mirror of the spatial side, with axes that encode time-specific properties
(causality, periodicity, multi-scale, forecasting):

| Axis | Controls | Examples |
|------|----------|----------|
| **Geometry** | Window shape (lookback, horizon, multi-scale, phase). | `temporal.geometry.FixedLookback`, `temporal.geometry.LookbackHorizon`, `temporal.geometry.MultiScale`, `temporal.geometry.PhaseWindow`, `temporal.geometry.StencilGeometry` |
| **Sampler** | Anchor placement in time. | `temporal.sampler.RegularStride` (alias `temporal.sampler.RegularStride`), `temporal.sampler.Random`, `temporal.sampler.Explicit` (alias `temporal.sampler.Explicit`), `temporal.sampler.StencilSampler` |
| **Window** | Temporal boundary treatment. | `temporal.window.CausalBoxcar`, `temporal.window.ExponentialDecay`, `temporal.window.TaperedTukey`, `temporal.window.Periodic` |
| **Aggregation** | Time → time reconstruction. | `temporal.aggregation.Fold` (RNN-like state-passing), `temporal.aggregation.Mean`, `temporal.aggregation.HierarchicalCombine`, `temporal.aggregation.Forecast` |

`temporal.aggregation.Fold` is the name for the RNN-like fold (renamed from the design's
`Sequential` to avoid clashing with operator-graph `Sequential` types in
downstream composition libraries).

### Temporal boundary policy

Every integer temporal geometry takes `boundary`, deciding what happens to
a window that overflows the time axis:

| `boundary` | Behaviour |
|---|---|
| `"drop"` (default) | The anchor yields no patch, so every emitted window is full length. `anchors()` / `n_anchors()` / `patch_anchors()` skip it too. `temporal.geometry.MultiScale` drops an anchor whose *longest* scale overflows; `temporal.geometry.PhaseWindow` drops each overflowing cycle slot. |
| `"shrink"` | The window is clipped to `[0, time_len)`, so edge windows are shorter. |
| `"raise"` | `ValueError` naming the anchor and the window. |

`temporal.geometry.PhaseWindow(period, phase_width)` returns one slot per cycle —
`[k·period + φ − w, k·period + φ + w + 1)` with `φ = anchor % period` —
so each anchor yields one patch per cycle. Each `TemporalPatch` records
its position among its anchor's windows as `window_index` (scale `k` of a
`temporal.geometry.MultiScale`), which `temporal.aggregation.HierarchicalCombine` keys on.
`temporal.aggregation.Mean` is a running per-step sum / count over each patch's
`indices` (NaN not counted, unreached steps get `fill_value`), and
`temporal.aggregation.Forecast` locates the horizon from the anchor
(`[anchor + 1, anchor + 1 + horizon)`), skipping windows that do not hold
all of it.

!!! warning "Default changed from implicit shrink to drop"
    Temporal geometries used to clamp silently (today's `"shrink"`). Pass
    `boundary="shrink"` to keep shorter edge windows.

### `TemporalPatcher` runner knobs

`TemporalPatcher` takes `SpatialPatcher`'s runner knobs and runs them
through the same helpers: `on_error` / `max_retries` / `retry_on` /
`capture_traceback` (failures land in `errors` as `PatchErrorRecord`s,
`"mask"` yields a NaN window), and `split` / `asplit` / `reduce` /
`two_pass` take `hooks`, `prefetch`, `journal`, `cache` and
`max_in_flight` / `max_in_flight_bytes`. `merge` / `amerge` / `reduce`
run the streaming-safety / `set_strict` check. Journal, cache and
`errors` key a patch by its `patch_anchors` key — the anchor, or
`(anchor, k)` for a multi-window geometry. Every method takes
`time_axis=` and `coord=` as keywords.

The series is read one window at a time: a `GridDomain` field
(`XarrayField`, `DaskField`) through `select({time_dim: window})` — its
time coordinate is the default `coord=` for stencil pipelines — and any
array with a `shape` (numpy, dask, xarray, zarr) by slicing it. `coord=`
is validated once per call (1-D, strictly increasing; evenly spaced for
stencil components), so a stencil split is O(N) in the axis length.
`temporal.sampler.RegularStride(check_full_scan=True)` raises
`IncompleteScanConfiguration` when the windows leave time steps uncovered.

## Spatiotemporal composition

`SpatioTemporalPatcher` composes a `SpatialPatcher` and a `TemporalPatcher`
with one of two coupling modes:

- `"product"` (default) — Cartesian product of every spatial anchor × every
  time anchor. The right shape for dense gridded data (climate output,
  regular satellite revisits).
- `"coupled"` — explicit `(space, time)` anchor pairs from the spatial
  sampler's `anchors_`. The right shape for event-triggered patches
  (methane plume detections, Argo profile locations, storm tracks).

Both modes read each spatial chip exactly as `SpatialPatcher.split` does —
the spatial patcher's geometry, boundary padding / reflection, window
weights and `on_error` policy (failures land in `spatial.errors`) — then
slice it along `time_axis` into the temporal windows, keeping the chip's
carrier (`GeoTensor`, `DataArray`, …). `split` / `asplit` take `hooks`,
`journal`, `cache` (for the spatial chips) and `max_in_flight` /
`max_in_flight_bytes`, plus `prefetch` on `split`. A patch is keyed
`(space_anchor, time_key)` — `time_key` being the temporal
`patch_anchors` key — and a chip whose every window is journaled is not
read again. Product mode reports an unknown total (`-1`) to
`on_split_start`; coupled mode the number of pairs.

### One anchor walk

`SpatialPatcher.split` / `asplit`, `AsyncSpatialPatcher.asplit`,
`TemporalPatcher.split` / `asplit` and both `SpatioTemporalPatcher`
couplings run on one anchor-walk core, so `on_error` / retries, hooks,
journal, `PatchCache`, prefetch (sync) and backpressure behave the same on
every path. Each call starts a fresh `errors` list (both passes of a
`two_pass` share one); a journaled patch is reported to the hooks'
`on_patch_skipped`, and `on_error` hooks receive the original exception
whether it is raised or swallowed by the policy.

## Operator-graph bridge

Operator-graph composition libraries (e.g.
[`geotoolz`](https://github.com/jejjohnson/geotoolz)) ship thin wrappers
that adapt the Patcher into their `Operator` world — typically a triple
of `GridSampler(patcher=patcher)`, `ApplyToChips(operator=operator)`, and
`MergePatches(aggregation=aggregation, domain=domain)`. geopatcher ships
them for pipekit in `geopatcher.integrations.pipekit` (the `[pipekit]`
extra); the core has no operator-graph dependency.

## Streaming aggregations

Every `spatial.aggregation.Aggregation` carries a `streaming_safe: ClassVar[bool]`. The
canonical streaming-safe member is `spatial.aggregation.OverlapAdd`, which accepts
`streaming=True, target_path=..., chunks=...` to accumulate into an
on-disk [zarr](https://zarr.dev) store instead of RAM (or
`writer="cog"` to finish as a Cloud-Optimized GeoTIFF). The exact streaming family
(`Sum`, `Mean`, `Variance`, `OverlapAdd`, `WeightedSum`, `InvVarWeightedMean`,
`HardVote`, `SoftVote`) is fully implemented. The approximate sketch
family (`ApproxQuantile`, `ApproxCardinality`, `ApproxMode`,
`StreamingHistogram`, `Reservoir`) provides global streaming summaries for
operational-scale jobs that need bounded reducer state rather than a full
materialised field.

### Aggregation fill and ties

Every dense aggregation honours the same per-cell contract:

- **Valid samples.** A sample counts only where it is not NaN, lies inside
  the domain (pad / reflect overhang is cropped) and lies inside the
  interior mask of a masked window (`spatial.geometry.PolygonIntersection` on a
  raster, `spatial.geometry.SphericalCap` on a grid). Infinities are values.
- **Uncovered cells** — no valid sample, or zero accumulated weight (a
  taper's zero edge) — get the aggregation's `fill_value`: NaN by default
  for `Sum`, `Mean`, `Max`, `Min`, `WeightedSum`, `OverlapAdd` (in RAM and
  streaming), `Variance` (which also fills cells with fewer than two
  samples, as `np.nanvar(ddof=1)`), `InvVarWeightedMean` (both `mu` and
  `var`), `Median` and `Mode`. `HardVote` / `SoftVote` default to `-1`
  ("no class"). Pass `fill_value=` to override, e.g. the domain's nodata;
  the label aggregations (`HardVote`, `SoftVote`, `Mode`) return `int64`
  for an integral fill and `float64` otherwise.
- **Ties.** `HardVote` / `SoftVote` pick the lowest class index; `Mode`
  picks the smallest value.
- `InvVarWeightedMean` treats `var == 0` as an exact observation: the cell
  takes that sample's `mu` with `var = 0`.
- Patch indices that are not a dense placement (a point-index array, a
  polygon id) raise `TypeError`; use `spatial.aggregation.ByIndex`, which returns the
  `[(anchor, data), …]` pairs, for ragged geometries.
- `MeanStd` / `MinMax` (global) skip NaN and, on a raster / grid domain,
  count only each chip's in-domain, in-mask cells.

For resumable local jobs, create a `PatchJournal(path)` and pass it to
`patcher.split(field, journal=journal)`. Anchors with successful journal rows
are skipped on restart. Iterator backpressure is available through
`max_in_flight` or `max_in_flight_bytes`; close patches explicitly (or use them
as context managers) when you want to release a slot before the object is
garbage-collected.

## Optional extras

`geopatcher` keeps the base install slim and gates each non-raster
Field adapter behind an extra:

```bash
pip install 'geotoolz-patcher[grid]'           # XarrayField
pip install 'geotoolz-patcher[vector]'         # GeoPandasField
pip install 'geotoolz-patcher[point]'          # XvecField
pip install 'geotoolz-patcher[xarray-raster]'  # RioXarrayField
pip install 'geotoolz-patcher[dask]'           # DaskField + Dask helpers
pip install 'geotoolz-patcher[streaming]'      # OverlapAdd(streaming=True)
pip install 'geotoolz-patcher[patch-full]'     # everything above
```

Each adapter raises a friendly `ImportError` pointing at the right extra if
the backend library is missing.

`DaskField.from_zarr(store, var="sst")` wraps one data variable of a zarr
store (`var` may be omitted when the store holds exactly one; zarr itself
comes with the `[streaming]` extra). `SpatialPatcher.to_delayed(field)` /
`to_dask_bag(field)` build one Dask task (bag: one single-element
partition) per patch around `patch_at`, with the field entering the graph
once as a shared delayed node. They are plain read tasks: the runner-level
policies of `split` — `on_error` / retries, hooks, journal, `PatchCache`,
prefetch — do not apply; failures, retries and caching are the scheduler's
concern.

## Where the framework draws the line

- **Mesh / `uxarray`** (`UXarrayField`) is deferred to v0.2.
- **Hierarchical Patcher-of-Patchers** is supported as a *recipe* on top of
  the framework rather than a dedicated class. See the streaming tutorial
  notebook.
- **Two-pass / global-context operators** (global normalisation,
  attention across patches) are explicitly out of scope; users write the
  two passes themselves on top of the existing primitives.
