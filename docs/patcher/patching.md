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
| **Geometry** | Shape + scale of the neighborhood (and the domain topology). | `SpatialRectangular`, `SpatialSphericalCap`, `SpatialKNNGraph`, `SpatialRadiusGraph`, `SpatialPolygonIntersection` |
| **Sampler** | Where anchors are placed; overlap is emergent. | `SpatialRegularStride`, `SpatialJitteredStride`, `SpatialRandom`, `SpatialPoissonDisk`, `SpatialExplicit`, `SpatialAlongTrack` |
| **Window** | Boundary treatment (spectral leakage, edge artefacts). | `SpatialBoxcar`, `SpatialHann`, `SpatialTukey`, `SpatialGaussian`, `SpatialCustom` |
| **Aggregation** | Local predictions → global field. | `SpatialOverlapAdd`, `SpatialMean`, `SpatialWeightedSum`, `SpatialInvVarWeightedMean`, `SpatialHardVote`, `SpatialByIndex`, … |

The Patcher composes them and exposes a tiny surface:

```python
patcher = SpatialPatcher(geometry=..., sampler=..., window=..., aggregation=...)
for patch in patcher.split(field):    # Iterator[Patch]
    out = operator(patch.data)
stitched = patcher.merge(outputs_as_patches, field.domain)
```

`split` is an iterator by design — streaming is the default; materialise
with `list(...)` when convenient.

## Determinism (stochastic samplers)

`SpatialRandom`, `SpatialJitteredStride`, `SpatialPoissonDisk`, and
`TemporalRandom` accept a `seed: int | None`. The contract (issue #18,
pinned by `tests/test_determinism.py`):

| `seed` value | Behavior |
|---|---|
| `int` | Two samplers with the same config return bit-identical anchors across calls *and* across instances. Use this whenever you need reproducible runs (ML evaluation, CI, journal-resume). |
| `None` (default) | The sampler re-seeds from OS entropy on every call; anchors will differ. Pick this for casual exploration when reproducibility doesn't matter. |

The Hypothesis round-trip suite (`tests/test_roundtrip.py`, issue #21)
leans on the `int` contract — given a seed, it shrinks failing
examples to the minimal `(shape, stride, seed)` triple and replays
them deterministically.

## Boundary policy

What happens when an anchor sits close enough to the edge that the
neighborhood would overflow the domain? `SpatialRectangular` exposes
this as a first-class parameter (issue #19):

```python
geom = SpatialRectangular(size=(256, 256), boundary="pad")
```

| Mode | Behavior |
|------|----------|
| `"drop"` (default) | Sampler clips so overflowing anchors are never emitted. Edge residual is silently dropped — exactly the pre-issue-19 behavior. |
| `"pad"` | Edge anchors are emitted; the patch is the full geometry size, padded in the overflow region with the reader's nodata (or `pad_value` when set). |
| `"reflect"` | Edge anchors are emitted; the overflow region is mirror-padded from the in-domain interior — the spectrally correct choice for overlap-add stitching with tapered windows (no DC dip at the scene boundary). Requires the overflow on each side to be smaller than the in-domain extent, else a clear `ValueError` is raised. |
| `"shrink"` | Edge anchors are emitted; the geometry clips the returned Window so the patch is *smaller* at the edge. Weights crop to match. |
| `"raise"` | Edge anchors are emitted; `SpatialPatcher.split` raises a `ValueError` on the first overflow. Useful with `SpatialExplicit` when the caller wants strict edge handling. |

`"pad"` and `"reflect"` are guaranteed by the patcher itself — the
overflowing window is clipped to the domain, read once, then padded up
to the full geometry size, with a `GeoTensor` chip's transform shifted so
its georeferencing stays exact. This is **field-independent**: it works
identically for `RasterField`, `RioXarrayField`, and any other `Field`.
Set a specific constant fill with `pad_value`:

```python
geom = SpatialRectangular(size=(256, 256), boundary="pad", pad_value=0.0)
```

Only `SpatialRectangular` on raster domains honors the parameter in v0.x;
graph and polygon geometries always behave as if `"drop"` (their natural
clipping is already correct), and `GridDomain` support is pending an
xarray-pad story.

## Mixed-CRS patching

Anchors and fields don't have to share a CRS (issue #20). Two
independent levels:

**Level 1 — anchor reprojection (cheap, metadata-only).** The
coordinate-consuming samplers take a `crs=` for coordinates expressed in
a CRS other than the domain's; they are reprojected to the domain CRS
before the pixel mapping. `SpatialAlongTrack` resamples by `spacing` in
*domain* units after the transform, and the new `SpatialExplicitCoords`
centres a chip on each world coordinate:

```python
# Event catalogue in lon/lat, imagery in UTM.
sampler = gp.SpatialExplicitCoords(
    coords=list(zip(catalog.lon, catalog.lat)),
    crs="EPSG:4326",            # None ⇒ coords already in the domain CRS
)
# Ground track in lon/lat over a UTM field.
sampler = gp.SpatialAlongTrack(track_lonlat, spacing=5_000.0, crs="EPSG:4326")
```

A `polar_guard` (`"warn"` / `"raise"` / `"ignore"`) flags unreliable
reprojection near the poles (`|lat| > 80°`) or across the ±180°
antimeridian when the source CRS is geographic.

**Level 2 — pixel reprojection (heavy, opt-in).** `ReprojectingRasterField`
presents the *destination* grid as its domain, so every sampler /
geometry / aggregation works on the target grid unchanged and each chip
is warped from the source:

```python
field = gp.ReprojectingRasterField(reader, dst_crs="EPSG:3857", resolution=30.0)
field.domain.crs                       # EPSG:3857 — samplers see the dst grid
patches = list(patcher.split(field))   # chips are (H, W) in dst_crs
```

Use Level 1 when the field is already on the grid you want and only the
anchor coordinates are foreign; reach for Level 2 when you need the whole
pipeline to run on a different grid than the source raster's.

## Caching reads across runs

Iterating on an operator means reading the same patches many times.
`PatchCache` (issue #24) is a cross-run, content-addressed on-disk cache
keyed by `sha256(field_id ‖ geometry+window config ‖ anchor)`: the second
*process* skips the source read entirely and only consults the field for
its `domain` metadata.

```python
cache = gp.PatchCache("./.geopatcher_cache", max_bytes=20 * 2**30)

for patch in patcher.split(field, cache=cache):   # run 1: reads + cache fill
    out = my_op_v1(patch.data)
for patch in patcher.split(field, cache=cache):   # run 2: zero source reads
    out = my_op_v2(patch.data)

cache.stats()   # {"hits": ..., "misses": ..., "bytes": ..., "entries": ...}
```

It composes with `journal=` (completion tracking) and `prefetch=`, and
plugs into random access via `IndexedPatchView(patcher, field, cache=cache)`.
Path- and URL-backed fields derive their identity automatically; pass
`PatchCache(..., field_id="scene")` for in-memory (`GeoTensor`-backed)
fields, which have no stable identity of their own.

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
| **Geometry** | Window shape (lookback, horizon, multi-scale, phase). | `TemporalFixedLookback`, `TemporalLookbackHorizon`, `TemporalMultiScale`, `TemporalPhaseWindow` |
| **Sampler** | Anchor placement in time. | `TemporalRegularStride`, `TemporalCausalRolling`, `TemporalEventTriggered`, `TemporalRandom`, `TemporalExplicit` |
| **Window** | Temporal boundary treatment. | `TemporalCausalBoxcar`, `TemporalExponentialDecay`, `TemporalTaperedTukey`, `TemporalPeriodic` |
| **Aggregation** | Time → time reconstruction. | `TemporalFold` (RNN-like state-passing), `TemporalMean`, `TemporalHierarchicalCombine`, `TemporalForecast` |

`TemporalFold` is the name for the RNN-like fold (renamed from the design's
`Sequential` to avoid clashing with operator-graph `Sequential` types in
downstream composition libraries).

## Spatiotemporal composition

`SpatioTemporalPatcher` composes a `SpatialPatcher` and a `TemporalPatcher`
with one of two coupling modes:

- `"product"` (default) — Cartesian product of every spatial anchor × every
  time anchor. The right shape for dense gridded data (climate output,
  regular satellite revisits).
- `"coupled"` — explicit `(space, time)` anchor pairs from the spatial
  sampler's `anchors_`. The right shape for event-triggered patches
  (methane plume detections, Argo profile locations, storm tracks).

## Operator-graph bridge

Operator-graph composition libraries (e.g.
[`geotoolz`](https://github.com/jejjohnson/geotoolz)) ship thin wrappers
that adapt the Patcher into their `Operator` world — typically a triple
of `GridSampler(patcher)`, `ApplyToChips(operator)`, and
`Stitch(aggregation, domain)`. Those wrappers live in the consuming
library, not here; geopatcher itself has no operator-graph dependency.

## Streaming aggregations

Every `SpatialAggregation` carries a `streaming_safe: ClassVar[bool]`. The
canonical streaming-safe member is `SpatialOverlapAdd`, which accepts
`streaming=True, target_path=...` to accumulate into an on-disk
[zarr](https://zarr.dev) store instead of RAM. The exact streaming family
(`Sum`, `Mean`, `Variance`, `OverlapAdd`, `WeightedSum`, `InvVarWeightedMean`,
`HardVote`, `SoftVote`) is fully implemented. The approximate sketch
family (`ApproxQuantile`, `ApproxCardinality`, `ApproxMode`,
`StreamingHistogram`, `Reservoir`) provides global streaming summaries for
operational-scale jobs that need bounded reducer state rather than a full
materialised field.

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
pip install 'geopatcher[grid]'           # XarrayField
pip install 'geopatcher[vector]'         # GeoPandasField
pip install 'geopatcher[point]'          # XvecField
pip install 'geopatcher[xarray-raster]'  # RioXarrayField
pip install 'geopatcher[dask]'           # DaskField + Dask helpers
pip install 'geopatcher[streaming]'      # OverlapAdd(streaming=True)
pip install 'geopatcher[patch-full]'     # everything above
```

Each adapter raises a friendly `ImportError` pointing at the right extra if
the backend library is missing.

## Where the framework draws the line

- **Mesh / `uxarray`** (`UXarrayField`) is deferred to v0.2.
- **Hierarchical Patcher-of-Patchers** is supported as a *recipe* on top of
  the framework rather than a dedicated class. See the streaming tutorial
  notebook.
- **Two-pass / global-context operators** (global normalisation,
  attention across patches) are explicitly out of scope; users write the
  two passes themselves on top of the existing primitives.
