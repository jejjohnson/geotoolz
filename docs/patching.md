# Patching

`geotoolz.patch` is the locality layer of the stack. Where the composition
core ([`Operator`](api/core.md), `Sequential`, `Graph`) settles *what to
compute*, and the [`Field` / `Domain`](#protocols-field-and-domain) Protocols
settle *what backend the data lives on*, the Patcher settles the third
orthogonal question: **what slice of the data does the operator see at once,
and how do local outputs become a global field?**

Three Patcher classes compose the four-axis framework:

- `SpatialPatcher` — neighborhoods in space (raster, grid, points, polygons).
- `TemporalPatcher` — windows along a time axis.
- `SpatioTemporalPatcher` — composition of the two with explicit coupling.

## The four spatial axes

| Axis | Controls | Examples |
|------|----------|----------|
| **Geometry** | Shape + scale of the neighborhood (and the domain topology). | `SpatialRectangular`, `SpatialSphericalCap`, `SpatialKNNGraph`, `SpatialRadiusGraph`, `SpatialPolygonIntersection` |
| **Sampler** | Where anchors are placed; overlap is emergent. | `SpatialRegularStride`, `SpatialJitteredStride`, `SpatialRandom`, `SpatialPoissonDisk`, `SpatialExplicit` |
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

## Protocols: `Field` and `Domain`

The Patcher consumes a `Field` (something with `domain`, `select(indexer)`,
`with_data(array)`). The raster path reuses
[`georeader.GeoData`](https://github.com/IPL-UV/georeader) verbatim through
the thin `RasterField` adapter; the non-raster Fields (`XarrayField`,
`GeoPandasField`, `XvecField`, `RioXarrayField`) live under
`geotoolz.patch.fields` and lazy-import their optional extras.

| Field | Domain | Backend |
|---|---|---|
| `RasterField`, `AsyncRasterField` | `RasterDomain` (`georeader.GeoDataBase`) | `RasterioReader`, `AsyncGeoTIFFReader`, `GeoTensor` |
| `RioXarrayField` | `RasterDomain` | rioxarray `DataArray` |
| `XarrayField` | `GridDomain` | `xarray.DataArray` (non-raster) |
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
`Sequential` to avoid clashing with `geotoolz.Sequential`).

## Spatiotemporal composition

`SpatioTemporalPatcher` composes a `SpatialPatcher` and a `TemporalPatcher`
with one of two coupling modes:

- `"product"` (default) — Cartesian product of every spatial anchor × every
  time anchor. The right shape for dense gridded data (climate output,
  regular satellite revisits).
- `"coupled"` — explicit `(space, time)` anchor pairs from the spatial
  sampler's `anchors_`. The right shape for event-triggered patches
  (methane plume detections, Argo profile locations, storm tracks).

## Operator wrappers

The Patcher composes inside a `Sequential` via three Operator wrappers:

| Wrapper | Shape |
|---|---|
| `GridSampler(patcher)` | `Field → list[Patch]` |
| `ApplyToChips(operator)` | `list[Patch] → list[Patch]` (maps per-chip) |
| `Stitch(aggregation, domain)` | `list[Patch] → Field` |

```python
pipe = Sequential([
    GridSampler(patcher),
    ApplyToChips(model_op),
    Stitch(SpatialOverlapAdd(), domain=field.domain),
])
result = pipe(field)
```

## Streaming aggregations

Every `SpatialAggregation` carries a `streaming_safe: ClassVar[bool]`. The
canonical streaming-safe member is `SpatialOverlapAdd`, which accepts
`streaming=True, target_path=...` to accumulate into an on-disk
[zarr](https://zarr.dev) store instead of RAM. The exact streaming family
(`Sum`, `Mean`, `Variance`, `OverlapAdd`, `WeightedSum`, `InvVarWeightedMean`,
`HardVote`, `SoftVote`) is fully implemented; the approximate (sketch)
family (`ApproxQuantile`, `ApproxCardinality`, `ApproxMode`,
`StreamingHistogram`, `Reservoir`) ships as stub classes that raise
`NotImplementedError` with a pointer at the streamable substitute — they
land in v0.2.

## Optional extras

`geotoolz` keeps the base install slim and gates each non-raster Field
adapter behind an extra:

```bash
pip install 'geotoolz[grid]'           # XarrayField
pip install 'geotoolz[vector]'         # GeoPandasField
pip install 'geotoolz[point]'          # XvecField
pip install 'geotoolz[xarray-raster]'  # RioXarrayField
pip install 'geotoolz[streaming]'      # OverlapAdd(streaming=True)
pip install 'geotoolz[patch-full]'     # everything above
```

Each adapter raises a friendly `ImportError` pointing at the right extra if
the backend library is missing.

## Where the framework draws the line

- **Mesh / `uxarray`** (`UXarrayField`) is deferred to v0.2.
- **Sketch aggregations** ship as stubs only — see the substitute table in
  the docstrings.
- **Hierarchical Patcher-of-Patchers** is supported as a *recipe* on top of
  the framework rather than a dedicated class. See the streaming tutorial
  notebook.
- **Two-pass / global-context operators** (global normalisation,
  attention across patches) are explicitly out of scope; users write the
  two passes themselves on top of the existing primitives.
