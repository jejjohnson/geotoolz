# Query → Matchup → Patch: A Cross-Package Design

**Status:** Implemented — this document describes the shipped surface; §8 lists what is not built
**Author:** @jejjohnson + Claude
**Date:** 2026-05-23 (updated to the shipped API)
**Affects:** `geocatalog`, `geotoolz`, `geopatcher`

---

## 1. Context

Today the three packages cover discovery / transform / patching as separate concerns:

- **`geocatalog`** is a *local* spatiotemporal index over files already on disk. Its `GeoCatalog` Protocol, `GeoSlice` wire format, GeoParquet 1.1 persistence, and DuckDB SQL backend handle "what do I have, where, when?" for raster, vector, and xarray sources.
- **`geotoolz`** is a `pipekit.Operator` library of remote-sensing domain operations on `GeoTensor` (radiometry, indices, masking, single-source geom ops, segmentation, etc.). Every operator is serializable to YAML and composes into `Sequential` / `Graph` pipelines.
- **`geopatcher`** is a four-axis (Geometry × Sampler × Window × Aggregation) patching framework with three patcher types (`SpatialPatcher`, `TemporalPatcher`, `SpatioTemporalPatcher`) and `Field` adapters for raster / xarray / vector / xvec / rio-xarray. Streaming-first, numpy + scipy in the core.

What's missing is the layer **above** local catalogs — discovering remote granules from external systems (NASA earthaccess, STAC endpoints, Google Earth Engine), deciding what's interesting, persisting that decision, finding *matchups* between heterogeneous sources, and optionally staging bytes — and the layer **between** patching and these matched sources so a downstream sampler can read co-located neighborhoods across LEO, GEO, vector, and point-cloud modalities.

This design covers all three packages because the user-facing workflow crosses all three: a single `geocatalog` query produces matchups, those matchups feed a `geopatcher` composite Field, which calls `geotoolz` operators per anchor to align secondary sources.

## 2. Goals and non-goals

### Goals

1. Issue a query against any of {earthaccess, STAC, GEE, CMR} via a uniform `Source` Protocol; iterate or persist the results.
2. Persist queries themselves (not just their results) so workflows are reproducible and shareable.
3. Compute matchups (pairwise or N-way) between persisted catalog entries with explicit spatial + temporal tolerances; persist matchups as first-class catalog artifacts.
4. Stage / download bytes for a matchup set or query tag with caching, retry, and parallelism.
5. Express cross-modality coregistration (LEO ↔ GEO, raster ↔ points, raster ↔ point-cloud, vector ↔ raster) as standard `pipekit.Operator`s in geotoolz so they compose declaratively and serialize to YAML.
6. Let a downstream `geopatcher` sampler emit *matched patches* — joint local neighborhoods across the matched sources — without changing any existing patcher, geometry, window, or aggregation code.

### Non-goals

- A new query DSL. `bounds + interval + filters dict` is sufficient.
- Scheduling, background workers, or distributed orchestration. That stays in `pipekit` or downstream tooling.
- Auth UI. Defer to each library's native flow (`earthaccess.login`, `ee.Authenticate`).
- Replacing rasterio / pyproj / odc-geo. Geotoolz keeps wrapping them.
- Cross-cloud abstraction inside geocatalog. Object storage belongs to `geocloud` (staging downloads through `geocloud.files`); the readers keep their native URI handling (rasterio / GDAL, fsspec).
- Bayesian / probabilistic patches. Weights stay deterministic.

## 3. High-level architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  GEOCATALOG                                     │
│                                                                                 │
│   external source ──┐    ┌──────────────┐         ┌──────────────────────┐      │
│   (earthaccess,     │    │ Source.query │ ingest  │  items.parquet       │      │
│    STAC, CMR)       ├───►│              ├────────►│  queries.parquet     │      │
│                     │    │              │         │  matchups.parquet    │      │
│                     │    └──────────────┘         └────────┬─────────────┘      │
│                                                            │                    │
│                                                            ▼                    │
│                                                   ┌──────────────────┐          │
│                                                   │ matchup engine   │          │
│                                                   │ (spatial+temporal│          │
│                                                   │  STRtree join)   │          │
│                                                   └────────┬─────────┘          │
│                                                            ▼                    │
│                                                   ┌──────────────────┐          │
│                                                   │  staging layer   │          │
│                                                   │ (geocloud.files) │          │
│                                                   └────────┬─────────┘          │
└────────────────────────────────────────────────────────────┼────────────────────┘
                                                             │ resolved URIs
                                                             ▼ + GeoSlices
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  GEOPATCHER                                     │
│                                                                                 │
│   ┌───────────────┐     ┌───────────────────────────────────────────┐           │
│   │ primary Field │     │            MatchedField                   │           │
│   │ (e.g. GEO)    ├────►│   primary + {name: secondary Field}       │           │
│   └───────────────┘     │           + {name: coreg Operator}        │           │
│   ┌───────────────┐     │                                           │           │
│   │ sec. Field(s) ├────►│                                           │           │
│   │ (e.g. LEO,    │     │            implements Field protocol      │           │
│   │  vector, pc)  │     └────────────────┬──────────────────────────┘           │
│   └───────────────┘                      ▼                                      │
│                       ┌─────────────────────────────────────────┐               │
│                       │  MatchedSpatialPatcher(SpatialPatcher)  │               │
│                       │  → iter MatchedPatch (members per src)  │               │
│                       └─────────────────────────────────────────┘               │
└─────────────────────────────────────────┼───────────────────────────────────────┘
                                          │ calls per-anchor
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   GEOTOOLZ                                      │
│                                                                                 │
│   geotoolz.geom.coregister                                                      │
│   ─────────────────────────                                                     │
│     RasterToRasterLike        RasterToPoints      PointsToRaster                │
│     RasterToPointCloud        PointCloudToRaster  VectorToRasterAgg             │
│     (SwathToGrid / GridToSwath: designed, not built — §8)                       │
│                                                                                 │
│   geotoolz.compositing                                                          │
│   ────────────────────                                                          │
│     StackMatched              BlendMatched (IVW, weighted mean)                 │
│                                                                                 │
│   All are pipekit.Operator subclasses with get_config() → YAML serializable     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

Three boundary rules that drive the rest of the document:

1. **Geocatalog never transforms pixels.** It returns URIs and metadata; staging is opt-in; loading remains delegated to `georeader`/`rasterio`/`xarray`.
2. **Geotoolz operators are the only place coregistration logic lives.** They take 1–N `GeoTensor`s in, return one out, are stateless, and serialize. Geopatcher and geocatalog both *call* them; neither *contains* them.
3. **Geopatcher's core stays numpy+scipy.** `MatchedField` adds a composite Field that dispatches to operators (provided by the user, typically from geotoolz). The 4-axis machinery is untouched.

## 4. GeoCatalog: Sources, queries, matchups, staging

### 4.1 File layout

```
src/geocatalog/_src/
  base.py                      # GeoCatalog Protocol, CatalogRow, errors
  geoslice.py                  # GeoSlice
  grid.py                      # slice_to_window, is_grid_aligned, count_steps
  factory.py, ops.py           # open_catalog; query / intersect / union
  _schema.py, _lazy.py         # row schema; lazily resolved extras-gated names
  _extras.py                   # require_extra / install hints ("geotoolz-catalog[...]")

  backends/
    memory.py                  # InMemoryGeoCatalog
    duckdb_backend.py          # DuckDBGeoCatalog

  formats/                     # each format's builder and loader, side by side
    raster.py                  # build_raster_catalog, load_raster, load_raster_timeseries
    vector.py                  # build_vector_catalog, load_vector
    xarray_backend.py          # build_xarray_catalog, load_xarray

  sources/
    _base.py                   # Source Protocol, SourceRow, AuthStatus
    earthaccess.py             # EarthAccessSource
    stac.py                    # STACSource (+ planetary_computer() / earth_search())
    cmr.py                     # CMRSource (lightweight REST, no extra needed)
    gee.py                     # GEESource — scaffolding only (query raises NotImplementedError)
    _umm.py                    # UMM-G granule decoder shared by earthaccess / CMR

  storage/
    bundle.py                  # CatalogBundle, QueryRecord, source_row_to_gdf_row
    parquet.py                 # to_geoparquet / from_geoparquet, migrations
    streaming.py               # StreamingParquetWriter, append_files
    stac.py                    # from_stac_items / from_stac_search, to_stac_collection

  matchup/
    engine.py                  # matchup(), MatchupRow — in-memory STRtree join
    spatial.py                 # Intersects, IouAtLeast, CentroidWithin, Contains
    temporal.py                # NearestInTime, WithinWindow, Synchronous

  staging/
    stage.py                   # stage(), LocalCache

  patch/
    field_for.py               # field_for() — bridge to a geopatcher RasterField
    domain.py                  # CatalogDomain
```

Public namespaces (`geocatalog.sources`, `geocatalog.storage`,
`geocatalog.matchup`, `geocatalog.staging`, `geocatalog.patch`)
re-export these, each name from exactly one of them (see the
[API reference](../api/reference.md)).

### 4.2 `Source` Protocol

```python
# geocatalog/_src/sources/_base.py

class Source(Protocol):
    """A remote data catalog that can be queried by bounds + interval + filters."""

    name: str  # stable identifier, e.g. "earthaccess", "stac.pc", "cmr"

    def query(
        self,
        bounds: Bounds,                      # lon/lat (EPSG:4326)
        interval: pd.Interval | None = None,
        *,
        collection: str | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[SourceRow]: ...

    def auth_status(self) -> AuthStatus: ...
```

Adapters are imported lazily and gated by optional extras; a missing
extra raises `ModuleNotFoundError` naming `geotoolz-catalog[<extra>]`
when the adapter is constructed:

```toml
[project.optional-dependencies]
earthaccess = ["earthaccess>=0.10"]
stac        = ["pystac>=1.10", "pystac-client>=0.7", "planetary-computer>=1.0"]
gee         = ["earthengine-api>=0.1.380"]
sources-all = ["geotoolz-catalog[earthaccess,stac,gee]"]
```

### 4.3 `SourceRow` — normalized output of every adapter

`SourceRow` (output of `Source.query`) is distinct from the in-catalog
`CatalogRow` (yielded by `GeoCatalog.iter_rows`). `CatalogBundle.ingest`
maps each `SourceRow` to an items-table row with
`source_row_to_gdf_row`, promoting the primary asset to `filepath`.

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | granule UR / STAC item id / EE asset path |
| `source` | `str` | `"earthaccess"`, `"stac.pc"`, `"stac.es"`, `"cmr"`, … |
| `collection` | `str` | e.g. `MOD09GA`, `sentinel-2-l2a` |
| `geometry` | `shapely` geometry | footprint, lon/lat |
| `interval` | `pd.Interval` | observation interval, UTC, `closed="both"` |
| `assets` | `Mapping[str, str]` | STAC-style asset map `{"red": "s3://...", …}` |
| `properties` | `Mapping[str, Any]` | sensor-specific (cloud cover, orbit, …) |
| `provenance` | `Mapping[str, Any]` | `{query_id, query_tag, …}`, stamped on ingest |

The items table adds `filepath`, `crs`, `href_signed`, and stores
`assets` / `properties` / `provenance` as JSON strings; `start_time` /
`end_time` become the `IntervalIndex`. No GeoParquet schema migration was
needed: local catalogs and bundle items share the catalog schema, and a
bundle's extra columns are ordinary extras.

### 4.4 Persistence: a `CatalogBundle` directory

`CatalogBundle.to_directory(path)` writes a directory of Parquet files;
`CatalogBundle.from_directory(path)` reads it back:

```
my_catalog/
  items.parquet              # GeoParquet — one row per granule (the bundle's catalog)
  queries.parquet            # one row per ingest call (omitted when empty)
  matchups.parquet           # one row per matched tuple (omitted when empty)
  _meta.json                 # bundle_schema_version, target_crs, backend,
                             # created_at, updated_at
```

The write is atomic (staged next to `path`, then swapped in), and files
the bundle does not own are kept. `bundle.queries_df` and
`bundle.matchups_df` expose the sidecar tables as DataFrames; the items
are `bundle.catalog`, an `InMemoryGeoCatalog`. Items are keyed by
`(source, collection, id)`; matchups by `matchup_id`.

**`queries.parquet`** — one `QueryRecord` per `ingest` call:

| Column | Type |
|---|---|
| `query_id` | `str` (uuid4 hex; also stamped into each item's `provenance["query_id"]`) |
| `source` | `str` |
| `collection` | `str \| null` |
| `bounds_wkt` | `str` (WGS84) |
| `time_start`, `time_end` | `datetime \| null` |
| `filters_json` | `str` (JSON-encoded) |
| `created_at` | `datetime` (UTC) |
| `n_returned` | `int` |
| `tag` | `str \| null` (user label) |
| `notes` | `str \| null` |

**`matchups.parquet`** — one `MatchupRow` per matched tuple:

| Column | Type |
|---|---|
| `matchup_id` | `str` (content hash of strategy parameters + members; stable across re-runs) |
| `strategy` | `str` (label, e.g. `"IouAtLeast(threshold=0.2) & NearestInTime(dt='6h')"`) |
| `tolerance_json` | `str` (JSON-encoded `MatchupRow.tolerance`, e.g. `{"spatial": {"type": "IouAtLeast", "threshold": 0.3}, "temporal": {"type": "NearestInTime", "dt_sec": 3600.0}, "join": "all", "crs": "EPSG:4326"}`) |
| `member_ids` | `array<str>` (refs the items' `id`) |
| `member_sources`, `member_collections` | `array<str>` (parallel to `member_ids`) |
| `member_roles` | `array<str>` (`"primary"` / `"secondary"` / named roles) |
| `geometry_intersect_wkt` | `str` (WKT of the common footprint, in the matchup's working CRS) |
| `time_reference` | `datetime` |
| `time_offset_sec` | `array<float>` (per member, relative to `time_reference`) |
| `query_set` | `str \| null` (the `tag` passed to `matchup` / `write_matchups`) |

### 4.5 Discovery vs. ingest

A deliberate split between "I want to see what's out there" and "I want
to persist what's out there":

- `Source.query(...)` returns an `Iterator[SourceRow]` — for ad-hoc
  exploration; never writes.
- `CatalogBundle.ingest(source, bounds=..., interval=..., collection=...,
  filters=..., tag=...) -> query_id` appends the results to the items
  table, records a `QueryRecord`, and stamps each item's
  `provenance["query_id"]`. `on_duplicate=` decides what a re-ingested
  `(source, collection, id)` does.

Discovery, ingest, matchup and staging are Python APIs only. The
`geocatalog` CLI covers local catalogs (`build`, `query`, `stats`,
`info`, `convert`, `migrate`); see the [CLI reference](../cli.md).

```python
import pandas as pd
from geocatalog.storage import CatalogBundle
from geocatalog.sources import EarthAccessSource

bundle = CatalogBundle.empty(crs="EPSG:4326")
june = pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both")
query_id = bundle.ingest(
    EarthAccessSource(),
    collection="MOD09GA",
    bounds=(-10, 35, 5, 45),
    interval=june,
    tag="iberia_summer24",
)
bundle.to_directory("my_catalog/")
```

### 4.6 Matchup engine

```python
# geocatalog/_src/matchup/engine.py

def matchup(
    primary: GeoCatalog | CatalogBundle | Iterable[SourceRow],
    secondary: GeoCatalog | CatalogBundle | Iterable[SourceRow]
    | Mapping[str, GeoCatalog | CatalogBundle | Iterable[SourceRow]],  # role -> input for N-way
    *,
    spatial: SpatialStrategy,           # IouAtLeast(0.2), CentroidWithin(buffer=5_000.0)
    temporal: TemporalStrategy,         # NearestInTime(dt="6h"), WithinWindow(start=, end=)
    join: Literal["all", "any"] = "all",
    tag: str | None = None,
    crs: CRS | None = None,             # working CRS; default: primary catalog's, else EPSG:4326
    include_self: bool = False,
) -> Iterator[MatchupRow]: ...
```

**Spatial strategies** (`spatial.py`):
- `Intersects()` — non-zero intersection
- `IouAtLeast(threshold)` — IoU ≥ threshold
- `CentroidWithin(buffer)` — secondary centroid within `buffer` of the primary footprint
- `Contains()` — secondary fully contained in primary footprint

**Temporal strategies** (`temporal.py`):
- `NearestInTime(dt)` — pick the secondary nearest in time, only if Δt ≤ dt
- `WithinWindow(start, end)` — all secondaries in [t+start, t+end] relative to primary
- `Synchronous(tolerance="0s")` — overlapping observation intervals

Implementation: an in-memory join — an STRtree per secondary role over footprints in the working CRS, the temporal strategy selecting candidates by position, the spatial strategy as the truth gate. Inputs are filtered with the catalog's own `query` (or `InMemoryGeoCatalog.where`) before the call; there is no selector argument. Candidates are ordered by `(source, collection, id)` so tie-breaks and `matchup_id`s are the same on every run; intervals are compared in UTC and rows with no times never match. Distances (`CentroidWithin.buffer`) are in units of the working CRS — pass a projected `crs=` for metres. Persist the output with `CatalogBundle.write_matchups` (`matchups.parquet`).

### 4.7 Staging layer

Explicit, never automatic:

```python
# geocatalog/_src/staging/stage.py

def stage(
    catalog: InMemoryGeoCatalog,
    *,
    dest: PathLike | str | None = None,      # cache root; default $GEOCATALOG_CACHE or ~/.cache/geocatalog
    assets: list[str] | None = None,         # asset keys; None = all
    parallel: int = 8,
    cache: LocalCache | None = None,
    retries: int = 3,
    on_error: str = "raise",                 # or "skip": keep the URI, continue
) -> InMemoryGeoCatalog: ...
```

- Returns a new catalog whose asset map has been rewritten to local paths; `filepath` follows the row's primary asset, and a `staged_from` column (JSON, keyed like `assets`) preserves the original URIs.
- Local paths are used in place; remote schemes download through `geocloud.files` (`[cloud]` extra) into a temp file renamed into place, one download per distinct URI. The cache key is the URI (expiring signature parameters removed), not the content.
- Reuses the shared retry/backoff policy of the catalog readers for transient failures (fatal errors such as `FileNotFoundError` are not retried).
- Earth Engine assets are not staged (there is no `ee` download path).

The staged catalog is a normal `GeoCatalog`, so the existing `load_raster` / `load_vector` / `load_xarray` loaders read the local files in place. `field_for(staged, slice_, asset=...)` (the `[patch]` extra) mosaics the rows a slice selects into one `geopatcher.RasterField`.

## 5. GeoToolz: `geom.coregister` and `compositing.matched`

All cross-modality alignment lives in geotoolz under the existing `geom` namespace, plus two new operators in `compositing`. Each follows the package's two-tier convention: a pure numpy/scipy primitive in `_src/array.py`, a `pipekit.Operator` wrapper in `_src/operators.py`.

### 5.1 New file layout

```
src/geotoolz/
  geom/
    _src/
      array.py          # existing array primitives (reproject, resample, rasterize, …)
      operators.py      # existing Operator wrappers (Reproject, Resample, Rasterize, …)
      coregister/                                     # NEW
        __init__.py
        array.py        # numpy/scipy primitives for cross-modality alignment
        operators.py    # pipekit.Operator wrappers
    coregister.py       # public re-exports: from geotoolz.geom.coregister import *
  compositing/
    _src/
      operators.py      # + StackMatched, BlendMatched (NEW)
```

### 5.2 Operator catalog

All operators are `pipekit.Operator` subclasses, `__call__(*inputs) → GeoTensor`, with `get_config()` for YAML round-trip.

| Operator | Inputs → Output | Builds on | Notes |
|---|---|---|---|
| `RasterToRasterLike(resampling=…)` | `(src, like) → aligned_src` | `Reproject` + `Resample` | Convenience for the common case; one op instead of two |
| `SwathToGrid(method="bowtie_aware", target_crs=…, target_res=…)` | `swath → grid` | rasterio + per-pixel lat/lon | **Not built.** Track-B gap; handles MODIS/VIIRS bowtie |
| `GridToSwath(time_match="nearest", dt_max="15min")` | `(grid_series, swath_like) → grid_at_swath_geom` | rasterio + temporal index | **Not built.** GEO → LEO acquisition geometry |
| `RasterToPoints(extract="nearest" \| "bilinear")` | `(raster, points) → xvec_cube` | xvec | Extract raster at point geometries → vector cube |
| `PointsToRaster(method="binned_stat", stat="mean", like=…)` | `(points, like) → raster` | scipy.stats.binned_statistic_2d | Bin point cube into grid |
| `RasterToPointCloud(k=…, max_radius=…)` | `(raster, cloud) → cloud_with_attrs` | scipy.spatial.KDTree | Sample raster onto cloud nodes |
| `PointCloudToRaster(method="idw" \| "binned_stat", like=…)` | `(cloud, like) → raster` | scipy KDTree + IDW | Rasterize point cloud |
| `VectorToRasterAgg(agg="mean" \| "majority" \| "count", like=…)` | `(vector, like) → raster` | extends `Rasterize` | Aggregation policy for overlapping features |
| `StackMatched(order=…, fill_value=NaN)` | `[t1, t2, …] → multi_band` | numpy stack + reproject-to-like | Compositing-style: N aligned tensors → 1 multi-band GeoTensor |
| `BlendMatched(weights=… \| "ivw", method="mean")` | `[t1, t2, …] → blended` | numpy weighted mean | IVW = inverse-variance weighting |

**xvec dependency.** Added as a new optional extra:

```toml
[project.optional-dependencies]
vector-cube = ["xvec>=0.4"]
```

`RasterToPoints` / `PointsToRaster` require it; the rest of geotoolz is unaffected.

### 5.3 Operator signatures (illustrative)

```python
# geotoolz/geom/_src/coregister/operators.py

class RasterToRasterLike(Operator):
    resampling: Resampling = Resampling.bilinear

    def __call__(self, src: GeoTensor, like: GeoTensor) -> GeoTensor:
        ...

    def get_config(self) -> dict:
        return {"resampling": self.resampling.name}


class SwathToGrid(Operator):
    method: Literal["bowtie_aware", "naive"] = "bowtie_aware"
    target_crs: str
    target_res: tuple[float, float]
    bounds: Bounds | None = None

    def __call__(self, swath: GeoTensor) -> GeoTensor: ...
    def get_config(self) -> dict: ...


class RasterToPoints(Operator):
    extract: Literal["nearest", "bilinear"] = "bilinear"
    out_var: str = "value"

    def __call__(self, raster: GeoTensor, points: "xvec.DataArray") -> "xvec.DataArray": ...
```

### 5.4 What this unlocks beyond matchups

Because these are stateless `pipekit.Operator`s, they're useful outside any matchup or patching context:

- A flat pipeline that stacks Sentinel-2 + Landsat for a single AOI: `Sequential([RasterToRasterLike(), StackMatched()])`.
- A station-validation script: `RasterToPoints()` to extract model output at in-situ buoy locations.
- A point-cloud-to-DEM conversion: `PointCloudToRaster(method="idw")`.

This is the payoff of putting them in geotoolz rather than burying them inside geopatcher.

## 6. GeoPatcher: `MatchedField` and `MatchedPatch`

### 6.1 File layout

```
src/geopatcher/_src/
  matched/
    __init__.py
    field.py        # MatchedField composite Field (+ footprint indexer)
    patch.py        # MatchedPatch / MatchedTemporalPatch / MatchedSpatioTemporalPatch
    patcher.py      # MatchedSpatialPatcher / MatchedTemporalPatcher /
                    # MatchedSpatioTemporalPatcher (split, per-source merge)
  spatial/, time/, …                         # unchanged
```

Public re-export at `geopatcher.matched`. There is no separate
aggregation module: per-source merging is the matched patchers'
``secondary_aggregators`` mapping (one ordinary aggregation per
secondary).

### 6.2 `MatchedField` — a composite Field

```python
# geopatcher/_src/matched/field.py

@dataclass(eq=False)
class MatchedField:
    """N co-registered Fields presented as one Field.

    Satisfies the `Field` Protocol via the primary (anchor space, CRS, domain).
    On select(), reads each secondary over the primary chip's footprint and
    pipes it through its coreg callable.
    """
    primary: Field
    secondaries: Mapping[str, Field]
    coreg: Mapping[str, Callable]   # any Callable; pipekit.Operator (e.g. from
                                    # geotoolz.geom.coregister) is the recommended choice
                                    # — see ADR-003 for why the type is the broader Callable.
    valid_mask: bool = True         # matched patchers emit per-source nodata masks

    @property
    def domain(self) -> Domain:
        return self.primary.domain

    def select(self, indexer: Any) -> dict[str, Any]:
        primary_data = self.primary.select(indexer)
        out = {"primary": primary_data}
        for name, sec in self.secondaries.items():
            # Raster domains: the primary window's bounds (primary CRS) →
            # a window on the secondary's own grid, rounded outward.
            raw = sec.select(_footprint_indexer(indexer, self.primary.domain, sec.domain))
            out[name] = self.coreg[name](raw, primary_data)   # any geotoolz op
        return out
```

Properties this design preserves:

1. **Existing samplers, geometries and windows work unchanged.** `MatchedField` *is* a `Field`: they see only the primary's domain, so anchor placement and indexers are the single-source ones.
2. **Heterogeneous grids are read by footprint.** A secondary on another resolution, origin or CRS is read over the primary chip's geographic footprint, so a reproject-to-like coreg (`RasterToRasterLike`) fills the whole chip. Non-raster secondaries (grid / vector / points) carry no affine transform and are read with the primary's indexer — they must share its index space.
3. **Geopatcher's core stays numpy + scipy.** The `coreg` dict holds opaque callables; geopatcher never imports `geotoolz`.
4. **Coregistration logic is reusable outside patching.** Same operators serve flat pipelines, validation scripts, and matchup builds.

`select` returns the per-source **dict**, not a carrier. Splitting and
merging go through the matched patchers (§6.4): a plain
`SpatialPatcher.split(matched_field)` yields `Patch(data=dict)` with no
masks, and a plain `SpatialPatcher.merge` cannot aggregate that dict.

### 6.3 `MatchedPatch` — the carrier

```python
# geopatcher/_src/matched/patch.py

@dataclass(eq=False)
class MatchedPatch:
    anchor: Anchor
    members: dict[str, Patch]                    # "primary" + secondaries by name
    valid_mask: dict[str, np.ndarray] | None     # False = nodata / NaN / off-swath
    weights: dict[str, np.ndarray] | None = None # each member's window weights
```

`MatchedTemporalPatch` (members are `TemporalPatch`es) and
`MatchedSpatioTemporalPatch` (`SpatioTemporalPatch`es, plus `space` /
`time` anchors) are the temporal mirrors.

`MatchedPatch` does not subclass `Patch` — it's a sibling carrier. Operators that want to consume one explicitly type against `MatchedPatch`; legacy operators see a single primary `Patch` via `mp.members["primary"]`.

`valid_mask[name]` is False where the member equals its carrier's
declared nodata (`GeoTensor.fill_value_default`, rioxarray `rio.nodata`)
and, for float data, where it is NaN / ±inf; a bare ndarray declares no
nodata, so only the float test applies. A matched patch also carries the
`Patch` release lifecycle (`close()` / `with mp: ...`) for
`max_in_flight`.

### 6.4 Splitting and merging: the matched patchers

```python
class MatchedSpatialPatcher:
    primary: SpatialPatcher                                  # sampler / geometry / window /
                                                             # primary aggregation / on_error
    secondary_aggregators: Mapping[str, spatial.aggregation.Aggregation]  # one per secondary (opt-in)

    def split(self, mfield, hooks=None, *, prefetch=0, journal=None,
              cache=None, max_in_flight=None) -> Iterator[MatchedPatch]: ...
    def merge(self, patches, mfield, hooks=None) -> dict[str, Any]: ...
    def merge_to_field(self, patches, mfield, hooks=None) -> dict[str, Any]: ...
```

- **Split** drives `primary.split` over the `MatchedField` and unpacks
  each per-source dict into a `MatchedPatch`. The primary's `on_error`
  policy covers every source's read and coregistration: `"skip"` drops
  the anchor, `"mask"` yields a `MatchedPatch` whose members are all-NaN
  on the primary's grid with an all-False `valid_mask`. Hooks fire once
  per matched patch with the members' summed bytes. `cache=` caches each
  source's raw read under its own identity (the coregistration still
  runs).
- **Merge** is per source and returns `dict[str, output]` — the primary
  under `"primary"`, each secondary that has an entry in
  `secondary_aggregators` under its name. Every value is the
  aggregation's raw output on the **primary's** grid (a bare
  `np.ndarray` for the dense aggregations; ADR-007), because the
  coregistration mapped each secondary there.
  `MatchedSpatialPatcher.merge_to_field` rebuilds each one through the
  primary's `with_data` (primary transform / CRS, each source's dtype).
- `MatchedTemporalPatcher` reads every source's whole series once
  (an indexer covering the primary's extent — the full `Window` for
  raster, `{}` for a `GridDomain`, or an explicit `indexer=`), checks
  every source has the primary's time length (a different cadence must
  be resampled by the coreg callable), and slices them in lockstep;
  `n_anchors` / `anchors` touch the primary only.
  `MatchedSpatioTemporalPatcher` drives the primary
  `SpatioTemporalPatcher` itself — its `coupling`, `coord=`, chip read
  pipeline, `on_error` policy, hooks, journal and backpressure, plus a
  per-source `cache` — checks every source of a chip has the primary's
  time length, slices them in lockstep, and merges to
  `{name: [(spatial_anchor, temporal_result), …]}`.

The matched patchers are the only API addition relative to the
single-source patchers, and they are new classes, so existing pipelines
are untouched.

### 6.5 Streaming guarantees

`MatchedSpatialPatcher.split` (and the temporal mirrors) yield one
matched patch at a time; with `max_in_flight` each matched patch holds
the read's backpressure slot until it is closed. The spatial and
temporal `merge` consume the patches in **one pass** and stream every
source: each source's aggregation runs on its own worker thread fed
through a small bounded queue, so only a few patches per source are
resident and a streaming aggregation (e.g. `spatial.aggregation.OverlapAdd` with Zarr
backing) keeps its memory bound on every source. Each aggregation gets
the usual `streaming_safe` check (a warning, or an error under
`set_strict`). `MatchedSpatioTemporalPatcher.merge` streams every
source the same way, but each source groups its patches by spatial
anchor and holds those groups until the pass ends.

**Future work.** A per-source `iter_patches` view and a dedicated
`MatchedAggregator` were sketched in earlier drafts of this section;
neither exists. The patchers above cover their use cases today.

## 7. End-to-end walkthrough: MODIS × Sentinel-2 patches over Iberia

```python
import pandas as pd

import geocatalog as gc
from geocatalog.matchup import IouAtLeast, NearestInTime
from geocatalog.sources import EarthAccessSource, STACSource
import geopatcher as gp
from geopatcher.matched import MatchedField, MatchedSpatialPatcher
from geotoolz.geom.coregister import RasterToRasterLike

june = pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both")

# 1. Discover & ingest into one bundle
bundle = gc.storage.CatalogBundle.empty(crs="EPSG:32629")
bundle.ingest(
    EarthAccessSource(),
    collection="MOD09GA",
    bounds=(-10, 35, 5, 45), interval=june,
    tag="iberia_summer24",
)
bundle.ingest(
    STACSource.planetary_computer(),
    collection="sentinel-2-l2a",
    bounds=(-10, 35, 5, 45), interval=june,
    filters={"eo:cloud_cover": {"lt": 20}},
    tag="iberia_summer24",
)

# 2. Build and persist matchups (inputs are filtered before the call)
modis = bundle.catalog.where("collection == 'MOD09GA'")
s2 = bundle.catalog.where("collection == 'sentinel-2-l2a'")
bundle.write_matchups(
    gc.matchup.matchup(modis, s2, spatial=IouAtLeast(0.2), temporal=NearestInTime(dt="6h")),
    tag="modis_s2_pairs_v1",
)
bundle.to_directory("my_catalog/")

# 3. Stage bytes for the assets we need
staged_modis = gc.staging.stage(modis, dest="./staged/", assets=["red", "nir"])
staged_s2 = gc.staging.stage(s2, dest="./staged/", assets=["red", "nir"])

# 4. One RasterField per source over the area of interest
aoi = gc.GeoSlice(
    bounds=(-9.5, 38.5, -8.5, 39.5), interval=june,
    resolution=(0.005, 0.005), crs="EPSG:4326",
)
modis_field = gc.patch.field_for(staged_modis, aoi, asset="red")
s2_field = gc.patch.field_for(staged_s2, aoi, asset="red")

matched = MatchedField(
    primary=modis_field,
    secondaries={"s2": s2_field},
    coreg={"s2": RasterToRasterLike(resampling="bilinear")},
)

# 5. Patch with any existing geopatcher sampler
patcher = MatchedSpatialPatcher(
    primary=gp.SpatialPatcher(
        geometry=gp.spatial.geometry.Rectangular(size=(512, 512)),
        sampler=gp.spatial.sampler.RegularStride(step=(256, 256)),
        window=gp.spatial.window.Boxcar(),
        aggregation=gp.spatial.aggregation.Mean(),
    ),
    secondary_aggregators={"s2": gp.spatial.aggregation.Mean()},
)

for matched_patch in patcher.split(matched):
    modis_chip = matched_patch.members["primary"].data     # (bands, H, W) MODIS
    s2_chip    = matched_patch.members["s2"].data          # (bands, H, W) S2 at MODIS grid
    mask       = matched_patch.valid_mask["s2"]            # where S2 is valid
    # → into model / training loop / further geotoolz pipeline

# 6. Merge per source back onto the MODIS grid (georeferenced carriers)
fields = patcher.merge_to_field(patcher.split(matched), matched)  # {"primary": …, "s2": …}
```

Three things to notice:

- The user never touches a coregistration class — they pick a `geotoolz` operator.
- The matchup is reproducible: `matchup_id`s are content hashes, so re-running step 2 over the same items yields the same ids, and `write_matchups` replaces them in place instead of duplicating them.
- Every step is independently usable: ingest without matchup, matchup without staging, stage without patching, patch without matchup (just `MatchedField(primary, {}, {})`).

## 8. Implementation status

Every piece below ships unless marked otherwise.

- **geocatalog** — `Source` Protocol, `SourceRow`, `EarthAccessSource`,
  `STACSource`, `CMRSource`; `CatalogBundle` (`ingest`,
  `write_matchups`, `to_directory` / `from_directory`); the `matchup`
  engine and its strategies; `stage` / `LocalCache`; `field_for`.
  Not shipped: `GEESource` is scaffolding (its `query` and
  `auth_status` raise `NotImplementedError`); there are no `search` /
  `ingest` / `matchup` / `stage` CLI verbs; the matchup join is
  in-memory (no DuckDB implementation); Earth Engine assets are not
  staged.
- **geotoolz** — `geom.coregister` (`RasterToRasterLike`,
  `RasterToPoints`, `PointsToRaster`, `RasterToPointCloud`,
  `PointCloudToRaster`, `VectorToRasterAgg`) and
  `compositing.StackMatched` / `BlendMatched`; `RasterToPoints` /
  `PointsToRaster` need the `[vector-cube]` extra (xvec). Not shipped:
  `SwathToGrid`, `GridToSwath`.
- **geopatcher** — `geopatcher.matched`: `MatchedField`,
  `MatchedPatch` (+ temporal / spatio-temporal mirrors) and the three
  matched patchers (§6).

## 9. Open questions

Resolved questions keep their decision for the record.

| # | Question | Decision |
|---|---|---|
| 1 | Matchup catalog as sibling Parquet vs. separate artifact? | **Sibling** (§4.4) |
| 2 | GEE scope: enumerate only, or run `ee.Image` recipes in staging? | **Enumerate only** — and `GEESource` is still scaffolding |
| 3 | STAC adapter: one generic + named factory helpers, or distinct subclasses per provider? | **Generic `STACSource(endpoint=...)` with class-method factories** |
| 4 | Auth surface: defer to libraries, or add `geocatalog auth status` aggregator? | **Defer**; `Source.auth_status()` per adapter |
| 5 | `MatchedPatch` subclasses `Patch` or sibling carrier? | **Sibling** (avoid LSP issues) |
| 6 | `MatchedField.coreg` typed as `dict[str, Operator]` (pipekit) or untyped `Callable`? | **Resolved (ADR-003):** typed as `Mapping[str, Callable]`; `pipekit.Operator` is the recommended value but not required, so geopatcher's core stays framework-free. |
| 7 | Should the matchup engine emit a *new catalog* or in-place new rows in `items.parquet`? | **New rows in `matchups.parquet`**; `items` stays atoms |
| 8 | Cache scope for staging: per-catalog, per-user, or per-host? | **Per-user** (`~/.cache/geocatalog/`), overridable by `$GEOCATALOG_CACHE` |
| 9 | Ship `BlendMatched(method="ivw")` with the first coregistration operators, or defer (uncertainty maps not always present)? | **Shipped** alongside `StackMatched` |
| 10 | Where does this design doc live? | `docs/catalog/design/query-matchup.md` in the geotoolz workspace; geopatcher's side is mirrored in its ADR-003 |

## 10. Appendix A: alternatives considered

### A1. Source adapters as plugins (entry points) vs. in-tree modules

**Chosen:** in-tree modules under `_src/sources/`, behind optional extras.
**Rejected:** entry-point plugin system. Premature; locks in a contract before we know what real adapters need. We can always promote to entry points later without breaking the in-tree code.

### A2. Coregistration strategies as a new `CoregistrationStrategy` class in geopatcher

**Chosen:** plain `pipekit.Operator`s in geotoolz, held in a `dict[str, Operator]` on `MatchedField`.
**Rejected:** dedicated `CoregistrationStrategy` ABC in geopatcher. Reasons:
- Duplicates work geotoolz's `geom` module already does for single-source ops.
- Forces geopatcher to import or re-implement reprojection, rasterization, KDTree binning.
- Breaks the "geopatcher core is numpy + scipy only" invariant.
- Loses YAML serializability that `pipekit.Operator` gives for free.

### A3. Merge `items.parquet` and `matchups.parquet` into one table

**Chosen:** two tables.
**Rejected:** unified "rows-with-optional-members" schema. Reasons:
- `items` are atoms; `matchups` are compositions of atoms. Different cardinality semantics (1:1 vs. 1:N).
- DuckDB joins are cheap; one combined schema would be sparse and confusing.

### A4. Async-first geocatalog ingest

**Chosen:** sync `Source.query` returning an `Iterator`. Internal parallelism via `concurrent.futures` where adapters benefit.
**Rejected:** `AsyncSource` Protocol. Adapters' upstream libraries (`earthaccess`, `pystac-client`, `ee`) are sync; async would mostly be wrapping with `asyncio.to_thread`. Revisit if `aiohttp`-native adapters appear.

### A5. Staging via symlinks vs. real download

**Chosen:** real download into cache (with re-use). Optionally support `--symlink` for local URIs.
**Rejected:** symlink-only. Defeats the purpose for cloud URIs; complicates cache invalidation.

## 11. Appendix B: API surface summary

```
geocatalog
  .sources
    .Source                         # Protocol
    .SourceRow                      # normalized output of every Source.query
    .AuthStatus                     # Source.auth_status() result
    .EarthAccessSource              # adapter
    .STACSource                     # adapter (+ .planetary_computer(), .earth_search())
    .GEESource                      # scaffolding only (not implemented)
    .CMRSource                      # adapter
  .storage
    .CatalogBundle                  # .empty / .from_catalog / .ingest / .write_matchups /
                                    # .to_directory / .from_directory
    .QueryRecord                    # one queries.parquet row
  .matchup
    .matchup(...)                   # functional; `geocatalog.matchup` is an ordinary module
    .MatchupRow                     # dataclass
    .IouAtLeast, .CentroidWithin, .Intersects, .Contains   # spatial strategies
    .NearestInTime, .WithinWindow, .Synchronous            # temporal strategies
  .staging
    .stage(...)                     # functional
    .LocalCache                     # cache class
  .patch
    .field_for(...)                 # staged catalog -> geopatcher RasterField ([patch])
    .CatalogDomain                  # one GeoSlice per catalog row
  .GeoCatalog                       # Protocol (unchanged)
  .GeoSlice
  .backends
    .DuckDBGeoCatalog, .InMemoryGeoCatalog

geotoolz.geom.coregister
  .RasterToRasterLike
  .RasterToPoints                   # requires [vector-cube] extra (xvec)
  .PointsToRaster                   # requires [vector-cube] extra
  .RasterToPointCloud
  .PointCloudToRaster
  .VectorToRasterAgg

geotoolz.compositing
  .StackMatched
  .BlendMatched                     # method="mean" | "weighted_mean" | "ivw"

geopatcher.matched
  .MatchedField
  .MatchedPatch
  .MatchedSpatialPatcher
  .MatchedTemporalPatcher           # temporal mirror
  .MatchedSpatioTemporalPatcher     # spatio-temporal mirror
```
