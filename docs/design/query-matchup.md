# Query → Matchup → Patch: A Cross-Package Design

**Status:** Draft, for discussion
**Author:** @jejjohnson + Claude
**Date:** 2026-05-23
**Affects:** `geocatalog`, `geotoolz`, `geopatcher`
**Branch (all repos):** `claude/geocatalog-query-matchup-design-zHXc4`

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
- Cross-cloud abstraction. `fsspec` + the libraries' native URI handling is enough.
- Bayesian / probabilistic patches. Weights stay deterministic.

## 3. High-level architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  GEOCATALOG                                     │
│                                                                                 │
│   external source ──┐    ┌──────────────┐         ┌──────────────────────┐      │
│   (earthaccess,     │    │              │         │  items.parquet       │      │
│    STAC, GEE, CMR)  ├───►│ Source.query ├────────►│  queries.parquet     │      │
│                     │    │              │         │  matchups.parquet    │      │
│                     │    └──────────────┘         └────────┬─────────────┘      │
│                                                            │                    │
│                                                            ▼                    │
│                                                   ┌──────────────────┐          │
│                                                   │ matchup engine   │          │
│                                                   │ (spatial+temporal│          │
│                                                   │  join, DuckDB)   │          │
│                                                   └────────┬─────────┘          │
│                                                            ▼                    │
│                                                   ┌──────────────────┐          │
│                                                   │  staging layer   │          │
│                                                   │  (fsspec cache)  │          │
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
│                       │  any existing SpatialPatcher / sampler  │               │
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
│     SwathToGrid               GridToSwath         RasterToPointCloud            │
│     VectorToRasterAgg         PointCloudToRaster                                │
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

### 4.1 New file layout

```
src/geocatalog/_src/
  base.py                      # GeoCatalog Protocol  [existing]
  geoslice.py                  # GeoSlice             [existing]
  memory.py, duckdb_backend.py # backends             [existing]
  raster.py, vector.py, xarray_backend.py            [existing]
  parquet.py, streaming.py, ops.py                   [existing]
  domain.py, _cli.py                                 [existing]

  sources/                     # NEW
    __init__.py                # Source Protocol, CatalogRow dataclass, registry
    earthaccess.py             # EarthAccessSource
    stac.py                    # STACSource (+ planetary_computer() / earth_search() helpers)
    gee.py                     # GEESource (asset enumeration only in v1)
    cmr.py                     # CMRSource (lightweight REST fallback)
    _extras.py                 # friendly ImportError messages

  matchup/                     # NEW
    __init__.py                # public API: matchup(), MatchupRow
    spatial.py                 # intersects, iou_threshold, buffer_contains, centroid_within
    temporal.py                # nearest_in_time, within_window, synchronous
    engine.py                  # combines strategies; emits MatchupRows; SQL-where-possible

  staging/                     # NEW
    __init__.py                # public API: stage(), LocalCache
    cache.py                   # fsspec-backed cache, key = (uri, asset)
    download.py                # parallel fetch, retry/backoff (shared with PR #51)
    gee.py                     # GEE materialization (ee.Image.getDownloadURL)
```

Top-level re-exports follow the existing pattern:

```python
# src/geocatalog/sources/__init__.py
from geocatalog._src.sources import (
    Source, CatalogRow, EarthAccessSource, STACSource, GEESource, CMRSource,
)
# src/geocatalog/matchup/__init__.py
from geocatalog._src.matchup import matchup, MatchupRow
# src/geocatalog/staging/__init__.py
from geocatalog._src.staging import stage, LocalCache
```

### 4.2 `Source` Protocol

```python
# src/geocatalog/_src/sources/__init__.py

class Source(Protocol):
    """A remote data catalog that can be queried by bounds + interval + filters."""

    name: str  # stable identifier, e.g. "earthaccess", "stac.pc", "gee"

    def query(
        self,
        bounds: Bounds,
        interval: pd.Interval | None = None,
        *,
        collection: str | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[CatalogRow]: ...

    def auth_status(self) -> AuthStatus: ...
```

Adapters live in `_src/sources/<name>.py`. None are required at install time — all guarded by try/except + `_extras.py`, matching the geopatcher pattern. New optional extras in `pyproject.toml`:

```toml
[project.optional-dependencies]
earthaccess = ["earthaccess>=0.10"]
stac        = ["pystac-client>=0.7", "planetary-computer>=1.0"]
gee         = ["earthengine-api>=0.1.380"]
sources-all = ["geocatalog[earthaccess,stac,gee]"]
```

### 4.3 `CatalogRow` — normalized output schema

A single schema that every adapter must produce. This is also the in-memory row shape for ingested entries and a strict superset of today's row schema (so existing in-memory + DuckDB backends continue to work without migration).

| Column | Type | Notes |
|---|---|---|
| `id` | `str` | granule UR / STAC item id / EE asset path |
| `source` | `str` | `"earthaccess"`, `"stac.pc"`, `"gee"`, `"cmr"` |
| `collection` | `str` | e.g. `MOD09GA`, `sentinel-2-l2a`, `COPERNICUS/S2_SR` |
| `geometry` | `shapely.Geometry` | footprint, in catalog target CRS |
| `time_start`, `time_end` | `datetime` (UTC) | observation interval |
| `assets` | `JSON` | `{"red": "s3://...", "nir": "..."}` — STAC-style asset map |
| `properties` | `JSON` | sensor-specific (cloud_cover, sza, orbit_number, …) |
| `_provenance` | `JSON` | `{query_id, fetched_at, source_version}` |
| `_schema_version` | `int` | existing column, reused |

Migration from today's catalog: existing `path: str` becomes `assets: {"default": path}` and `source: "local"`. The existing schema-migrations framework (PR #39) gets its first non-empty registration: `v0 → v1` performs this remapping.

### 4.4 Persistence: a bundle of three Parquet files

A catalog becomes a *directory* of GeoParquet files (already idiomatic for partitioned writes):

```
my_catalog/
  items.parquet              # one row per granule (CatalogRow schema)
  queries.parquet            # one row per Source.query() invocation
  matchups.parquet           # one row per matched tuple
  _meta.json                 # schema_version, target_crs, created_at
```

Both new sibling tables are first-class — they have `query_id` / `matchup_id` primary keys, and DuckDB views join them to `items` for everything (lineage, "which items came from query X", "members of matchup Y").

**`queries.parquet` schema:**

| Column | Type |
|---|---|
| `query_id` | `str` (uuid) |
| `source` | `str` |
| `collection` | `str` |
| `bounds_wkt` | `str` |
| `time_start`, `time_end` | `datetime` |
| `filters_json` | `str` (JSON-encoded) |
| `created_at` | `datetime` |
| `n_returned` | `int` |
| `tag` | `str \| null` (user label) |
| `notes` | `str \| null` |

**`matchups.parquet` schema:**

| Column | Type |
|---|---|
| `matchup_id` | `str` (uuid) |
| `strategy` | `str` (`"nearest_in_time"`, `"within_window"`, `"synchronous"`) |
| `tolerance_json` | `str` (e.g. `{"dt": "1h", "spatial": "iou>0.3"}`) |
| `member_ids` | `array<str>` (refs `items.id`) |
| `member_sources` | `array<str>` (parallel to `member_ids`) |
| `member_roles` | `array<str>` (`"primary"` / `"secondary"` / etc.) |
| `geometry_intersect` | `shapely` (the common footprint) |
| `time_reference` | `datetime` |
| `time_offset_sec` | `array<float>` (per member, relative to `time_reference`) |
| `created_at` | `datetime` |
| `query_set` | `str \| null` (tag) |

### 4.5 Discovery vs. ingest verbs

A deliberate split between "I want to see what's out there" and "I want to persist what's out there":

- `Source.query()` returns an `Iterator[CatalogRow]` — for ad-hoc exploration; never writes.
- `catalog.ingest(source, query) → query_id` materializes results into `items.parquet`, writes a `queries.parquet` row, and stamps each item's `_provenance.query_id`.

CLI form:

```bash
# Discover (one-shot, prints, doesn't persist)
geocatalog search earthaccess MOD09GA --bbox -10 35 5 45 \
    --start 2024-06-01 --end 2024-06-30 --limit 50

geocatalog search stac.pc sentinel-2-l2a --bbox -10 35 5 45 \
    --start 2024-06-01 --end 2024-06-30 --filter "eo:cloud_cover<20"

# Ingest into catalog (persists items + records the query)
geocatalog ingest earthaccess MOD09GA --bbox -10 35 5 45 \
    --start 2024-06-01 --end 2024-06-30 --tag "iberia_summer24" \
    --out my_catalog/

# Matchup across already-ingested items
geocatalog matchup my_catalog/ \
    --primary  "source=earthaccess,collection=MOD09GA" \
    --secondary "source=stac.pc,collection=sentinel-2-l2a" \
    --strategy nearest_in_time --dt 6h --spatial iou>0.2 \
    --tag "modis_s2_pairs_v1"

# Stage / download for a matchup set or query tag
geocatalog stage my_catalog/ --matchup-tag modis_s2_pairs_v1 \
    --dest ./staged/ --asset red,nir,scl --parallel 8
```

### 4.6 Matchup engine

```python
# src/geocatalog/_src/matchup/__init__.py

def matchup(
    catalog: GeoCatalog,
    *,
    primary: Selector,                 # filter dict, e.g. {"source": "earthaccess", "collection": "MOD09GA"}
    secondary: Selector | list[Selector],
    spatial: SpatialStrategy,           # IntersectsAtLeast(iou=0.2), CentroidWithin(buffer="5km")
    temporal: TemporalStrategy,         # NearestInTime(dt="6h"), WithinWindow(start=, end=)
    join: Literal["all", "any"] = "all",
    tag: str | None = None,
) -> Iterator[MatchupRow]: ...
```

**Spatial strategies** (`spatial.py`):
- `Intersects()` — non-zero intersection
- `IouAtLeast(t: float)` — IoU ≥ t
- `CentroidWithin(buffer: str | float)` — secondary centroid within buffer of primary
- `Contains()` — secondary fully contained in primary footprint

**Temporal strategies** (`temporal.py`):
- `NearestInTime(dt: str)` — pick secondary nearest in time, only if Δt ≤ dt
- `WithinWindow(start: timedelta, end: timedelta)` — all secondaries in [t+start, t+end] relative to primary
- `Synchronous(tolerance: str = "0s")` — overlapping observation intervals

Implementation: emit DuckDB SQL where possible (range joins on time, spatial via DuckDB's `spatial` extension), fall back to shapely + pandas merge_asof for the non-SQL bits. Matchup output is itself a `GeoCatalog` of `MatchupRow`s and supports the same `query(bounds, interval)` calls — patchers downstream don't need to distinguish "items" from "matchups".

### 4.7 Staging layer

Explicit, never automatic:

```python
# src/geocatalog/_src/staging/__init__.py

def stage(
    catalog: GeoCatalog,
    *,
    dest: PathLike,
    assets: list[str] | None = None,         # asset keys; None = all
    parallel: int = 8,
    cache: LocalCache | None = None,
    retries: int = 3,
) -> GeoCatalog: ...
```

- Returns a new catalog whose `assets` columns have been rewritten to local paths (a `_staged_path` field per asset preserves the original URI).
- `LocalCache` is fsspec-backed; default location `~/.cache/geocatalog/` or `$GEOCATALOG_CACHE`.
- GEE-specific path: `staging/gee.py` materializes via `ee.Image.getDownloadURL` or `ee_export_image` — does not bypass EE compute.
- Reuses the retry/backoff machinery from PR #51 (currently scoped to raster loaders).

The staged catalog is a normal `GeoCatalog`, so the existing `load_raster` / `load_vector` / `xarray_backend` loaders see it as local files and read them in-place.

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
| `SwathToGrid(method="bowtie_aware", target_crs=…, target_res=…)` | `swath → grid` | rasterio + per-pixel lat/lon | Track-B gap; handles MODIS/VIIRS bowtie |
| `GridToSwath(time_match="nearest", dt_max="15min")` | `(grid_series, swath_like) → grid_at_swath_geom` | rasterio + temporal index | GEO → LEO acquisition geometry |
| `RasterToPoints(extract="nearest" \| "bilinear")` | `(raster, points) → xvec_cube` | xvec | Extract raster at point geometries → vector cube |
| `PointsToRaster(method="binned_stat", stat="mean", like=…)` | `(points, like) → raster` | scipy.stats.binned_statistic_2d | Bin point cube into grid |
| `RasterToPointCloud(k=…, max_radius=…)` | `(raster, cloud) → cloud_with_attrs` | scipy.spatial.KDTree | Sample raster onto cloud nodes |
| `PointCloudToRaster(method="idw" \| "binned_stat", like=…)` | `(cloud, like) → raster` | scipy KDTree + IDW | Rasterize point cloud |
| `VectorToRasterAgg(agg="mean" \| "majority" \| "count", like=…)` | `(vector, like) → raster` | extends `Rasterize` | Aggregation policy for overlapping features |
| `StackMatched(order=…, fill=NaN)` | `[t1, t2, …] → multi_band` | numpy stack + reproject-to-like | Compositing-style: N aligned tensors → 1 multi-band GeoTensor |
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

### 6.1 New file layout

```
src/geopatcher/_src/
  matched/                                  # NEW
    __init__.py
    field.py        # MatchedField composite Field
    patch.py        # MatchedPatch carrier
    patcher.py      # MatchedSpatialPatcher (+ Temporal / SpatioTemporal variants)
    aggregation.py  # MatchedAggregator (per-source dict of aggregators)
  spatial/, time/, …                         # unchanged
```

Public re-export at `geopatcher.matched`.

### 6.2 `MatchedField` — a composite Field

```python
# geopatcher/_src/matched/field.py

@dataclass(eq=False)
class MatchedField:
    """N co-registered Fields presented as one Field.

    Satisfies the `Field` Protocol via the primary (anchor space, CRS, domain).
    On select(), delegates to each secondary and pipes through its coreg callable.
    """
    primary: Field
    secondaries: Mapping[str, Field]
    coreg: Mapping[str, Callable]   # any Callable; pipekit.Operator (e.g. from
                                    # geotoolz.geom.coregister) is the recommended choice
                                    # — see ADR-003 for why the type is the broader Callable.
    valid_mask: bool = True         # emit per-source nodata masks

    @property
    def domain(self) -> Domain:
        return self.primary.domain

    def select(self, indexer: Any) -> MatchedPatch:
        # `Field.select` takes a single `indexer` (the shape is decided by
        # the primary's Domain — Window for raster, dict[str, slice] for
        # grid, etc.). MatchedField forwards the same indexer to every
        # member.
        primary_patch = self.primary.select(indexer)
        members: dict[str, Patch] = {"primary": primary_patch}
        masks: dict[str, np.ndarray] = {}
        for name, sec in self.secondaries.items():
            raw = sec.select(indexer)
            aligned = self.coreg[name](raw, primary_patch)  # any geotoolz op
            members[name] = aligned
            if self.valid_mask:
                masks[name] = _compute_mask(aligned)
        return MatchedPatch(anchor=primary_patch.anchor, members=members, valid_mask=masks)
```

Three properties this design preserves:

1. **Existing samplers, geometries, windows, and aggregations work unchanged.** `MatchedField` *is* a `Field`; `SpatialPatcher(MatchedField(...))` is valid.
2. **Geopatcher's core stays numpy + scipy.** The `coreg` dict holds opaque callables; geopatcher never imports `geotoolz`.
3. **Coregistration logic is reusable outside patching.** Same operators serve flat pipelines, validation scripts, and matchup builds.

### 6.3 `MatchedPatch` — the carrier

```python
# geopatcher/_src/matched/patch.py

@dataclass
class MatchedPatch:
    anchor: Anchor
    members: dict[str, Patch]                    # "primary" + secondaries by name
    valid_mask: dict[str, np.ndarray] | None     # NaN / out-of-swath per source
    weights: dict[str, np.ndarray] | None = None # per-source window weights, optional
```

`MatchedPatch` does not subclass `Patch` — it's a sibling carrier. Operators that want to consume one explicitly type against `MatchedPatch`; legacy operators see a single primary `Patch` via `mp.members["primary"]`.

### 6.4 Aggregation back to global field(s)

Merge is per-source. `MatchedSpatialPatcher.merge(patches)` returns a `dict[str, Field]`:

```python
class MatchedSpatialPatcher:
    primary: SpatialPatcher
    secondary_aggregators: dict[str, Aggregation]  # per-source aggregator

    def merge(self, patches: Iterable[MatchedPatch]) -> dict[str, Field]:
        ...
```

This is the only API shape change relative to existing patchers, and it's introduced by a new class so backwards compat is untouched.

### 6.5 Streaming guarantees

`MatchedField.iter_patches` yields one `MatchedPatch` at a time. Memory is bounded by `patch_size × len(secondaries)`. All existing streaming aggregators (e.g. `SpatialOverlapAdd` with Zarr backing) work per-source.

## 7. End-to-end walkthrough: MODIS × Sentinel-2 patches over Iberia

```python
import geocatalog as gc
from geocatalog.sources import EarthAccessSource, STACSource
from geocatalog.matchup import IouAtLeast, NearestInTime
import geopatcher as gp
from geopatcher.matched import MatchedField, MatchedSpatialPatcher
from geotoolz.geom.coregister import RasterToRasterLike

# 1. Discover & ingest
cat = gc.DuckDBGeoCatalog.open("my_catalog/", target_crs="EPSG:32629", create=True)

cat.ingest(
    EarthAccessSource(),
    collection="MOD09GA",
    bounds=(-10, 35, 5, 45), interval=("2024-06-01", "2024-06-30"),
    tag="iberia_summer24",
)

cat.ingest(
    STACSource.planetary_computer(),
    collection="sentinel-2-l2a",
    bounds=(-10, 35, 5, 45), interval=("2024-06-01", "2024-06-30"),
    filters={"eo:cloud_cover": {"lt": 20}},
    tag="iberia_summer24",
)

# 2. Build matchups
matchup_id = cat.matchup(
    primary={"source": "earthaccess", "collection": "MOD09GA"},
    secondary={"source": "stac.pc", "collection": "sentinel-2-l2a"},
    spatial=IouAtLeast(0.2),
    temporal=NearestInTime(dt="6h"),
    tag="modis_s2_pairs_v1",
)

# 3. Stage bytes for the matched assets only
staged = cat.stage(matchup_tag="modis_s2_pairs_v1", dest="./staged/",
                   assets=["red", "nir", "scl"], parallel=8)

# 4. Build a MatchedField from a matchup row
modis_field = staged.field_for(matchup_id, role="primary")     # RasterField
s2_field    = staged.field_for(matchup_id, role="secondary")   # RasterField

matched = MatchedField(
    primary=modis_field,
    secondaries={"s2": s2_field},
    coreg={"s2": RasterToRasterLike(resampling="bilinear")},
)

# 5. Patch with any existing geopatcher sampler
patcher = MatchedSpatialPatcher(
    primary=gp.SpatialPatcher(
        geometry=gp.spatial.SpatialRectangular(size=(512, 512)),
        sampler=gp.spatial.SpatialRegularStride(stride=(256, 256)),
        window=gp.spatial.SpatialBoxcar(),
        aggregation=gp.spatial.SpatialMean(),
    ),
    secondary_aggregators={"s2": gp.spatial.SpatialMean()},
)

for matched_patch in patcher.split(matched):
    modis_chip = matched_patch.members["primary"].data     # (bands, H, W) MODIS
    s2_chip    = matched_patch.members["s2"].data          # (bands, H, W) S2 at MODIS grid
    mask       = matched_patch.valid_mask["s2"]            # where S2 is valid
    # → into model / training loop / further geotoolz pipeline
```

Three things to notice:

- The user never touches a coregistration class — they pick a `geotoolz` operator.
- The matchup is reproducible: re-running step 2 with the same tag against an updated catalog produces a new `matchup_id` but the same matchup logic, persisted.
- Every step is independently usable: ingest without matchup, matchup without staging, stage without patching, patch without matchup (just `MatchedField(primary, {}, {})`).

## 8. Phasing

A suggested four-phase rollout. Each phase ships independently and is useful on its own.

### Phase 1 — `geocatalog` source adapters (no breaking changes)

- `_src/sources/__init__.py` with `Source` Protocol, `CatalogRow`
- `EarthAccessSource`, `STACSource`, `CMRSource`
- `catalog.ingest()` + `geocatalog ingest` CLI
- `queries.parquet` schema + migration v0 → v1
- GEE deferred to Phase 3

**Exit criteria:** `geocatalog ingest earthaccess MOD09GA …` works end-to-end; round-trip test ingest → query → load_raster passes.

### Phase 2 — `geocatalog` matchup engine

- `_src/matchup/` with spatial + temporal strategy classes
- `matchups.parquet` schema
- `catalog.matchup()` + `geocatalog matchup` CLI
- DuckDB-backed implementation for performance

**Exit criteria:** MODIS × S2 matchup over Iberia completes in < 30s for ~1k granules each.

### Phase 3 — `geotoolz` coregistration operators

- `geotoolz.geom.coregister` submodule, eight new operators
- `geotoolz.compositing.StackMatched` / `BlendMatched`
- xvec optional extra
- YAML round-trip tests for all new operators
- (Parallel track) `GEESource` in geocatalog

**Exit criteria:** `RasterToRasterLike` parity-tests against existing `Reproject + Resample`; `SwathToGrid` produces a regular-grid MOD09GA tile from raw swath input.

### Phase 4 — `geopatcher` matched field

- `geopatcher.matched` submodule with `MatchedField` / `MatchedPatch` / `MatchedSpatialPatcher`
- Streaming determinism tests (extends existing Hypothesis suite)
- Notebook recipe: MODIS × S2 matched patches → torch DataLoader
- `geocatalog.stage().field_for()` helper that returns ready-to-go Fields

**Exit criteria:** The end-to-end walkthrough in §7 runs as a notebook.

### Phase 5 (optional / later) — staging polish

- `geocatalog.staging` with cache, parallelism, retry
- `obstore` / `fsspec` adapter for multi-cloud (lines up with existing branch `copilot/feat-io-fsspec-obstore-path-resolution`)

## 9. Open questions

| # | Question | Default if undecided |
|---|---|---|
| 1 | Matchup catalog as sibling Parquet vs. separate artifact? | **Sibling** (§4.4) |
| 2 | GEE scope: enumerate only, or run `ee.Image` recipes in staging? | **Enumerate only in v1**; recipes deferred |
| 3 | STAC adapter: one generic + named factory helpers, or distinct subclasses per provider? | **Generic `STACSource(endpoint=...)` with class-method factories** |
| 4 | Auth surface: defer to libraries, or add `geocatalog auth status` aggregator? | **Defer**; add aggregator if users ask |
| 5 | `MatchedPatch` subclasses `Patch` or sibling carrier? | **Sibling** (avoid LSP issues) |
| 6 | `MatchedField.coreg` typed as `dict[str, Operator]` (pipekit) or untyped `Callable`? | **Resolved (ADR-003):** typed as `Mapping[str, Callable]`; `pipekit.Operator` is the recommended value but not required, so geopatcher's core stays framework-free. |
| 7 | Should the matchup engine emit a *new catalog* or in-place new rows in `items.parquet`? | **New rows in `matchups.parquet`**; `items` stays atoms |
| 8 | Cache scope for staging: per-catalog, per-user, or per-host? | **Per-user** (`~/.cache/geocatalog/`), overridable by env var |
| 9 | Do we want `BlendMatched(method="ivw")` in Phase 3 or defer (uncertainty maps not always present)? | **Defer**; ship `StackMatched` first |
| 10 | Where does this design doc finally live? | TBD — currently `~/query-matchup-design.md`; candidates: `geocatalog/docs/design/`, a new cross-package `docs/` repo, or split into per-package ADRs |

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
    .CatalogRow                     # normalized output schema
    .EarthAccessSource              # adapter
    .STACSource                     # adapter (+ .planetary_computer(), .earth_search())
    .GEESource                      # adapter (Phase 3)
    .CMRSource                      # adapter
  .matchup
    .matchup(...)                   # functional
    .MatchupRow                     # dataclass
    .IouAtLeast, .CentroidWithin, .Intersects, .Contains   # spatial strategies
    .NearestInTime, .WithinWindow, .Synchronous            # temporal strategies
  .staging
    .stage(...)                     # functional
    .LocalCache                     # cache class
  .GeoCatalog                       # (existing) extended with .ingest, .matchup, .stage
  .GeoSlice                         # (existing)
  .DuckDBGeoCatalog, .InMemoryGeoCatalog  # (existing)

geotoolz.geom.coregister
  .RasterToRasterLike
  .SwathToGrid
  .GridToSwath
  .RasterToPoints                   # requires [vector-cube] extra (xvec)
  .PointsToRaster                   # requires [vector-cube] extra
  .RasterToPointCloud
  .PointCloudToRaster
  .VectorToRasterAgg

geotoolz.compositing
  .StackMatched                     # new
  .BlendMatched                     # new (Phase 3+, deferred if uncertainty rare)

geopatcher.matched
  .MatchedField
  .MatchedPatch
  .MatchedSpatialPatcher
  .MatchedTemporalPatcher           # mirror, Phase 4+
  .MatchedSpatioTemporalPatcher     # mirror, Phase 4+
```
