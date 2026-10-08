# The geostack

This repository ships **four packages that are designed as one stack**:
read products (`geoproducts`), find data (`geocatalog`), cut it into
model-sized pieces (`geopatcher`), and compute on it (`geotoolz`) — with the operator-graph composition core
supplied by the external [pipekit](https://github.com/jejjohnson/pipekit)
framework. Each package works standalone, but the seams between them are
first-class API.

```mermaid
flowchart LR
    subgraph catalog["geotoolz-catalog · import geocatalog"]
        SRC[STAC / CMR /<br/>earthaccess sources] --> CAT[(GeoCatalog<br/>GeoParquet)]
        CAT -->|query bbox + time| GS[GeoSlice]
        GS --> LD[load_raster /<br/>load_xarray / load_vector]
        CAT --> STG[staging.stage]
    end
    subgraph patcher["geotoolz-patcher · import geopatcher"]
        F[Field] --> SP[SpatialPatcher<br/>Geometry × Sampler ×<br/>Window × Aggregation]
        SP -->|split| P[patches]
        P2[operated patches] -->|merge| ST[stitched field]
    end
    subgraph toolz["geotoolz · import geotoolz"]
        OPS[Operator families<br/>radiometry · indices · qa ·<br/>geom · learn · einx · …]
    end
    STG -->|field_for| F
    P -->|ApplyToChips| OPS
    OPS --> P2
```

| Package (dist) | Import | Role |
|---|---|---|
| `geotoolz-catalog` | `geocatalog` | *Which files cover my bbox + time window?* Queryable spatiotemporal index with GeoParquet interchange, in-memory + DuckDB backends, discovery sources, matchup, staging. |
| `geotoolz-patcher` | `geopatcher` | *How do I turn a huge field into model-sized patches and back?* Four-axis Patcher (Geometry × Sampler × Window × Aggregation) over a substrate-agnostic `Field` protocol. |
| `geotoolz` | `geotoolz` | *What do I compute on each piece?* Carrier-preserving `pipekit.Operator` families for remote sensing, from radiometry to spectral indices to matched filters. |
| `geotoolz-products` | `geoproducts` | *How do I open this mission's or provider's product?* Readers that turn EO data products into georeader `GeoData` / `GeoTensor`s; independent of the operator library. |

## The seams

These are the deliberate integration points — each one is a small,
documented contract rather than an import tangle:

**1. `GeoSlice` — the catalog ↔ loader ↔ patcher wire format.**
A frozen `(bounds, interval, resolution, crs)` request for data.
Catalogs produce them, loaders consume them, and
[exact grid alignment](catalog/design/exact-grid-alignment.md)
(`align=`, `count_steps`, `is_grid_aligned`) keeps slice shapes honest
against co-registered products.

**2. `staging.field_for` — catalog rows become patcher Fields.**
`geocatalog.staging.stage()` resolves remote URIs into a local cache, and
`field_for()` mosaics the staged rows onto a `GeoSlice` grid with
`load_raster` — in the slice CRS, across files in any CRS — and wraps the
result as one `geopatcher` `RasterField`, so a catalog query drops straight
into `SpatialPatcher.split` / `merge` (enabled by the
`geotoolz-catalog[patch]` extra).

**3. `patch_ops` — the patcher joins the operator graph.**
`geotoolz.patch_ops` (same classes as `geopatcher.integrations.pipekit`)
wraps a `SpatialPatcher` as pipeline stages: `GridSampler → ApplyToChips →
MergePatches` composes tile-predict-stitch inference inside a `Sequential`, and
the label-aware `StratifiedSample` / `BalancedSampler` emit the same
`Patch` carrier for training-time draws. See the
[patching module guide](patch_ops.md).

**4. Coregistration operators plug into matched patching.**
`geopatcher.matched.MatchedField` fans one anchor out across N sources and takes a
*coregistration callable* per secondary — the intended callables are
`geotoolz.geom.coregister` operators (`RasterToRasterLike`,
`RasterToPoints`, …). The catalog's [matchup engine](catalog/design/query-matchup.md)
finds the row pairs; the patcher reads them; the operators align them.

**5. One obstore pool per process.**
`geocloud.store` owns the pooled `obstore` client and the `[obstore]`
extras of geotoolz-products and geocatalog depend on it, so a
pipeline touching the same bucket through the catalog's staging, the
patcher's `CogField`, and geoproducts' product readers reuses one
HTTP/2 connection pool.

## End to end in one screen

```python
import geocatalog as gc
import geopatcher as gp
import geotoolz as gz
from geotoolz.patch_ops import ApplyToChips, GridSampler, MergePatches

# 1. Discover + index (catalog)
cat = gc.sources.from_stac_search(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    collections=["sentinel-2-l2a"], bounds=aoi_bbox, datetime="2024-06",
    asset_key="B04",                     # one row per item, filepath = its B04 href
)

# 2. Stage + bridge to Fields (catalog → patcher seam)
staged = gc.staging.stage(cat, dest="./cache")
field = gc.patch.field_for(staged, aoi_slice)  # aoi_slice: GeoSlice (bounds, CRS, resolution)

# 3. Patch + operate + stitch (patcher → operators seam)
patcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(256, 256)),
    sampler=gp.spatial.sampler.RegularStride(step=(192, 192)),
    window=gp.spatial.window.Hann(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
ndvi = gz.Sequential([
    gz.DNToReflectance(scale=1e-4),        # geotoolz radiometry
    gz.NDVI(nir=3, red=2),         # geotoolz indices
])
pipe = gz.Sequential([
    GridSampler(patcher=patcher),
    ApplyToChips(operator=ndvi),
    MergePatches(aggregation=gp.spatial.aggregation.OverlapAdd(), domain=field.domain),
])
ndvi_scene = pipe(field)        # GeoTensor (H, W) on the AOI grid · NaN = no data
```

The [catalog → patch → operate recipe](recipes/integration-with-geocatalog-and-geopatcher.md)
walks through this pipeline step by step, and each package's section of
this site covers its own axis in depth.

## Install

Everything ships from this repo as three distributions:

```bash
pip install geotoolz                      # operators only
pip install 'geotoolz[patch]'             # + geotoolz-patcher (geopatcher)
pip install geotoolz-catalog              # catalog only
pip install 'geotoolz-catalog[patch]'     # catalog + patcher bridge
```

Pre-PyPI, install from a clone (`uv sync --all-packages`) or via git URLs
with `subdirectory=packages/<name>`.
