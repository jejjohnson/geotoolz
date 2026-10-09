# The geostack

> **Find the data, read it, cut it to size, compute on it — one composable stack.**
> Five Python packages that interlock end to end, with the Operator /
> Sequential / Graph composition core supplied by
> [pipekit](https://github.com/jejjohnson/pipekit).

![The geostack stage by stage: search an archive, catalog it, save what you need, stream windows lazily, split into overlapping patches, apply an operator per patch, combine with a window-weighted merge, write a COG or zarr](assets/diagrams/geostack-intro.png)

## Where it comes from

If you have used xarray, you know **split → apply → combine**: group the
data, run a function on each group, glue the results back together. A
geospatial pipeline is the same idea with more steps on either side.
Before you can split a scene you have to **search** an archive for it,
**catalog** what you found so the next question is a query rather than a
crawl, **save** (stage) only the files that matter, and **stream** them as
lazy windows because the scene does not fit in memory. After you combine,
you **write** a georeferenced result.

Most codebases rebuild that loop per project, so every step is hard-wired
to the next. The geostack gives each step its own package and one small,
typed seam between neighbours: a `GeoSlice` is the request every catalog
answers, a georeader `GeoTensor` is the array every reader returns and
every operator takes, and a `Patch` is what the patcher hands an operator.
Because operators are plain `pipekit` objects, the patcher itself becomes
an operator — *tile → predict → stitch* is one more `Sequential`.

## The packages

| Package | Import | Use it to… | Docs |
|---|---|---|---|
| `geotoolz` | `geotoolz` | compute on rasters with carrier-preserving operators (radiometry, indices, masks, geometry, ML, …) | [Operators](operators/index.md) |
| `geotoolz-patcher` | `geopatcher` | split a field into patches, run an operator per patch, stitch back (Geometry × Sampler × Window × Aggregation) | [Patcher](patcher/index.md) |
| `geotoolz-catalog` | `geocatalog` | find, index, join, load and stage files (`GeoSlice` queries over GeoParquet) | [Catalog](catalog/index.md) |
| `geotoolz-cloud` | `geocloud` | reach object storage: one client pool, file verbs, credentials, COG reads and writes | [Cloud](cloud/index.md) |
| `geotoolz-products` | `geoproducts` | read Earth-observation products (GOES-R ABI, Himawari AHI, Carbon Mapper, …) as `GeoTensor`s | [Products](products/index.md) |

![Dependency layers: the five packages stand on georeader; object storage in geotoolz-cloud; geotoolz on pipekit; cross-package links are opt-in extras](assets/diagrams/stack-layers.png)

Solid arrows are hard dependencies, dotted ones opt-in extras, and nothing
points back up: install only the packages you need. [How the packages
interlock](geostack.md) walks through every seam.

## Install

```bash
pip install geotoolz                          # operators
pip install geotoolz-patcher                  # patcher
pip install geotoolz-catalog                  # catalog
pip install geotoolz-cloud                    # object storage, COG writes
pip install geotoolz-products                 # product readers
```

| To connect… | Install |
|---|---|
| operators ↔ patcher (`geotoolz.patch_ops`) | `geotoolz[patch]` |
| catalog → patcher (`geocatalog.patch.field_for`) | `geotoolz-catalog[patch]` |
| catalog staging from buckets (`geocatalog.staging.stage`) | `geotoolz-catalog[cloud]` |
| catalog ← STAC searches (`geocatalog.sources.from_stac_search`) | `geotoolz-catalog[stac]` |
| patcher ↔ COGs in buckets (`geopatcher.fields.CogField`) | `geotoolz-patcher[cog]` |
| COG reads (`geocloud.cog.CogSource`) | `geotoolz-cloud[cog]` |
| a sensor's reader and its bucket helpers | `geotoolz-products[goes]` / `[himawari]` / `[carbonmapper]` |
| a sensor's presets as geotoolz operators | `geotoolz-products[operators]` |

Each package page lists its own extras. Pre-PyPI, install from a clone:

```bash
git clone https://github.com/jejjohnson/geotoolz && cd geotoolz
uv sync --all-packages --all-groups --all-extras
```

## Quickstart — catalog → patcher → operators

Summer-2024 NDVI over Lake Tahoe: discover Sentinel-2 L2A on the Planetary
Computer, mosaic the red and near-infrared bands onto one grid, and run
NDVI tile by tile with feathered seams. Every binding is typed and every
array is annotated with its shape. It needs
`pip install 'geotoolz[patch]' 'geotoolz-catalog[stac,cloud,patch]'`.

```python
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
from georeader.geotensor import GeoTensor

import geocatalog as gc
import geopatcher as gp
import geotoolz as gz
from geotoolz.patch_ops import ApplyToChips, GridSampler, MergePatches

client: pystac_client.Client = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
aoi: gc.GeoSlice = gc.GeoSlice(
    bounds=(-120.25, 38.85, -119.85, 39.30),              # Lake Tahoe, lon/lat
    interval=pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-09-30"), closed="both"),
    resolution=(0.0001, 0.0001),                          # ≈ 10 m → grid (H, W) = (4500, 4000)
    crs="EPSG:4326",
)

# 1 · find — one catalog per band, staged to local disk
bands: dict[str, gc.GeoCatalog] = {
    band: gc.staging.stage(
        gc.sources.from_stac_search(client, collections=["sentinel-2-l2a"], bounds=aoi.bounds,
                            datetime="2024-06-01/2024-09-30", asset_key=band),
        dest="./cache",
    )
    for band in ("B04", "B08")
}

# 2 · read — mosaic each band onto the AOI grid, stack red + NIR
red: gp.RasterField = gc.patch.field_for(bands["B04"].query(aoi), aoi)   # (1, 4500, 4000) uint16
nir: gp.RasterField = gc.patch.field_for(bands["B08"].query(aoi), aoi)   # (1, 4500, 4000) uint16
scene: GeoTensor = gz.StackBands()([red.reader, nir.reader])         # (2, 4500, 4000) uint16

# 3 · cut + compute — 256² tiles, 64 px overlap, Hann-feathered seams
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(256, 256), boundary="pad"),
    sampler=gp.spatial.sampler.RegularStride(step=(192, 192)),
    window=gp.spatial.window.Hann(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
ndvi: gz.Sequential = gz.DNToReflectance(scale=1e-4) | gz.NDVI(red=0, nir=1)  # (2, h, w) → (h, w)
field: gp.RasterField = gp.RasterField(scene)
tiled: gz.Sequential = gz.Sequential([
    GridSampler(patcher=patcher),                        # field → list[Patch]       (2, 256, 256) each
    ApplyToChips(operator=ndvi),                         # list[Patch] → list[Patch] (256, 256) each
    MergePatches(aggregation=gp.spatial.aggregation.OverlapAdd(), domain=field.domain),
])
result: GeoTensor = tiled(field)                        # (4500, 4000) float64 · NaN = no data
```

`MergePatches` places the chips on the domain's grid and keeps the band
axes the chips carry, so NDVI's `(h, w)` chips merge into one `(H, W)`
`GeoTensor` on the scene's transform and CRS.

## Next steps

- **[How the packages interlock](geostack.md)** — the seams, and an
  advanced example that uses all of them.
- **Tutorials** on real Sentinel-2 data over Lake Tahoe:
  [catalog](catalog/notebooks/end_to_end_lake_tahoe.ipynb) ·
  [patcher](patcher/notebooks/patcher_lake_tahoe.ipynb) ·
  [operators](operators/notebooks/operators_lake_tahoe.ipynb).
- **[Capability index](capabilities.md)** — every public name in the stack,
  with a one-line summary. Search it before writing a helper.
- **[Building with agents](agents.md)** — the Claude Code plugin and
  `llms.txt` for building on the stack with an AI agent.
