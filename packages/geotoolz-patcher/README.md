# geopatcher
> Part of the [geotoolz monorepo](https://github.com/jejjohnson/geotoolz) — ships as the `geotoolz-patcher` distribution; the import name is unchanged.

[![Tests](https://github.com/jejjohnson/geotoolz/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/geotoolz/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/geotoolz/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/geotoolz/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/geotoolz/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/geotoolz)
[![PyPI version](https://img.shields.io/pypi/v/geotoolz-patcher.svg)](https://pypi.org/project/geotoolz-patcher/)

> **Split a geospatial field into local patches, run an operator per patch, and stitch the outputs back into a global result — along four independently composable axes.**

`geopatcher` is the *locality layer* between **catalogs**
([geocatalog](https://github.com/jejjohnson/geotoolz/tree/main/packages/geotoolz-catalog)),
**readers** ([geoproducts](https://github.com/jejjohnson/geotoolz/tree/main/packages/geotoolz-products))
and **operators** ([geotoolz](https://github.com/jejjohnson/geotoolz)). It answers a single
question: *what slice of the data does my operator see at once, and how do
local outputs become a global field?*

<p align="center"><img src="../../docs/assets/diagrams/patcher-axes.png" alt="SpatialPatcher: a Field is split along Geometry, Sampler, Window and Aggregation axes, an operator runs per patch, and the aggregation stitches the result" width="100%"></p>

## 30-second elevator pitch

Sliding-window inference, tile-based training, hierarchical patching,
COG/zarr streaming — all the same four-axis composition. Pick a
**Geometry** (shape of the neighborhood), a **Sampler** (where anchors
go), a **Window** (boundary treatment), and an **Aggregation** (local →
global merge). Plug in any `Field` (raster, xarray grid, GeoPandas
polygons, xvec points) and any per-patch callable. `patcher.split`
returns an iterator and `spatial.aggregation.OverlapAdd` defaults to an in-memory
accumulator; flip `streaming=True` + `target_path=…` + `chunks=…` to
back the accumulator with disk-resident zarr for >1 TB outputs.

## Install

```bash
pip install geotoolz-patcher                       # base — RasterField only
pip install 'geotoolz-patcher[grid]'               # XarrayField
pip install 'geotoolz-patcher[vector]'             # GeoPandasField
pip install 'geotoolz-patcher[point]'              # XvecField
pip install 'geotoolz-patcher[xarray-raster]'      # RioXarrayField
pip install 'geotoolz-patcher[streaming]'          # disk-backed OverlapAdd
pip install 'geotoolz-patcher[dask]'               # DaskField, geopatcher.run Dask bridge
pip install 'geotoolz-patcher[jax]'                # geopatcher.run batched splitting
pip install 'geotoolz-patcher[cog]'                # CogField (geotoolz-cloud COG engine)
pip install 'geotoolz-patcher[patch-full]'         # all of the above
pip install 'geotoolz-patcher[pipekit]'            # pipekit operator-graph bridge
```

> `pipekit` isn't on PyPI yet; the `[pipekit]` extra resolves via `uv sync
> --extra pipekit` against the GitHub source. Plain `pip install` of that
> extra will start working once `pipekit` ships to PyPI.

## Quickstart

```python
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp

# 1. Wrap any GeoTensor (or GeoData reader) as a Field.
arr: np.ndarray = np.outer(np.linspace(0, 1, 512), np.linspace(0, 1, 512)).astype(np.float32)[None]  # (1, 512, 512)
field: gp.RasterField = gp.RasterField(
    GeoTensor(arr, transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000), crs="EPSG:32611")
)

# 2. Compose the four axes.
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry    = gp.spatial.geometry.Rectangular(size=(128, 128)),
    sampler     = gp.spatial.sampler.RegularStride(step=(96, 96)),    # 32 px overlap
    window      = gp.spatial.window.Hann(),                          # feather the seams
    aggregation = gp.spatial.aggregation.OverlapAdd(),
)

# 3. Split → operate per patch → merge.
patches: list[gp.Patch] = list(patcher.split(field))         # 25 patches, data (1, 128, 128) each
out: list[gp.Patch] = [
    p.with_data(np.asarray(p.data) * 2.0)                   # your operator here
    for p in patches
]
stitched: np.ndarray = patcher.merge(out, field.domain)     # (1, 512, 512) float64
```

`patch.anchor` is the patch origin in pixel coordinates (`(0, 0)`,
`(0, 96)`, …) and `field.domain` carries the grid the merge writes onto.
For independent local jobs, swap the loop for
`geopatcher.run.parallel_map`; for global-context operators, use the codified
`reduce` / `two_pass` helpers. See the
[concepts page](https://jejjohnson.github.io/geotoolz/patcher/concepts/) for the
full mental model.

## Where things live

The root holds what every job touches; everything else has one home,
named for the task:

| Namespace | What's there |
|---|---|
| `geopatcher` | `SpatialPatcher`, `AsyncSpatialPatcher`, `TemporalPatcher`, `SpatioTemporalPatcher`; `Patch` carriers; `Field` / `Domain`; `RasterField` |
| `geopatcher.spatial` | the four axes — `geometry` (`Rectangular`, …), `sampler` (`RegularStride`, …), `window` (`Hann`, …), `aggregation` (`OverlapAdd`, …) |
| `geopatcher.temporal` | the same four axes along time, plus `stencils` (`TimeStencil`, …) |
| `geopatcher.fields` | `XarrayField`, `RioXarrayField`, `DaskField`, `GeoPandasField`, `XvecField`, `CogField`, … and the domain types |
| `geopatcher.matched` | patching co-registered sources together (`MatchedField`, `Matched*Patcher`) |
| `geopatcher.run` | `parallel_map`, `prefetch_iterable`, Dask (`to_delayed`) and JAX (`batch_split`) bridges, `PatchCache`, `IndexedPatchView` |
| `geopatcher.observe` | `PatcherHook`, `PatchJournal`, `PatchErrorRecord`, strict mode |
| `geopatcher.config` | `axis_envelope` / `from_config` round-trips |

Cloud-Optimized GeoTIFF reads and the shared object-store pool live in
[geotoolz-cloud](https://github.com/jejjohnson/geotoolz/tree/main/packages/geotoolz-cloud)
(`geocloud`); `geopatcher.fields.CogField` is their `Field`.

## With geotoolz operators

The `[pipekit]` bridge (`geopatcher.integrations.pipekit`, re-exported as
`geotoolz.patch_ops`) makes *split → operate → merge* one more operator
pipeline:

```python
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp
import geotoolz as gz
from geotoolz.patch_ops import ApplyToChips, GridSampler, MergePatches

rng: np.random.Generator = np.random.default_rng(0)
scene: GeoTensor = GeoTensor(
    rng.integers(1, 10_000, size=(4, 512, 512), dtype=np.uint16),  # (4, 512, 512) uint16
    transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
    crs="EPSG:32611",
    fill_value_default=0,
    attrs={"band_names": ["blue", "green", "red", "nir"]},
)
field: gp.RasterField = gp.RasterField(scene)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(128, 128)),
    sampler=gp.spatial.sampler.RegularStride(step=(96, 96)),
    window=gp.spatial.window.Hann(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)

ndvi: gz.Sequential = gz.DNToReflectance(scale=1e-4) | gz.NDVI(nir="nir", red="red")
tiled: gz.Sequential = gz.Sequential([
    GridSampler(patcher=patcher),          # field → 25 Patches     (4, 128, 128) uint16
    ApplyToChips(operator=ndvi),           # (4, 128, 128) uint16 → (128, 128) float64
    MergePatches(aggregation=gp.spatial.aggregation.OverlapAdd(), domain=field.domain),
])
ndvi_map: GeoTensor = tiled(field)         # (512, 512) float64 · NaN = no data
```

The domain fixes the grid and the patches fix the bands, so NDVI's
one-band chips merge into one `(H, W)` map on the scene's transform and
CRS.

## Next steps

- **Concepts:** [concepts](https://jejjohnson.github.io/geotoolz/patcher/concepts/) — the four-axis abstraction with diagrams.
- **15-min walkthrough:** [quickstart](https://jejjohnson.github.io/geotoolz/patcher/quickstart/) — Lake Tahoe Sentinel-2 NDVI inference.
- **Recipes:** [streaming OverlapAdd](https://jejjohnson.github.io/geotoolz/patcher/recipes/streaming-overlap-add/), [on-error policies](https://jejjohnson.github.io/geotoolz/patcher/recipes/on-error-policies/), [PatchJournal resume](https://jejjohnson.github.io/geotoolz/patcher/recipes/journal-and-resume/).
- **Demo notebook:** [patcher_lake_tahoe](https://jejjohnson.github.io/geotoolz/patcher/notebooks/patcher_lake_tahoe/) — patcher slice of the Lake Tahoe scenario.
- **See the full end-to-end story:** [`docs/catalog/notebooks/end_to_end_lake_tahoe.ipynb`](https://github.com/jejjohnson/geotoolz/blob/main/docs/catalog/notebooks/end_to_end_lake_tahoe.ipynb) — catalog → operators → patcher.
- **API reference:** [docs site](https://jejjohnson.github.io/geotoolz/patcher/api/reference/).

## License

MIT — see [LICENSE](LICENSE).
