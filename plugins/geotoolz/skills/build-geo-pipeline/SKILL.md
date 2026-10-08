---
name: build-geo-pipeline
description: Build a geospatial / Earth-observation pipeline on the geotoolz stack — search an archive, index and query files, read satellite products, tile a large scene, run per-pixel operators (indices, masks, radiometry, ML), stitch and write — composing geocatalog, geoproducts, geopatcher, geotoolz and geocloud instead of writing that code. Use whenever a task involves rasters, GeoTIFF / COG / NetCDF / Zarr, satellite imagery, STAC, tiling or patch-wise inference in a project that uses (or could use) these packages.
---

# Build on the geotoolz stack

The stack already does the plumbing; your code should be the science. Before
writing any helper — a reader, a retry loop, a tiler, a band-math function,
an object-store client — look it up:

1. **The capability index** lists every public name with a one-line summary:
   <https://jejjohnson.github.io/geotoolz/capabilities/>. Or search the
   installed version directly:

   ```python
   import importlib, inspect, pkgutil
   for pkg in ("geocatalog", "geoproducts", "geopatcher", "geotoolz", "geocloud"):
       try:
           root = importlib.import_module(pkg)
       except ImportError:
           continue  # that part of the stack is not installed
       for info in [None, *pkgutil.walk_packages(root.__path__, pkg + ".")]:
           name = pkg if info is None else info.name
           if "._" in name:
               continue
           try:
               mod = importlib.import_module(name)
           except ImportError:
               continue  # a module behind an extra that is not installed
           for attr in getattr(mod, "__all__", []):
               try:
                   obj = getattr(mod, attr)
               except (AttributeError, ImportError):
                   continue  # a lazy name behind an extra that is not installed
               doc = (inspect.getdoc(obj) or "").split("\n")[0]
               print(f"{name}.{attr}: {doc}")
   ```

2. Compose what exists. If something is *almost* there, configure or wrap it
   (a `pipekit.Lambda`, an operator parameter) before writing a replacement.

## Which package does what

| Step | Package (pip) | Use |
|---|---|---|
| discover scenes in STAC / CMR / earthaccess | `geotoolz-catalog` | `geocatalog.sources` (`STACSource`, `from_stac_search`) |
| index files, query by area + time | `geotoolz-catalog` | `geocatalog.build.build_raster_catalog`, `catalog.query(GeoSlice(...))` |
| read pixels on a target grid | `geotoolz-catalog` | `geocatalog.load.load_raster` / `load_raster_timeseries` / `load_xarray` |
| copy remote assets locally | `geotoolz-catalog` | `geocatalog.staging.stage` |
| read a sensor product (GOES, Himawari, …) | `geotoolz-products` | `geoproducts.<sensor>.Reader`, `.aws` download helpers, `geoproducts.stack` |
| tile, process, stitch with overlap blending | `geotoolz-patcher` | `geopatcher.SpatialPatcher` + `spatial.geometry/sampler/window/aggregation` |
| band math, masks, radiometry, ML on rasters | `geotoolz` | operator families (`gz.NDVI`, `gz.DNToReflectance`, `gz.qa.*`, `gz.learn.*`, …) |
| write COG / GeoTIFF / Zarr | `geotoolz` | `gz.WriteCOG`, `gz.WriteGeoTIFF`, `gz.WriteZarr` |
| s3:// gs:// az:// https:// reads, COG windows | `geotoolz-cloud` | `geocloud.store.get_obstore`, `geocloud.cog.CogSource` |
| list, download, upload, copy, sync, sign objects | `geotoolz-cloud` | `geocloud.files` (`ls`, `download`, `upload`, `copy`, `sync`, `sign`) |

Everything is `georeader.GeoTensor` in and out, and every operator is a
`pipekit.Operator`, so steps chain with `gz.Sequential([...])` or `|`.

## The canonical pipeline

```python
from glob import glob

import numpy as np
import pandas as pd
from georeader.geotensor import GeoTensor

import geopatcher as gp
import geotoolz as gz
from geocatalog import GeoSlice
from geocatalog.build import build_raster_catalog
from geocatalog.patch import field_for
from geocatalog.staging import stage
from geotoolz.patch_ops import ApplyToChips, GridSampler, MergePatches

# 1. Index once — local paths or s3:// / gs:// / https:// URIs.
catalog = build_raster_catalog(
    sorted(glob("scenes/*.tif")), filename_regex=r"_(?P<date>\d{8})", date_format="%Y%m%d"
)

# 2. Ask for an area and a time window; only overlapping files are opened.
aoi = GeoSlice(
    bounds=(500_000.0, 4_000_000.0, 506_000.0, 4_006_000.0),
    interval=pd.Interval(pd.Timestamp("2024-06-10"), pd.Timestamp("2024-06-20"), closed="both"),
    resolution=(10.0, 10.0),
    crs="EPSG:32630",
)
# field_for reads local files: stage copies remote URIs into ./cache
# (local paths pass through), then the rows mosaic onto the AOI grid.
field = field_for(stage(catalog.query(aoi), dest="cache"), aoi)  # RasterField, (4, 600, 600)

# 3. Tile → compute → stitch as one pipekit pipeline.
patcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(256, 256)),
    sampler=gp.spatial.sampler.RegularStride(step=(192, 192)),   # 64 px overlap
    window=gp.spatial.window.Hann(),                              # seamless blending
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
ndvi = gz.Sequential([gz.DNToReflectance(scale=1e-4), gz.NDVI(nir=3, red=2)])

# The merge grid fixes the output shape: NDVI turns 4 bands into 1, so merge
# into a 1-band grid on the field's transform / CRS (not field.domain).
_, height, width = field.domain.shape
out_grid = GeoTensor(
    np.zeros((1, height, width), np.float32),
    transform=field.domain.transform, crs=field.domain.crs,
)
pipeline = gz.Sequential([
    GridSampler(patcher=patcher),
    ApplyToChips(operator=ndvi),
    MergePatches(aggregation=gp.spatial.aggregation.OverlapAdd(), domain=out_grid),
])
ndvi_scene = pipeline(field)                          # (1, 600, 600) float, NaN nodata

# 4. Write. The merge returns a plain array: put it back on the merge grid,
# declaring its NaN gaps (invalid or uncovered cells) as the nodata.
gz.WriteCOG(path="ndvi.tif")(
    GeoTensor(ndvi_scene, transform=out_grid.transform, crs=out_grid.crs, fill_value_default=np.nan)
)
```

When the per-patch operator keeps the band count, merging into
`field.domain` is fine; when it changes it, merge into a grid with the
output's band count, as above.

## Your own operators: the two contracts

Everything above composes because each piece keeps two contracts. A step you
write yourself has to keep them too, or it will run alone and break inside a
pipeline.

**pipekit `Operator` (how steps compose):**

- **Subclass and implement.** Subclass `pipekit.Operator` and implement
  `_apply`; never override `__call__`. It runs `_apply` on data and builds a
  graph node on a `pipekit.Input`, so the step works eagerly, in `a | b`
  and in a `pipekit.Graph`.
- **Keyword-only constructor that holds configuration only.** Use
  `__init__(self, *, ...)` and store each argument under its own name.
  `get_config()` is then derived for you, must be JSON, and
  `Operator.from_state(op.state)` rebuilds the step. Never pass a raster
  to the constructor: a second raster is a positional `_apply` argument,
  so a `Graph` can wire it in.
- **Flag the special cases.**
  - Set `forbid_in_yaml = True` when the step holds a live object (a
    model, a callable, an open handle).
  - Set `_terminal = True` when it returns something other than a raster
    (a table, a number, `None`); `Sequential` then accepts it only as the
    last step.
- **Don't rewrite pipekit.** It already has `Retry`, `Try` / `Coalesce`,
  `Branch` / `Switch`, `Cache` / `Memoize`, `Tap` / `Snapshot`,
  `AssertShape` / `AssertDType`, `ThreadMap` / `ProcessMap` /
  `BatchedMap`, and `dumps` / `loads`.

**`georeader.GeoTensor` (what flows between steps):**

- **Shape.** A raster is `(C, H, W)`, or `(T, C, H, W)` for a time stack.
  The spatial axes are the last two and the bands are on axis `-3`, never
  `0`.
- **Same carrier out.** A GeoTensor input returns a GeoTensor on the same
  `transform` and `crs`; a plain array returns a plain array. Never mutate
  the input.
- **Fresh, consistent `attrs`.** Every output gets a new `attrs` dict. Each
  per-band key (`band_names`, `wavelengths`, …) has one entry per output
  band, or is dropped. Write band names under `band_names`.
- **Nodata.** `fill_value_default` *is* nodata. Keep invalid pixels
  invalid, and pick the output fill by meaning: `False` for masks, `0` for
  labels and counts, `NaN` for new float quantities.
- **Several rasters.** They must share a grid: the same `(H, W)`, CRS and
  transform. Put them on one grid first (`geocatalog.load.load_raster`
  onto a `GeoSlice`, or georeader's `read_reproject_like`).
- **Masks.** `True` means drop the pixel; this is what `gz.ApplyMask`
  expects.

`geotoolz.carrier` holds the helpers the built-in operators use to keep these
rules, so your step keeps them the same way:

```python
import numpy as np
from pipekit import Operator

from geotoolz.carrier import mask_invalid_to_nan, over_frames, wrap_like


class ZScore(Operator):
    """Standardise each pixel against a fixed mean and std (bands kept)."""

    def __init__(self, *, mean: float, std: float) -> None:  # keyword-only, config only
        self.mean = mean  # stored under its own name → get_config() for free
        self.std = std

    @over_frames  # a (T, C, H, W) stack runs frame by frame
    def _apply(self, gt):  # (C, H, W) → (C, H, W) float
        values = mask_invalid_to_nan(gt)  # a float copy; an invalid pixel is NaN in every band
        out = (values - self.mean) / self.std
        return wrap_like(gt, out, fill_value_default=np.nan)  # same carrier, fresh attrs, NaN fill
```

Check a new step with the same checks geotoolz runs on its own operators. It
fails naming the first broken rule:

```python
from geotoolz.testing import check_operator


def test_zscore(scene):  # scene: GeoTensor (4, H, W) with band_names and a few nodata pixels
    out = check_operator(ZScore(mean=0.3, std=0.1), scene)
    assert out.shape == scene.shape
```

## Anti-patterns — use the stack instead

| Don't write… | Use |
|---|---|
| a `for` loop over windows with manual overlap handling | `geopatcher.SpatialPatcher` + a `spatial.window` + `OverlapAdd` |
| `rasterio.open` / `merge` over many files to build a mosaic | `geocatalog` (`build_raster_catalog` → `query` → `load_raster`) |
| a STAC / CMR search client | `geocatalog.sources` |
| boto3 / s3fs / obstore client code, upload / download / copy loops | `geocloud.files` (URI verbs on the pooled client), `geocloud.store.get_obstore`, or `geocatalog.staging.stage` |
| band arithmetic on raw arrays (`(nir - red) / (nir + red)`) | the operator (`gz.NDVI`, `gz.spectral.BandRatio`, …) — it handles nodata, dtype and georeferencing |
| manual nodata masks or rebuilding a `GeoTensor` by hand | the operators carry `fill_value_default` through; in your own step, `geotoolz.carrier` (`wrap_like`, `mask_invalid_to_nan`, `resolve_band`, `require_grid_match`) |
| retry loops, `try` / `except` fallbacks, memo dicts or thread pools around steps | `pipekit` `Retry`, `Try` / `Coalesce`, `Cache`, `ThreadMap` / `BatchedMap` |
| a parser for GOES / Himawari files | `geoproducts.goes`, `geoproducts.himawari` |

If the stack genuinely lacks something you need, keep your addition small
and shaped like the stack (the two contracts above) so it composes — and consider
proposing it upstream at <https://github.com/jejjohnson/geotoolz>.
