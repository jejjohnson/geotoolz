# Quickstart — Lake Tahoe NDVI in 15 minutes

This page is the markdown mirror of [`notebooks/operators_lake_tahoe.ipynb`](notebooks/operators_lake_tahoe.ipynb).
It walks through a small operator-composition pipeline against one
Sentinel-2 scene from Microsoft Planetary Computer, the same canonical
scenario used across the [`geocatalog`](https://github.com/jejjohnson/geotoolz/tree/main/packages/geotoolz-catalog)
and [`geopatcher`](https://github.com/jejjohnson/geotoolz/tree/main/packages/geotoolz-patcher) repos.

**Scenario**: cloud-free Sentinel-2 NDVI over Lake Tahoe, summer 2024
(`-120.25, 38.85, -119.85, 39.30`, `2024-06-01..2024-09-30`,
`sentinel-2-l2a`, cloud cover < 20 %).

> **Why inline operators?** To show the composition pattern end to end,
> this quickstart defines `Scale`, `CloudMask`, and `NDVI` inline as
> small `Operator` subclasses. The library ships tested equivalents —
> `gz.DNToReflectance(scale=1e-4)`, `gz.S2SCL(qa_band=2)` and
> `gz.NDVI(nir=1, red=0)` — that also handle band names and nodata; swap
> them in once the pattern is clear.

## 0. Install

```bash
uv pip install \
  "pipekit @ git+https://github.com/jejjohnson/pipekit#subdirectory=packages/pipekit" \
  "geotoolz @ git+https://github.com/jejjohnson/geotoolz@main#subdirectory=packages/geotoolz"
uv pip install rioxarray planetary-computer pystac-client matplotlib
```

The full multi-repo flow (catalog → patch → operate) is documented in
the canonical
[`docs/catalog/notebooks/end_to_end_lake_tahoe.ipynb`](https://github.com/jejjohnson/geotoolz/blob/main/docs/catalog/notebooks/end_to_end_lake_tahoe.ipynb).
Here we focus on the **operator-composition slice**.

## 1. Load one Sentinel-2 scene

We dodge the STAC search step for the quickstart and load one scene
directly via its MPC asset URL through `rioxarray`. (The notebook shows
a STAC variant.)

```python
import planetary_computer
import rioxarray
import xarray as xr
from georeader.geotensor import GeoTensor

# A representative Lake Tahoe scene, summer 2024. These URLs are
# placeholders — replace with the asset hrefs from a STAC search
# (see notebooks/operators_lake_tahoe.ipynb for the full search).
B04_URL = "https://…/T10SEH_20240715T184921_B04_10m.tif"  # Red
B08_URL = "https://…/T10SEH_20240715T184921_B08_10m.tif"  # NIR
SCL_URL = "https://…/T10SEH_20240715T184921_SCL_20m.tif"  # Cloud classes

# planetary_computer signs the URL with a short-lived SAS token.
# Repeat the same pattern for B08 (NIR) and SCL (cloud classes); the
# notebook does this end-to-end — here we sketch the shape.
b04 = rioxarray.open_rasterio(planetary_computer.sign(B04_URL))  # (1, H, W)
b08 = rioxarray.open_rasterio(planetary_computer.sign(B08_URL))  # (1, H, W)
scl = rioxarray.open_rasterio(planetary_computer.sign(SCL_URL))  # (1, H, W)

# Stack red + NIR + SCL into a single (C, H, W) DataArray.
scene = xr.concat([b04, b08, scl], dim="band")
gt = GeoTensor(
    values=scene.values,
    transform=scene.rio.transform(),
    crs=scene.rio.crs,
)
```

The `GeoTensor` carries the array plus `transform` and `crs`. Operators
preserve those by rewrapping their result with
`geotoolz.carrier.wrap_like(gt, new_array)`.

## 2. Define three operators inline

```python
import numpy as np
from pipekit import Operator
from geotoolz.carrier import wrap_like


class Scale(Operator):
    """DN → reflectance via a single scale factor."""

    def __init__(self, *, scale: float = 1e-4) -> None:
        self.scale = scale

    def _apply(self, gt):
        return wrap_like(gt, np.asarray(gt, dtype=np.float32) * self.scale)


class CloudMask(Operator):
    """Boolean drop-mask from a Sentinel-2 SCL band.

    Marks SCL classes 3 (cloud shadow), 8 (cloud-medium), 9 (cloud-high),
    10 (thin cirrus) as ``True`` — i.e. *True-to-drop*, matching the
    convention used by ``geotoolz.mask.ApplyMask`` / ``geotoolz.qa.MaskClouds``.
    """

    DROP_CLASSES = (3, 8, 9, 10)

    def __init__(self, *, qa_band: int = 2) -> None:
        self.qa_band = qa_band

    def _apply(self, gt):
        drop = np.isin(np.asarray(gt)[self.qa_band], self.DROP_CLASSES)
        return wrap_like(gt, drop, fill_value_default=False)


class NDVI(Operator):
    """(NIR - Red) / (NIR + Red + eps); collapses the band axis."""

    def __init__(self, *, nir: int = 1, red: int = 0, eps: float = 1e-10) -> None:
        self.nir, self.red, self.eps = nir, red, eps

    def _apply(self, gt):
        a = np.asarray(gt, dtype=np.float32)
        nir, red = a[self.nir], a[self.red]
        return wrap_like(gt, (nir - red) / (nir + red + self.eps), fill_value_default=np.nan)
```

Each operator follows the same contract: a keyword-only constructor that
stores every argument under its own name, and an `_apply` that does the
work and rewraps the result with `wrap_like` (which keeps `transform` /
`crs` and declares the output's fill value: `False` for a mask, `NaN` for
a new float quantity). `get_config()` is derived automatically from the
constructor (`NDVI(nir=1, red=0).get_config()` is
`{"nir": 1, "red": 0, "eps": 1e-10}`), so the operator round-trips with no
extra code.

## 3. Compose

The simplest shape — a `Sequential` chain:

```python
from pipekit import Sequential

pipe = Sequential([Scale(scale=1e-4), NDVI(nir=1, red=0)])
ndvi = pipe(gt)        # GeoTensor in, GeoTensor out
```

For cloud masking before NDVI, where you need to *split* the scene into
"clear mask" and "reflectance", apply the mask, then run NDVI, reach for
`Graph`:

```python
import geotoolz as gz


class ApplyDropMask(Operator):
    """Set pixels where the drop-mask is True to NaN; keeps carrier metadata."""

    def _apply(self, gt, drop):
        # Graph supplies upstream node values as separate positional args.
        masked = np.where(np.asarray(drop), np.nan, np.asarray(gt, dtype=np.float32))
        return wrap_like(gt, masked, fill_value_default=np.nan)


img = gz.Input("image")
scaled = Scale(scale=1e-4)(img)
drop = CloudMask(qa_band=2)(img)
clean = ApplyDropMask()(scaled, drop)
ndvi = NDVI(nir=1, red=0)(clean)

g = gz.Graph(inputs={"image": img}, outputs={"ndvi": ndvi})
result = g(image=gt)
ndvi_gt = result["ndvi"]
```

## 4. Visualise

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(ndvi_gt.values, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_title("Lake Tahoe NDVI — Sentinel-2 L2A, summer 2024")
ax.axis("off")
fig.colorbar(im, ax=ax, shrink=0.7, label="NDVI")
plt.show()
```

## 5. Iterate

A researcher's typical loop:

1. **Insert a `Tap`** to log shape/range mid-pipeline:
   ```python
   import geotoolz as gz

   pipe = Sequential([Scale(), gz.Tap(lambda gt: print(gt.values.shape)), NDVI()])
   ```
2. **`Snapshot`** the intermediate to inspect later without breaking the
   chain:
   ```python
   import geotoolz as gz

   snap = gz.Snapshot()
   pipe = Sequential([Scale(), snap.at("reflectance"), NDVI()])
   _ = pipe(gt)
   refl = snap["reflectance"]
   ```
3. **`Branch`** on a runtime predicate (e.g. only reproject if the CRS is
   geographic):
   ```python
   import geotoolz as gz

   pipe = Sequential([
       gz.Branch(
           predicate=lambda g: g.crs.is_geographic,
           if_true=gz.Reproject(dst_crs="EPSG:32610"),
           if_false=gz.Identity(),
       ),
       Scale(),
       NDVI(),
   ])
   ```

## Next

- The full version of this walk-through as an executable notebook:
  [`notebooks/operators_lake_tahoe.ipynb`](notebooks/operators_lake_tahoe.ipynb).
- The cross-repo end-to-end notebook (catalog → patch → operate):
  [`docs/catalog/notebooks/end_to_end_lake_tahoe.ipynb`](https://github.com/jejjohnson/geotoolz/blob/main/docs/catalog/notebooks/end_to_end_lake_tahoe.ipynb).
- Concept overview: [Concepts](concepts.md).
- Recipes:
  - [Define an operator](how-to/define-an-operator.md)
  - [Branching pipelines](how-to/branching-pipelines.md)
  - [Integration with geocatalog & geopatcher](how-to/integration-with-geocatalog-and-geopatcher.md)
