# Quickstart — Lake Tahoe NDVI in 15 minutes

This page is the markdown mirror of [`notebooks/operators_lake_tahoe.ipynb`](notebooks/operators_lake_tahoe.ipynb).
It walks through a small operator-composition pipeline against one
Sentinel-2 scene from Microsoft Planetary Computer, the same canonical
scenario used across the [`geocatalog`](https://github.com/jejjohnson/geocatalog)
and [`geopatcher`](https://github.com/jejjohnson/geopatcher) repos.

**Scenario**: cloud-free Sentinel-2 NDVI over Lake Tahoe, summer 2024
(`-120.25, 38.85, -119.85, 39.30`, `2024-06-01..2024-09-30`,
`sentinel-2-l2a`, cloud cover < 20 %).

> **Pre-alpha note.** Several `geotoolz` operator modules (`indices`,
> `cloud`, `radiometry`, …) are in flux. To keep this quickstart stable
> as the named-op surface churns, we define `Scale`, `CloudMask`, and
> `NDVI` inline as small `Operator` subclasses. The patterns transfer
> directly to the named imports once you adopt them.

## 0. Install

```bash
uv pip install "git+https://github.com/jejjohnson/geotoolz@main"
uv pip install rioxarray planetary-computer pystac-client matplotlib
```

The full multi-repo flow (catalog → patch → operate) is documented in
the canonical
[`geocatalog/docs/notebooks/end_to_end_lake_tahoe.ipynb`](https://github.com/jejjohnson/geocatalog/blob/main/docs/notebooks/end_to_end_lake_tahoe.ipynb).
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

# A representative Lake Tahoe scene, summer 2024.
# This URL is a placeholder — replace with the asset href from a STAC search
# (see notebooks/operators_lake_tahoe.ipynb for the full search).
B04_URL = (
    "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/"
    "10/S/EH/2024/07/15/S2A_MSIL2A_20240715T184921_R070_T10SEH_20240716T013507.SAFE/"
    "GRANULE/L2A_T10SEH_A047445_20240715T185751/IMG_DATA/R10m/T10SEH_20240715T184921_B04_10m.tif"
)

# planetary_computer signs the URL with a short-lived SAS token
b04_url_signed = planetary_computer.sign(B04_URL)
b04 = rioxarray.open_rasterio(b04_url_signed)  # (1, H, W) DataArray

# Stack red + NIR (and SCL for masking) into a single (C, H, W) DataArray
# in the notebook; here we sketch the shape.
scene = xr.concat([b04_red, b04_nir, b04_scl], dim="band")
gt = GeoTensor(
    values=scene.values,
    transform=scene.rio.transform(),
    crs=scene.rio.crs,
)
```

The `GeoTensor` carries the array plus `transform` and `crs`. Operators
preserve those across `gt.array_as_geotensor(new_array)`.

## 2. Define three operators inline

```python
from pipekit import Operator


class Scale(Operator):
    """DN → reflectance via a single scale factor."""

    def __init__(self, *, scale: float = 1e-4) -> None:
        self.scale = scale

    def _apply(self, gt):
        return gt.array_as_geotensor(gt.values.astype("float32") * self.scale)

    def get_config(self):
        return {"scale": self.scale}


class CloudMask(Operator):
    """Boolean mask of clear pixels from a Sentinel-2 SCL band.

    Keeps SCL classes 4 (vegetation), 5 (not-vegetated), 6 (water),
    7 (unclassified), 11 (snow); drops cloud / shadow classes.
    """

    KEEP_CLASSES = (4, 5, 6, 7, 11)

    def __init__(self, *, scl_idx: int = 2) -> None:
        self.scl_idx = scl_idx

    def _apply(self, gt):
        scl = gt.values[self.scl_idx]
        import numpy as np
        clear = np.isin(scl, self.KEEP_CLASSES)
        return gt.array_as_geotensor(clear.astype("uint8"))

    def get_config(self):
        return {"scl_idx": self.scl_idx}


class NDVI(Operator):
    """(NIR - Red) / (NIR + Red + eps)."""

    def __init__(self, *, nir_idx: int = 1, red_idx: int = 0, eps: float = 1e-10) -> None:
        self.nir_idx, self.red_idx, self.eps = nir_idx, red_idx, eps

    def _apply(self, gt):
        a = gt.values
        nir, red = a[self.nir_idx], a[self.red_idx]
        return gt.array_as_geotensor((nir - red) / (nir + red + self.eps))

    def get_config(self):
        return {"nir_idx": self.nir_idx, "red_idx": self.red_idx, "eps": self.eps}
```

Each operator follows the same two-method contract: `_apply` does the
work, `get_config` round-trips the constructor args.

## 3. Compose

The simplest shape — a `Sequential` chain:

```python
from pipekit import Sequential

pipe = Sequential([Scale(scale=1e-4), NDVI(nir_idx=1, red_idx=0)])
ndvi = pipe(gt)        # GeoTensor in, GeoTensor out
```

For cloud masking before NDVI, where you need to *split* the scene into
"clear mask" and "reflectance", apply the mask, then run NDVI, reach for
`Graph`:

```python
import geotoolz as gz
import numpy as np
from pipekit import Operator


class ApplyMask(Operator):
    """Zero-out pixels where mask == 0; preserves carrier metadata."""

    def _apply(self, inputs):
        gt, mask = inputs  # tuple of (reflectance, clear-mask)
        masked = gt.values * mask.values[None, :, :]
        return gt.array_as_geotensor(masked.astype(np.float32))

    def get_config(self):
        return {}


img = gz.Input("image")
scaled = Scale(scale=1e-4)(img)
clear = CloudMask(scl_idx=2)(img)
clean = ApplyMask()((scaled, clear))
ndvi = NDVI(nir_idx=1, red_idx=0)(clean)

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
   pipe = Sequential([Scale(), gz.Tap(lambda gt: print(gt.values.shape)), NDVI()])
   ```
2. **`Snapshot`** the intermediate to inspect later without breaking the
   chain:
   ```python
   snap = gz.Snapshot()
   pipe = Sequential([Scale(), snap.at("reflectance"), NDVI()])
   _ = pipe(gt)
   refl = snap["reflectance"]
   ```
3. **`Branch`** on a runtime predicate (e.g. only reproject if the CRS is
   geographic):
   ```python
   pipe = Sequential([
       gz.Branch(
           predicate=lambda g: g.crs.is_geographic,
           if_true=ReprojectToUTM(),
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
  [`geocatalog/docs/notebooks/end_to_end_lake_tahoe.ipynb`](https://github.com/jejjohnson/geocatalog/blob/main/docs/notebooks/end_to_end_lake_tahoe.ipynb).
- Concept overview: [Concepts](concepts.md).
- Recipes:
  - [Define an operator](recipes/define-an-operator.md)
  - [Branching pipelines](recipes/branching-pipelines.md)
  - [Integration with geocatalog & geopatcher](recipes/integration-with-geocatalog-and-geopatcher.md)
