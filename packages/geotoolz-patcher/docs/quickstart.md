# Quickstart — Lake Tahoe Sentinel-2 in 15 minutes

This walkthrough applies a per-patch operator to a real Sentinel-2 scene
over Lake Tahoe and stitches the result back into a global field. It is
the patcher-focused slice of the canonical cross-repo scenario:

> **Cloud-free Sentinel-2 NDVI over Lake Tahoe, summer 2024.**

The full end-to-end story (catalog → operators → patcher) lives at
[`geocatalog/docs/notebooks/end_to_end_lake_tahoe.ipynb`](https://github.com/jejjohnson/geocatalog/blob/main/docs/notebooks/end_to_end_lake_tahoe.ipynb).
This page covers only the **patcher** slice — the same code is also
shipped as a runnable notebook at
[`docs/notebooks/patcher_lake_tahoe.ipynb`](notebooks/patcher_lake_tahoe.ipynb).

## Setup

```bash
pip install 'geopatcher[xarray-raster,streaming]'
```

You also need `rioxarray`, `pystac-client`, and `planetary-computer` on
the path to resolve a real Sentinel-2 asset URL. If you already have a
local GeoTIFF, skip straight to the [`RasterField` step](#3-wrap-as-a-rasterfield).

## The canonical scenario

| Parameter | Value |
|---|---|
| AOI bbox (EPSG:4326) | `(-120.25, 38.85, -119.85, 39.30)` |
| Date range | `2024-06-01` / `2024-09-30` |
| STAC root | `https://planetarycomputer.microsoft.com/api/stac/v1` |
| Collection | `sentinel-2-l2a` |
| Cloud cover | `< 20 %` |
| Asset | `B04` (Red) — for the patcher demo we use a single band |

## 1. Resolve a Sentinel-2 asset

```python
import planetary_computer
import pystac_client

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = (-120.25, 38.85, -119.85, 39.30)

catalog = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=BBOX,
    datetime="2024-06-01/2024-09-30",
    query={"eo:cloud_cover": {"lt": 20}},
)
items = sorted(search.items(), key=lambda it: it.properties["eo:cloud_cover"])
item = items[0]
red_href = item.assets["B04"].href
print(item.id, "cloud cover:", item.properties["eo:cloud_cover"])
```

## 2. Open the COG lazily

```python
import rioxarray

red = rioxarray.open_rasterio(red_href, masked=True)
print(red.shape, red.rio.crs)
```

## 3. Wrap as a `RasterField`

```python
from georeader.rio_xarray_reader import RioXarrayReader

import geopatcher as gp

reader = RioXarrayReader(red)
field = gp.RasterField(reader)
```

`geopatcher.RasterField` is a one-attribute dataclass that adapts
anything satisfying the `georeader.GeoData` Protocol — `RasterioReader`,
`AsyncGeoTIFFReader`, `RioXarrayReader`, in-memory `GeoTensor` — to the
Patcher's `Field` Protocol.

## 4. Compose a `SpatialPatcher`

256×256 patches with 32-pixel overlap on each side, Hann-tapered for
clean seams, accumulated with `SpatialOverlapAdd`:

```python
patcher = gp.SpatialPatcher(
    geometry    = gp.SpatialRectangular(size=(256, 256)),
    sampler     = gp.SpatialRegularStride(step=(224, 224)),  # 32-px overlap
    window      = gp.SpatialHann(),
    aggregation = gp.SpatialOverlapAdd(),
)
```

`step = size - overlap`. A `SpatialHann` window has zero weight at the
patch boundary, so OverlapAdd's normalisation hides the seams.

## 5. Per-patch operator — channel normalisation

A toy operator that z-score normalises each patch (subtract mean,
divide by stddev) — enough to show the shape-preserving plumbing
without depending on a trained model:

```python
import dataclasses
import numpy as np


def normalise(arr: np.ndarray) -> np.ndarray:
    """Center and scale each patch to unit variance."""
    a = np.asarray(arr, dtype=np.float32)
    mu = np.nanmean(a)
    sigma = np.nanstd(a) + 1e-6
    return (a - mu) / sigma


outputs = []
for patch in patcher.split(field):
    new_data = normalise(patch.data)
    outputs.append(patch.with_data(new_data))
```

Streaming is the default — `patcher.split` returns an `Iterator[Patch]`
so memory stays bounded regardless of scene size. Call
`list(patcher.split(field))` for the eager case.

## 6. Stitch with `SpatialOverlapAdd`

```python
stitched = patcher.merge(outputs, field.domain)
print(stitched.shape, stitched.dtype)
```

For a >1 TB output that won't fit in RAM, swap the in-memory aggregation
for the disk-backed one **and** stream the patches in — never build the
full `outputs` list. The `normalise → with_data` step is now done inline
so only one patch is alive at a time:

```python
agg = gp.SpatialOverlapAdd(
    streaming=True,
    target_path="out/tahoe.zarr",
    chunks=(256, 256),
)
stitched_zarr = agg.merge(
    (patch.with_data(normalise(patch.data)) for patch in patcher.split(field)),
    field.domain,
)
```

The generator expression means at most one patch worth of bytes lives
in RAM during the merge — bounded regardless of scene size. See
[`recipes/streaming-overlap-add.md`](recipes/streaming-overlap-add.md)
for the full pattern.

## 7. Inspect the result

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.asarray(field.domain)[0], cmap="viridis")
axes[0].set_title(f"Input — Sentinel-2 B04\n{field.domain.shape}")
axes[1].imshow(np.asarray(stitched), cmap="viridis")
axes[1].set_title(f"Per-patch normalised\n{stitched.shape}")
plt.show()
```

## What you've just built

- A bounded-memory pipeline that splits a Sentinel-2 scene into 256×256
  patches with 32-pixel overlap.
- A per-patch operator that runs independently on each chip.
- A `SpatialOverlapAdd` reconstruction that hides seams thanks to the
  `SpatialHann` window.
- The same code swaps to disk-backed streaming by changing two lines of
  the aggregation config.

## Next steps

- **Concepts** — read [`concepts.md`](concepts.md) for the full four-axis
  abstraction, boundary policies, and determinism contracts.
- **Recipes:**
    - [Streaming overlap-add](recipes/streaming-overlap-add.md) — bounded-memory pipelines for >1 TB outputs.
    - [On-error policies](recipes/on-error-policies.md) — raise / skip / mask / retry.
    - [Journal & resume](recipes/journal-and-resume.md) — `PatchJournal` for restartable jobs.
- **Notebook:** [`notebooks/patcher_lake_tahoe.ipynb`](notebooks/patcher_lake_tahoe.ipynb) — runnable mirror of this walkthrough.
- **Full end-to-end:** [`geocatalog/docs/notebooks/end_to_end_lake_tahoe.ipynb`](https://github.com/jejjohnson/geocatalog/blob/main/docs/notebooks/end_to_end_lake_tahoe.ipynb) — catalog → operators → patcher.
