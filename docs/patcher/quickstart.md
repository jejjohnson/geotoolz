# Quickstart — patch a Sentinel-2 scene

Run a per-patch operator over a real Sentinel-2 band and stitch it back
with feathered seams. The [landing-page quickstart](index.md#quickstart)
does the same on a synthetic array; the
[tutorial notebook](notebooks/patcher_lake_tahoe.ipynb) adds plots.

## Install

```bash
pip install 'geotoolz-patcher[xarray-raster]' pystac-client planetary-computer
```

## Find a scene

Search the Planetary Computer for the clearest summer-2024 Sentinel-2 L2A
scene over Lake Tahoe that is fully inside the swath.
`planetary_computer.sign_inplace` signs each asset
URL so you can read it.

```python
import planetary_computer
import pystac
import pystac_client

catalog: pystac_client.Client = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
items: list[pystac.Item] = list(
    catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=(-120.25, 38.85, -119.85, 39.30),             # Lake Tahoe, lon/lat
        datetime="2024-06-01/2024-09-30",
        query={"eo:cloud_cover": {"lt": 20}, "s2:nodata_pixel_percentage": {"lt": 1}},
    ).items()
)
item: pystac.Item = min(items, key=lambda it: it.properties["eo:cloud_cover"])
red_href: str = item.assets["B04"].href                    # signed COG URL
```

## Patch, operate, stitch

Open the red band lazily, wrap it as a `geopatcher.fields.RioXarrayField`
and run a z-score per 256-pixel patch. Each patch reads only its own
window of the COG.

```python
import numpy as np
import planetary_computer
import pystac_client
import rioxarray
import xarray as xr

import geopatcher as gp

catalog: pystac_client.Client = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
item = min(
    catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=(-120.25, 38.85, -119.85, 39.30),
        datetime="2024-06-01/2024-09-30",
        query={"eo:cloud_cover": {"lt": 20}, "s2:nodata_pixel_percentage": {"lt": 1}},
    ).items(),
    key=lambda it: it.properties["eo:cloud_cover"],
)

# Shapes: the full tile is (1, 10980, 10980); keep a 2048-pixel corner to stay quick.
red: xr.DataArray = rioxarray.open_rasterio(item.assets["B04"].href, masked=True, chunks={"x": 1024, "y": 1024})
red = red.isel(x=slice(0, 2048), y=slice(0, 2048))         # (1, 2048, 2048) float32, lazy
field: gp.fields.RioXarrayField = gp.fields.RioXarrayField(red)

patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(256, 256)),
    sampler=gp.spatial.sampler.RegularStride(step=(224, 224)),  # 32 px overlap
    window=gp.spatial.window.Hann(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)


def zscore(chip: xr.DataArray) -> np.ndarray:
    """Centre and scale one patch."""
    a: np.ndarray = np.asarray(chip, dtype=np.float32)    # (1, 256, 256) float32
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-6)     # (1, 256, 256) float32


outputs: list[gp.Patch] = [p.with_data(zscore(p.data)) for p in patcher.split(field)]  # 81 patches
stitched: xr.DataArray = patcher.merge_to_xarray(outputs, field)  # (1, 2048, 2048) float32, coords kept · NaN = no data
```

- **Overlap** is `size − step` = 32 pixels. The Hann window feathers each
  seam, and `OverlapAdd` divides by the summed weights.
- **No data.** Pixels outside the swath stay NaN, and so do the first row
  and column, which the Hann taper weights at zero — see
  [Window convention](patching.md#window-convention). Eight 224-pixel
  steps plus one patch cover 2048 pixels exactly; on other sizes, pick a
  [boundary policy](patching.md#boundary-policy) for the trailing edge.
- **Carriers.** `merge` returns a bare array; `merge_to_xarray` puts it
  back in a `DataArray` with the source's coords.

## Next steps

- Keep memory flat on the full tile: [Stream to disk](recipes/streaming-overlap-add.md).
- Run patches in parallel and survive bad reads: [Handle read failures](recipes/on-error-policies.md).
- Restart a long job where it stopped: [Journal and resume](recipes/journal-and-resume.md).
- Combine with catalog search and geotoolz operators: the
  [stack quickstart](../index.md#quickstart-catalog-patcher-operators).
