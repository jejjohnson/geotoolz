# Global statistics

Normalise every patch by the whole scene's mean and standard deviation,
not the patch's own. `SpatialPatcher.reduce` streams one pass into a
global aggregation; `SpatialPatcher.two_pass` runs that pass and then
applies your function to every patch.

## Reduce, then apply

```python
import numpy as np
import rasterio
from geoarray import GeoArray

import geopatcher as gp

field: gp.RasterField = gp.RasterField(
    GeoArray(
        np.outer(np.linspace(0, 1, 512), np.linspace(0, 1, 512)).astype(np.float32)[None],  # (1, 512, 512) float32
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(128, 128)),
    sampler=gp.spatial.sampler.RegularStride(step=(96, 96)),
    window=gp.spatial.window.Hann(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)

stats: dict[str, float] = patcher.reduce(field, gp.spatial.aggregation.MeanStd())  # {'mean': 0.25, 'std': 0.209}
bounds: dict[str, float] = patcher.reduce(field, gp.spatial.aggregation.MinMax())  # {'min': 0.0, 'max': 1.0}


def standardise(chip: GeoArray, s: dict[str, float]) -> np.ndarray:
    """Scale one patch by the scene's statistics."""
    return (np.asarray(chip) - s["mean"]) / s["std"]  # (1, 128, 128) float32 → (1, 128, 128) float32


z: np.ndarray = patcher.two_pass(
    field,
    reduce_with=gp.spatial.aggregation.MeanStd(),
    apply=standardise,
)                                                     # (1, 512, 512) float64
```

- **Both passes go through `split`,** so `on_error`, hooks, journal,
  prefetch and backpressure apply to each, and both share one `errors`
  list.
- **Every patch is read twice.** Pass `cache=` (a
  [PatchCache](patch-cache.md)) to serve the second pass from disk.
- **Same anchors.** The anchor list is built once, so an unseeded random
  sampler places both passes identically.
- **Other merges.** `aggregation=` sets the second pass's merge; it
  defaults to the patcher's own.

`reduce` takes any streaming aggregation, including the sketches
(`ApproxQuantile`, `StreamingHistogram`, …) for a global median or
histogram — see [Streaming aggregations](../patching.md#streaming-aggregations).

## Over an async field

`AsyncSpatialPatcher.reduce` and `two_pass` are the same passes as
coroutines, over an `AsyncField` (a `fields.CogField`, say), with
`asplit`'s `journal`, `cache` and backpressure arguments:

```python
import asyncio
from typing import Any

import numpy as np
import rasterio
from geoarray import GeoArray
from rasterio.windows import Window

import geopatcher as gp


class AsyncRaster(gp.RasterField):
    """A RasterField with an awaitable read, standing in for a remote COG."""

    async def aselect(self, window: Window) -> GeoArray:
        await asyncio.sleep(0)                        # the network round trip
        return self.select(window)                    # (1, 128, 128) float32


field: AsyncRaster = AsyncRaster(
    GeoArray(
        np.outer(np.linspace(0, 1, 512), np.linspace(0, 1, 512)).astype(np.float32)[None],  # (1, 512, 512) float32
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
patcher: gp.AsyncSpatialPatcher = gp.AsyncSpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(128, 128)),
    sampler=gp.spatial.sampler.RegularStride(step=(96, 96)),
    window=gp.spatial.window.Hann(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)


async def run() -> tuple[dict[str, float], np.ndarray]:
    stats: dict[str, Any] = await patcher.reduce(
        field, gp.spatial.aggregation.MeanStd(), max_in_flight=8
    )                                                 # {'mean': 0.25, 'std': 0.209}
    z: np.ndarray = await patcher.two_pass(
        field,
        reduce_with=gp.spatial.aggregation.MeanStd(),
        apply=lambda chip, s: (np.asarray(chip) - s["mean"]) / s["std"],
        max_in_flight=8,
    )                                                 # (1, 512, 512) float64
    return stats, z


stats, z = asyncio.run(run())
```

`apply` runs on the event loop: keep it cheap, or offload it yourself.
