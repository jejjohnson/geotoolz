# Async and prefetch

Overlap slow patch reads with your operator. A synchronous `split` reads
ahead on a background thread with `prefetch=N`; an async `asplit` awaits
each read on your event loop.

| Field | Use |
|---|---|
| any `Field` | `patcher.split(field, prefetch=N)` |
| an `AsyncField` — `geopatcher.fields.AsyncRasterField`, `geopatcher.fields.CogField`, or any field with an `aselect` (or async `select`) | `patcher.asplit(field)` on a `SpatialPatcher` or `AsyncSpatialPatcher` |

## Read ahead on a thread

```python
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp

field: gp.RasterField = gp.RasterField(
    GeoTensor(
        np.ones((1, 512, 512), dtype=np.float32),       # (1, 512, 512) float32
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

outputs: list[gp.Patch] = [
    p.with_data(np.asarray(p.data) * 2.0)             # (1, 128, 128) float32 → (1, 128, 128) float32
    for p in patcher.split(field, prefetch=4)          # up to 4 reads ahead
]
stitched: np.ndarray = patcher.merge(outputs, field.domain)  # (1, 512, 512) float64
```

- `prefetch=0` (the default) is the serial path.
- Reader exceptions are re-raised on your thread, so `try` / `except`
  around the loop works.
- To stop early, `break` and close (or drop) the iterator: the reader
  thread exits, even while it waits on a `max_in_flight` slot, and the
  patches it read ahead are closed.

## Await reads on an event loop

`asplit` walks the same anchors as `split` and awaits one read at a time.
It never reads ahead, so you choose the concurrency — for example with
`asyncio.gather` over a batch.

```python
import asyncio
from typing import Any

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

import geopatcher as gp


class SlowAsyncField:
    """Stand-in for a remote store: each read awaits 1 ms of I/O."""

    def __init__(self, inner: gp.RasterField) -> None:
        self.inner = inner

    @property
    def domain(self) -> Any:
        return self.inner.domain

    async def aselect(self, indexer: Window) -> GeoTensor:
        await asyncio.sleep(0.001)
        return self.inner.select(indexer)

    def with_data(self, data: np.ndarray) -> GeoTensor:
        return self.inner.with_data(data)


async def model(chip: GeoTensor) -> np.ndarray:
    """Stand-in async operator."""
    await asyncio.sleep(0.001)
    return np.asarray(chip) * 2.0                      # (1, 128, 128) float32 → (1, 128, 128) float32


async def run() -> np.ndarray:
    field: SlowAsyncField = SlowAsyncField(
        gp.RasterField(
            GeoTensor(
                np.ones((1, 512, 512), dtype=np.float32),  # (1, 512, 512) float32
                transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
                crs="EPSG:32611",
            )
        )
    )
    patcher: gp.AsyncSpatialPatcher = gp.AsyncSpatialPatcher(
        geometry=gp.spatial.geometry.Rectangular(size=(128, 128)),
        sampler=gp.spatial.sampler.RegularStride(step=(96, 96)),
        window=gp.spatial.window.Hann(),
        aggregation=gp.spatial.aggregation.OverlapAdd(),
    )
    outputs: list[gp.Patch] = []
    async for patch in patcher.asplit(field, max_in_flight=8):
        with patch:                                    # hand the slot back promptly
            outputs.append(patch.with_data(await model(patch.data)))
    return await patcher.amerge(outputs, field.domain)  # (1, 512, 512) float64


stitched: np.ndarray = asyncio.run(run())             # (1, 512, 512) float64
```

- A plain `RasterField` has only a synchronous `select`; `asplit` needs a
  field whose read is awaitable.
- `SpatialPatcher.asplit` works the same way; `AsyncSpatialPatcher` also
  makes `split` an alias of `asplit` and `patch_at` a coroutine.
- Under `max_in_flight` / `max_in_flight_bytes`, a patch may be closed on
  any thread (inside `asyncio.to_thread`, say); its slot returns to the
  loop that issued it.
