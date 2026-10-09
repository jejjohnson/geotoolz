# Observability hooks

Add progress bars, metrics or tracing to a patch run without changing
it. A hook is any object with some of the `geopatcher.observe.PatcherHook`
callbacks; pass a list to `split(..., hooks=[...])` or
`merge(..., hooks=[...])`.

## Callbacks

| Callback | When |
|---|---|
| `on_split_start(n_anchors)` | before the first read; `-1` when the total is not cheap to know |
| `on_patch_start(anchor[, coord_value])` | before each read |
| `on_patch_done(anchor, runtime_s, bytes_[, coord_value])` | after each read |
| `on_patch_skipped(anchor)` | instead of start / done, for a patch the journal already holds |
| `on_error(anchor, exc)` | on every failed read — re-raised or swallowed by the `on_error` policy, each retry included |
| `on_split_end()` | after the last patch |
| `on_merge_start(n_patches)` / `on_merge_end(output_bytes)` | around `merge` |

- Implement only the callbacks you need; the patcher passes as many
  arguments as each callback accepts.
- An exception inside a hook becomes a `RuntimeWarning`, so observability
  code never aborts a run.

## Count reads and bytes

```python
from typing import Any

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp


class Stats:
    """Count patches, bytes and read time."""

    def __init__(self) -> None:
        self.n: int = 0
        self.bytes: int = 0
        self.seconds: float = 0.0

    def on_patch_done(self, anchor: Any, runtime_s: float, bytes_: int) -> None:
        self.n += 1
        self.bytes += bytes_
        self.seconds += runtime_s


field: gp.RasterField = gp.RasterField(
    GeoTensor(
        np.ones((1, 256, 256), dtype=np.float32),       # (1, 256, 256) float32
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(64, 64)),
    sampler=gp.spatial.sampler.RegularStride(step=(64, 64)),
    window=gp.spatial.window.Boxcar(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
stats: Stats = Stats()
patches: list[gp.Patch] = list(patcher.split(field, hooks=[stats]))  # 16 × (1, 64, 64) float32
print(stats.n, stats.bytes)                           # 16 262144
```

## Show a progress bar

```python
from typing import Any

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor
from tqdm import tqdm

import geopatcher as gp


class TqdmHook:
    """A tqdm bar that advances once per patch."""

    def __init__(self) -> None:
        self.pbar: tqdm | None = None

    def on_split_start(self, n_anchors: int) -> None:
        self.pbar = tqdm(total=None if n_anchors < 0 else n_anchors)

    def on_patch_done(self, anchor: Any, runtime_s: float, bytes_: int) -> None:
        if self.pbar is not None:
            self.pbar.update(1)

    def on_split_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


field: gp.RasterField = gp.RasterField(
    GeoTensor(
        np.ones((1, 256, 256), dtype=np.float32),       # (1, 256, 256) float32
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(64, 64)),
    sampler=gp.spatial.sampler.RegularStride(step=(64, 64)),
    window=gp.spatial.window.Boxcar(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
for patch in patcher.split(field, hooks=[TqdmHook()]):  # 16/16 on the bar
    _ = np.asarray(patch.data).mean()                   # (1, 64, 64) float32 → () float32
```

## Trace with OpenTelemetry

One span per patch, closed on success or failure. This needs
`opentelemetry-api` and a configured tracer provider.

```python
from typing import Any

from opentelemetry import trace


class OpenTelemetryHook:
    """One span per patch read."""

    def __init__(self, tracer: trace.Tracer) -> None:
        self.tracer = tracer
        self.spans: dict[str, trace.Span] = {}

    def on_patch_start(self, anchor: Any) -> None:
        span = self.tracer.start_span("geopatcher.patch")
        span.set_attribute("geopatcher.anchor", repr(anchor))
        self.spans[repr(anchor)] = span

    def on_patch_done(self, anchor: Any, runtime_s: float, bytes_: int) -> None:
        span = self.spans.pop(repr(anchor), None)
        if span is not None:
            span.set_attribute("geopatcher.runtime_s", runtime_s)
            span.set_attribute("geopatcher.bytes", bytes_)
            span.end()

    def on_error(self, anchor: Any, exc: Exception) -> None:
        span = self.spans.pop(repr(anchor), None)
        if span is not None:
            span.record_exception(exc)
            span.end()


hook: OpenTelemetryHook = OpenTelemetryHook(trace.get_tracer(__name__))
# patcher.split(field, hooks=[hook])
```
