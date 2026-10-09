# Handle read failures

Keep a bulk run going when some patch reads fail — a transient HTTP 503,
a missing tile, a corrupt block. Pick the policy with
`SpatialPatcher(on_error=...)`; the
[On-error policies table](../patching.md#on-error-policies) defines each
one exactly.

| Your situation | `on_error` |
|---|---|
| Development and CI: stop on the first failure | `"raise"` (default) |
| Bulk inference that tolerates gaps | `"skip"` |
| Downstream needs a complete grid (training matrices, a stitched map with holes) | `"mask"` |
| Remote reads with transient errors (S3, HTTP COGs, remote zarr) | `"retry"` |

The examples below share a stand-in field whose reads fail on one column
of patches, so each fence runs on its own.

## Skip failures and inspect them

```python
from typing import Any

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

import geopatcher as gp


class FlakyField:
    """A RasterField whose reads fail on the column of patches at x = 128."""

    def __init__(self, inner: gp.RasterField) -> None:
        self.inner = inner

    @property
    def domain(self) -> Any:
        return self.inner.domain

    def select(self, indexer: Window) -> GeoTensor:
        if indexer.col_off == 128:
            raise OSError(f"read failed at {indexer}")
        return self.inner.select(indexer)

    def with_data(self, data: np.ndarray) -> GeoTensor:
        return self.inner.with_data(data)


field: FlakyField = FlakyField(
    gp.RasterField(
        GeoTensor(
            np.ones((1, 256, 256), dtype=np.float32),   # (1, 256, 256) float32
            transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
            crs="EPSG:32611",
        )
    )
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(64, 64)),
    sampler=gp.spatial.sampler.RegularStride(step=(64, 64)),
    window=gp.spatial.window.Boxcar(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
    on_error="skip",
    capture_traceback=False,                            # keep the records small
)

patches: list[gp.Patch] = list(patcher.split(field))  # 12 × (1, 64, 64) float32
errors: list[gp.observe.PatchErrorRecord] = patcher.errors  # 4 records
for record in errors:
    print(record.anchor, record.kind, record.message)   # (0, 128) OSError read failed at …
stitched: np.ndarray = patcher.merge(patches, field.domain)  # (1, 256, 256) float64 · NaN = skipped
```

Failed anchors never reach the iterator. `patcher.errors` holds the
latest call's failures; a `SpatioTemporalPatcher` records its chip
failures in `stp.spatial.errors`.

## Mask failures as NaN patches

`"mask"` yields a NaN patch of the geometry's shape in place of each
failure, so every anchor keeps its slot.

```python
from typing import Any

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

import geopatcher as gp


class FlakyField:
    """A RasterField whose reads fail on the column of patches at x = 128."""

    def __init__(self, inner: gp.RasterField) -> None:
        self.inner = inner

    @property
    def domain(self) -> Any:
        return self.inner.domain

    def select(self, indexer: Window) -> GeoTensor:
        if indexer.col_off == 128:
            raise OSError(f"read failed at {indexer}")
        return self.inner.select(indexer)

    def with_data(self, data: np.ndarray) -> GeoTensor:
        return self.inner.with_data(data)


field: FlakyField = FlakyField(
    gp.RasterField(
        GeoTensor(
            np.ones((1, 256, 256), dtype=np.float32),   # (1, 256, 256) float32
            transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
            crs="EPSG:32611",
        )
    )
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(64, 64)),
    sampler=gp.spatial.sampler.RegularStride(step=(64, 64)),
    window=gp.spatial.window.Boxcar(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
    on_error="mask",
)

patches: list[gp.Patch] = list(patcher.split(field))  # 16 × (1, 64, 64), 4 all-NaN
holes: int = sum(bool(np.isnan(np.asarray(p.data)).all()) for p in patches)  # 4
stitched: np.ndarray = patcher.merge(patches, field.domain)  # (1, 256, 256) float64 · NaN = masked
```

Aggregations ignore NaN samples, so a masked patch is a transparent hole.
Where overlapping patches cover it, their values fill it in.

## Retry transient errors

`"retry"` re-reads exceptions that match `retry_on` up to `max_retries`
times, then records the failure as `"skip"` does.

```python
from collections import Counter
from typing import Any

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

import geopatcher as gp


class OnceFlakyField:
    """A RasterField whose first read of each window times out."""

    def __init__(self, inner: gp.RasterField) -> None:
        self.inner = inner
        self.attempts: Counter[tuple[int, int]] = Counter()

    @property
    def domain(self) -> Any:
        return self.inner.domain

    def select(self, indexer: Window) -> GeoTensor:
        key = (indexer.row_off, indexer.col_off)
        self.attempts[key] += 1
        if self.attempts[key] == 1:
            raise TimeoutError(f"slow read at {key}")
        return self.inner.select(indexer)

    def with_data(self, data: np.ndarray) -> GeoTensor:
        return self.inner.with_data(data)


field: OnceFlakyField = OnceFlakyField(
    gp.RasterField(
        GeoTensor(
            np.ones((1, 256, 256), dtype=np.float32),   # (1, 256, 256) float32
            transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
            crs="EPSG:32611",
        )
    )
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(64, 64)),
    sampler=gp.spatial.sampler.RegularStride(step=(64, 64)),
    window=gp.spatial.window.Boxcar(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
    on_error="retry",
    max_retries=3,
    retry_on=(OSError, TimeoutError, "rasterio.errors.RasterioIOError"),
)

patches: list[gp.Patch] = list(patcher.split(field))  # 16 × (1, 64, 64) float32, each read twice
attempts: list[gp.observe.PatchErrorRecord] = patcher.errors  # 16 records, all retry_count=0
```

`patcher.errors` records every failed attempt with its `retry_count`, so
an anchor that recovered on a retry still has a record. An anchor is
lost only when it has a record with `retry_count == max_retries`.

Only I/O-shaped errors are retried by default: a `ValueError` or
`KeyError` is a bug and surfaces at once. Each failed attempt reaches the
hooks' `on_error`, so you can count retries — see
[Observability hooks](../observability.md).

## Run in parallel with a journal

`geopatcher.run.parallel_map` reads through `patcher.split`, so the
patcher's `on_error` covers reads. Its own `on_error` covers your
operator, and a `PatchJournal` makes the run resumable.

```python
from pathlib import Path

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp
from geopatcher.observe import PatchJournal
from geopatcher.run import parallel_map

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
    on_error="retry",                                   # read failures
    max_retries=3,
)


def model(chip: GeoTensor) -> np.ndarray:
    """Stand-in model that fails on one patch."""
    a: np.ndarray = np.asarray(chip)                    # (1, 64, 64) float32
    if chip.transform.c == 500_000 + 10 * 192:
        raise ValueError("model diverged")
    return a * 2.0                                      # (1, 64, 64) float64


Path("out").mkdir(exist_ok=True)
journal: PatchJournal = PatchJournal("out/run.jsonl")
outputs: list[gp.Patch] = parallel_map(
    patcher, field, model,
    n_workers=4,
    journal=journal,                                    # skip "ok" anchors, commit every result
    on_error="skip",                                    # operator failures
)                                                       # 12 × (1, 64, 64) float64
```

| Setting | Governs | Failures land in |
|---|---|---|
| `SpatialPatcher(on_error=...)` | reading each patch (`Field.select`, including batched `select_many`) | `patcher.errors` |
| `parallel_map(on_error=...)` | your operator; `"raise"` cancels every queued patch | the journal, as `"error"` rows |

Set both to `"skip"` for long unattended runs and both to `"raise"` in
CI. Rerunning with the same journal retries only anchors without an
`"ok"` row — see [Journal and resume](journal-and-resume.md).
